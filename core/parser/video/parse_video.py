import cv2
import base64
from openai import OpenAI
import assemblyai as aai
import logging
from core.models.video import TimeSeriesData, ParseVideoResult
import tomli
import os
from typing import Optional, Dict, Any
from ollama import AsyncClient
from core.parser.video.track_video import track_video

logger = logging.getLogger(__name__)


def debug_object(title, obj):
    logger.debug("\n".join(["-" * 100, title, "-" * 100, f"{obj}", "-" * 100]))


def load_config() -> Dict[str, Any]:
    config_path = os.path.join(os.path.dirname(__file__), "../../../databridge.toml")
    with open(config_path, "rb") as f:
        return tomli.load(f)


class VisionModelClient:
    def __init__(self, config: Dict[str, Any]):
        self.config = config["parser"]["vision"]
        self.provider = self.config.get("provider", "ollama")
        self.model_name = self.config.get("model_name", "llama3.2-vision")

        if self.provider == "openai":
            self.client = OpenAI()
        elif self.provider == "ollama":
            base_url = self.config.get("base_url", "http://localhost:11434")
            self.client = AsyncClient(host=base_url)
        else:
            raise ValueError(f"Unsupported vision model provider: {self.provider}")

    async def get_frame_description(self, image_base64: str, context: str) -> str:
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": context},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                            },
                        ],
                    }
                ],
                max_tokens=300,
            )
            return response.choices[0].message.content
        else:  # ollama
            response = await self.client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": context, "images": [image_base64]}],
            )
            return response["message"]["content"]


class VideoParser:
    def __init__(
        self,
        video_path: str,
        assemblyai_api_key: str,
        frame_sample_rate: Optional[int] = None,
        track_every_n_frames: Optional[int] = None,
        tracking_confidence_threshold: float = 0.3,
    ):
        """
        Initialize the video parser

        Args:
            video_path: Path to the video file
            assemblyai_api_key: API key for AssemblyAI
            frame_sample_rate: Sample every nth frame for description (optional, defaults to config value)
            track_every_n_frames: Sample every nth frame for tracking (optional, defaults to 1)
            tracking_confidence_threshold: Minimum confidence for object detections
        """
        logger.info(f"Initializing VideoParser for {video_path}")
        self.config = load_config()
        self.video_path = video_path
        self.frame_sample_rate = frame_sample_rate or self.config["parser"]["vision"].get(
            "frame_sample_rate", 120
        )
        self.track_every_n_frames = track_every_n_frames or 1
        self.tracking_confidence_threshold = tracking_confidence_threshold

        # Initialize video capture for basic properties
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            logger.error(f"Failed to open video file: {video_path}")
            raise ValueError(f"Could not open video file: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps
        self.cap.release()

        # Initialize AssemblyAI
        aai.settings.api_key = assemblyai_api_key
        aai_config = aai.TranscriptionConfig(speaker_labels=True)
        self.transcriber = aai.Transcriber(config=aai_config)
        self.transcript = TimeSeriesData(time_to_content={})

        # Initialize vision model client
        self.vision_client = VisionModelClient(self.config)

        logger.info(f"Video loaded: {self.duration:.2f}s duration, {self.fps:.2f} FPS")

    def frame_to_base64(self, frame) -> str:
        """Convert a frame to base64 string"""
        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            logger.error("Failed to encode frame to JPEG")
            raise ValueError("Failed to encode frame")
        return base64.b64encode(buffer).decode("utf-8")

    def get_transcript_object(self) -> aai.Transcript:
        """
        Get the transcript object from AssemblyAI
        """
        logger.info("Starting video transcription")
        transcript = self.transcriber.transcribe(self.video_path)
        if transcript.status == "error":
            logger.error(f"Transcription failed: {transcript.error}")
            raise ValueError(f"Transcription failed: {transcript.error}")
        if not transcript.words:
            logger.warning("No words found in transcript")
        logger.info("Transcription completed successfully!")

        return transcript

    def get_transcript(self) -> TimeSeriesData:
        """
        Get timestamped transcript of the video using AssemblyAI

        Returns:
            TimeSeriesData object containing transcript
        """
        logger.info("Starting video transcription")
        transcript = self.get_transcript_object()
        # divide by 1000 because assemblyai timestamps are in milliseconds
        time_to_text = (
            {u.start / 1000: u.text for u in transcript.utterances} if transcript.utterances else {}
        )
        debug_object("Time to text", time_to_text)
        self.transcript = TimeSeriesData(time_to_content=time_to_text)
        return self.transcript

    async def get_frame_descriptions(self) -> TimeSeriesData:
        """
        Get descriptions for sampled frames using configured vision model

        Returns:
            TimeSeriesData object containing frame descriptions
        """
        logger.info("Starting frame description generation")

        # Return empty TimeSeriesData if frame_sample_rate is -1 (captioning disabled)
        if self.frame_sample_rate == -1:
            logger.info("Frame captioning is disabled (frame_sample_rate = -1)")
            return TimeSeriesData(time_to_content={})

        frame_count = 0
        time_to_description = {}
        last_description = None
        logger.info("Starting main loop for frame description generation")
        while True:
            logger.info(f"Frame count: {frame_count}")
            ret, frame = self.cap.read()
            if not ret:
                logger.info("Reached end of video")
                break

            if frame_count % self.frame_sample_rate == 0:
                logger.info(f"Processing frame at {frame_count / self.fps:.2f}s")
                timestamp = frame_count / self.fps
                logger.debug(f"Processing frame at {timestamp:.2f}s")

                img_base64 = self.frame_to_base64(frame)

                context = f"""Describe this frame from a video. Focus on the main elements, actions, and any notable details. Here is the transcript around the time of the frame:
                ---
                {self.transcript.at_time(timestamp, padding=10)}
                ---

                Here is a description of the previous frame:
                ---
                {last_description if last_description else 'No previous frame description available, this is the first frame'}
                ---

                In your response, only provide the description of the current frame, using the above information as context.
                """

                last_description = await self.vision_client.get_frame_description(
                    img_base64, context
                )
                time_to_description[timestamp] = last_description

            frame_count += 1

        logger.info(f"Generated descriptions for {len(time_to_description)} frames")
        return TimeSeriesData(time_to_content=time_to_description)

    async def process_video(self) -> ParseVideoResult:
        """
        Process the video to get transcript, frame descriptions, and tracking data

        Returns:
            ParseVideoResult containing metadata, transcript, frame descriptions, and tracking data
        """
        logger.info("Starting full video processing")
        metadata = {
            "duration": self.duration,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "frame_sample_rate": self.frame_sample_rate,
            "track_every_n_frames": self.track_every_n_frames,
        }

        # Get transcript and frame descriptions
        transcript = self.get_transcript()
        frame_descriptions = await self.get_frame_descriptions()

        # Get tracking data
        logger.info("Starting object tracking")
        tracking_metadata = await track_video(
            video_path=self.video_path,
            track_every_n_frames=self.track_every_n_frames,
            confidence_threshold=self.tracking_confidence_threshold
        )
        logger.info("Object tracking completed")

        result = ParseVideoResult(
            metadata=metadata,
            transcript=transcript,
            frame_descriptions=frame_descriptions,
            tracking_metadata=tracking_metadata,
        )
        logger.info("Video processing completed successfully")
        return result

    def __del__(self):
        """Clean up video capture object"""
        if hasattr(self, "cap"):
            logger.debug("Releasing video capture resources")
            self.cap.release()
