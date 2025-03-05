import cv2
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from dataclasses import dataclass
from ultralytics import YOLO
import os
import json
import torch

logger = logging.getLogger(__name__)

@dataclass
class TrackedObject:
    track_id: int
    class_id: int
    class_name: str
    trajectory: List[Dict[str, Any]]  # List of {frame_idx, bbox, confidence}

class VideoTracker:
    def __init__(
        self,
        video_path: str,
        track_every_n_frames: int = 1,
        confidence_threshold: float = 0.2,  # Lower confidence threshold
        model_name: str = "yolov8x.pt"  # Use the extra large model for better accuracy
    ):
        """Initialize video tracker with YOLO

        Args:
            video_path: Path to video file
            track_every_n_frames: Process every nth frame
            confidence_threshold: Minimum confidence for detections
            model_name: YOLO model to use
        """
        self.video_path = video_path
        self.track_every_n_frames = track_every_n_frames
        self.confidence_threshold = confidence_threshold
        
        # Initialize video properties
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.cap.release()
        
        # Initialize YOLO model with improved settings
        self.model = YOLO(model_name)
        
        # Store tracked objects
        self.tracked_objects: Dict[int, TrackedObject] = {}
            
    def _update_trajectories(self, frame_idx: int, result):
        """Update object trajectories with new tracking results"""
        if result.boxes is None or len(result.boxes) == 0:
            return
            
        # Get boxes, classes, and track IDs
        boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2] format
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        track_ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else None
        
        if track_ids is None:
            return
            
        # Update each tracked object
        for box, confidence, class_id, track_id in zip(boxes, confidences, class_ids, track_ids):
            track_id = int(track_id)
            class_id = int(class_id)
            class_name = result.names[class_id]
            
            if track_id not in self.tracked_objects:
                # New object
                self.tracked_objects[track_id] = TrackedObject(
                    track_id=track_id,
                    class_id=class_id,
                    class_name=class_name,
                    trajectory=[]
                )
                
            # Add new position
            self.tracked_objects[track_id].trajectory.append({
                "frame_idx": frame_idx,
                "timestamp": frame_idx / self.fps,
                "bbox": box.tolist(),
                "confidence": float(confidence)
            })
            
    async def track(self) -> Dict[str, Any]:
        """Process video and return tracking results"""
        # Create output directory if it doesn't exist
        output_dir = os.path.join(os.path.dirname(self.video_path), "tracking_output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Get base filename without extension
        base_name = os.path.splitext(os.path.basename(self.video_path))[0]
        output_video = os.path.join(output_dir, f"{base_name}_tracked.mp4")
        output_json = os.path.join(output_dir, f"{base_name}_tracking.json")
        
        logger.info(f"Will save tracked video to: {output_video}")
        logger.info(f"Will save tracking data to: {output_json}")
        
        # Run tracking on video with improved parameters
        results = self.model.track(
            source=self.video_path,
            save=True,
            project=output_dir,
            name=f"{base_name}_tracked",
            classes=[0, 32],  # person and sports ball
            conf=0.3,
            vid_stride=self.track_every_n_frames,
            agnostic_nms=True,
            retina_masks=True,
            tracker="botsort.yaml",  # Use BoTSORT instead of ByteTrack
            device=0 if torch.cuda.is_available() else 'cpu'
        )
        
        # Process results
        for frame_idx, result in enumerate(results):
            if frame_idx % self.track_every_n_frames == 0:
                self._update_trajectories(frame_idx, result)
        
        # Prepare output
        tracked_objects = []
        for obj in self.tracked_objects.values():
            # Increase minimum trajectory length for more stable tracks
            if len(obj.trajectory) > 10:  # Filter out very short tracks
                # Calculate average confidence
                avg_conf = sum(p["confidence"] for p in obj.trajectory) / len(obj.trajectory)
                if avg_conf > 0.3:  # Only keep tracks with good average confidence
                    tracked_objects.append({
                        "track_id": obj.track_id,
                        "class_id": obj.class_id,
                        "class_name": obj.class_name,
                        "avg_confidence": float(avg_conf),
                        "trajectory": obj.trajectory
                    })
        
        # Sort objects by average confidence
        tracked_objects.sort(key=lambda x: x["avg_confidence"], reverse=True)
        
        # Save tracking data to JSON
        tracking_data = {
            "fps": self.fps,
            "total_frames": self.total_frames,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "tracking_params": {
                "model": "yolov8x",
                "confidence_threshold": self.confidence_threshold,
                "tracker": "botsort",
                "min_trajectory_length": 10,
                "classes_tracked": ["person", "sports ball"]
            },
            "objects": tracked_objects
        }
        
        with open(output_json, 'w') as f:
            json.dump(tracking_data, f, indent=2)
            
        logger.info(f"Saved tracking visualization to: {output_video}")
        logger.info(f"Saved tracking data to: {output_json}")
        logger.info(f"Tracked {len(tracked_objects)} objects: " + 
                   ", ".join(f"{obj['class_name']} (conf: {obj['avg_confidence']:.2f})" 
                           for obj in tracked_objects[:5]))
        
        return tracking_data

async def track_video(
    video_path: str,
    track_every_n_frames: int = 1,
    confidence_threshold: float = 0.3
) -> Dict[str, Any]:
    """Helper function to track objects in a video
    
    Args:
        video_path: Path to video file
        track_every_n_frames: Process every nth frame
        confidence_threshold: Minimum confidence for detections
        
    Returns:
        Dictionary containing tracking results
    """
    tracker = VideoTracker(
        video_path=video_path,
        track_every_n_frames=track_every_n_frames,
        confidence_threshold=confidence_threshold
    )
    return await tracker.track() 