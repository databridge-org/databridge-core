import os
import pytest
import logging
from core.parser.video.parse_video import VideoParser
from core.parser.video.track_video import track_video

logger = logging.getLogger(__name__)

"""
Test Video Requirements:
- Place a test video file at: core/tests/assets/test_video.mp4
- Duration: 5-10 seconds
- Resolution: 720p (1280x720) recommended
- Format: MP4
- Content: Should contain moving objects that YOLO can detect:
    * People
    * Vehicles (cars, bikes, etc.)
    * Animals (dogs, cats, etc.)
    * Common objects (chairs, phones, etc.)
- Motion: Objects should move across the frame to test tracking
- Scene: Well-lit scene with clear visibility
"""

TEST_VIDEO_PATH = os.path.join(os.path.dirname(__file__), "../assets/video-highlights.mp4")

@pytest.mark.asyncio
async def test_video_tracking():
    """Test that video tracking works correctly
    
    Required test video should contain moving objects that YOLO can detect.
    The test will verify:
    1. Basic detection works (objects are found)
    2. Tracking works (objects maintain IDs)
    3. Trajectory recording works (positions are logged)
    """
    # Skip if test video doesn't exist
    if not os.path.exists(TEST_VIDEO_PATH):
        pytest.skip(f"Test video not found at {TEST_VIDEO_PATH}. See core/tests/assets/README.md for requirements.")
        
    # Test direct tracking
    tracking_results = await track_video(
        video_path=TEST_VIDEO_PATH,
        track_every_n_frames=5,  # Process every 5th frame for faster processing
        confidence_threshold=0.3
    )
    
    # Verify tracking results structure
    assert isinstance(tracking_results, dict)
    assert "fps" in tracking_results
    assert "total_frames" in tracking_results
    assert "frame_width" in tracking_results
    assert "frame_height" in tracking_results
    assert "objects" in tracking_results
    assert isinstance(tracking_results["objects"], list)
    
    # Verify that objects were detected and tracked
    # Note: This requires a test video with detectable objects
    assert len(tracking_results["objects"]) > 0, "No objects detected in test video"
    
    # If objects were detected, verify their structure
    obj = tracking_results["objects"][0]
    assert "track_id" in obj
    assert "class_id" in obj
    assert "class_name" in obj
    assert "trajectory" in obj
    assert isinstance(obj["trajectory"], list)
    assert len(obj["trajectory"]) > 0, "No trajectory points recorded for tracked object"
    
    # Verify trajectory point structure
    point = obj["trajectory"][0]
    assert "frame_idx" in point
    assert "timestamp" in point
    assert "bbox" in point
    assert "confidence" in point
    assert len(point["bbox"]) == 4, "Bounding box should have 4 coordinates [x1, y1, x2, y2]"

@pytest.mark.asyncio
async def test_video_parser_integration():
    """Test that video tracking is integrated with video parser
    
    This test verifies that tracking metadata is properly integrated
    with other video processing results (transcript and frame descriptions).
    """
    # Skip if test video doesn't exist
    if not os.path.exists(TEST_VIDEO_PATH):
        pytest.skip(f"Test video not found at {TEST_VIDEO_PATH}. See core/tests/assets/README.md for requirements.")
        
    # Initialize parser with tracking enabled
    parser = VideoParser(
        video_path=TEST_VIDEO_PATH,
        assemblyai_api_key="your-api-key",  # Replace with test API key
        track_every_n_frames=5,  # Process every 5th frame
        tracking_confidence_threshold=0.3
    )
    
    # Process video
    results = await parser.process_video()
    
    # Verify basic video parsing still works
    assert results.metadata
    assert results.transcript
    assert results.frame_descriptions
    
    # Verify tracking metadata is present and properly structured
    assert results.tracking_metadata
    assert "fps" in results.tracking_metadata
    assert "total_frames" in results.tracking_metadata
    assert "objects" in results.tracking_metadata
    
    # Verify that objects were detected and tracked
    assert len(results.tracking_metadata["objects"]) > 0, "No objects detected in test video"
    
    # Verify tracked object structure
    obj = results.tracking_metadata["objects"][0]
    assert "track_id" in obj
    assert "class_name" in obj
    assert "trajectory" in obj
    assert len(obj["trajectory"]) > 0, "No trajectory points recorded for tracked object" 