"""Non-GUI implementation of directionaly aware people counter, basis for GUI implementation."""

from pathlib import Path
from typing import Literal
import sys
import argparse
import cv2

from utils.person_tracker import PersonTracker
from utils.helpers import FrameDetections, TrackedPerson
from utils.directional_counter import DirectionalCounter


def run_counter(
    video_path: Path | str,
    model_name: str = "models/ft_yolo11s.pt",
    confidence_threshold: float = 0.4,
    line_position: float = 0.5,
    orientation: str = 'vertical',
    count_direction: str = 'right',
    min_track_age: int = 5
):
    """
    Run directional counter on a video without GUI
    
    Args:
        video_path: Path to video file
        model_name: Path to YOLO model file
        confidence_threshold: Minimum detection confidence
        line_position: Normalized position of the couting line
        orientation: 'horizontal' or 'vertical' of boundary line
        count_direction: Direction to count crosses ('up', 'down', 'left', 'right')
        min_track_age: Minimum track age to be counted
    """

    video_path = Path(video_path)
    
    # Initialize tracker
    tracker = PersonTracker(
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        iou_threshold=0.3
    )
    
    # Initialize tracker
    counter = DirectionalCounter(
        line_position=line_position,
        orientation=orientation,
        count_direction=count_direction,
        min_track_age=min_track_age
    )
    
    print(f"Processing video: {video_path}")
    print(f"Counting line: {orientation} at {line_position:.2f}")
    print(f"Count direction: {count_direction}")
    print(f"Minimum track age: {min_track_age} frames\n")
    
    # Get video statistics
    cap = cv2.VideoCapture(str(video_path))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    
    # Iterate over video frame
    frame_count = 0
    for detections in tracker.track_video(video_path, return_frames=False):
        frame_count += 1
        
        for person in detections.persons:
            # Get track history
            trajectory = tracker.get_track_trajectory(person.track_id)
            
            # Normalize cordinates (0.0-1.0)
            normalized_trajectory = [
                (x / w, y / h) for x, y in trajectory
            ]
            
            # Check if barrier is crossed
            counter.check_crossing(person.track_id, normalized_trajectory)
        
        if frame_count % 500 == 0:
            print(f"Processed {frame_count} frames - Current count: {counter.count} - Total crosses {counter.crosses}")
    
    print(f"\nProcessed {frame_count} frames total")
    
    tracker_metrics = tracker.get_metrics(min_track_length=10)
    counter_stats = counter.get_stats()
    
    # Info printout
    print("\nResults:")
    
    print(f"\nLine Position: {counter_stats['line_position']:.2f} ({counter_stats['orientation']})")
    print(f"Count Direction: {counter_stats['count_direction']}")
    print(f"Minimum Track Age: {counter_stats['min_track_age']} frames")
    
    print(f"\nFinal Count: {counter_stats['total_count']}")
    print(f"Total Crosses: {counter_stats['total_crosses']}")
    print(f"Unique Tracks Counted: {counter_stats['unique_counted_tracks']}")
    
    print(f"\nTotal Frames: {tracker_metrics['total_frames_processed']}")
    print(f"Processing Time: {tracker_metrics['total_processing_time']:.2f}s")
    print(f"FPS: {tracker_metrics['fps']:.2f}")
    print(f"Avg Time per Frame: {tracker_metrics['avg_time_per_frame']*1000:.2f}ms")
    print(f"Avg Track Length: {tracker_metrics['avg_track_length']:.1f} frames")


if __name__ == "__main__":
    # Default video path
    DEFAULT_VIDEO = Path(__file__).parent.parent / "data" / "mivia" / "videos" / "C_O_S_2.mkv"
    
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description="Directional People Counter - Command Line Interface",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "video_path", 
        type=str, 
        nargs="?",
        default=str(DEFAULT_VIDEO),
        help="Path to video file"
    )
    parser.add_argument(
        "--orientation", 
        type=str, 
        choices=["horizontal", "vertical"], 
        default="vertical",
        help="Orientation of the counting line"
    )
    parser.add_argument(
        "--count-direction", 
        type=str, 
        choices=["up", "down", "left", "right"], 
        default="right",
        help="Direction to count crosses"
    )
    
    args = parser.parse_args()
    
    # Validate video path
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)
    
    # Validate direction matches orientation
    if args.orientation == "horizontal" and args.count_direction not in ["up", "down"]:
        print(f"Error: For horizontal line, count direction must be 'up' or 'down'")
        sys.exit(1)
    if args.orientation == "vertical" and args.count_direction not in ["left", "right"]:
        print(f"Error: For vertical line, count direction must be 'left' or 'right'")
        sys.exit(1)
    
    # Run counter
    run_counter(
        video_path=video_path,
        model_name="models/ft_yolo11s.pt",
        confidence_threshold=0.4,
        line_position=0.5,
        orientation=args.orientation,
        count_direction=args.count_direction,
        min_track_age=3
    )
