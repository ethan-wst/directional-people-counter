"""
person_tracker.py

YOLO-based person tracking for detecting and tracking people in video streams.
Uses bounding box IoU association to maintain track identities across frames.
"""

from typing import Optional, Iterator
from pathlib import Path
import numpy as np
import time
import cv2

from ultralytics import YOLO
from .helpers import TrackedPerson, FrameDetections, Track


# person_tracker.py
# 
# YOLO-based person tracking utility for detecting and tracking people in video streams.
# Uses bounding box IoU association to maintain track identities across frames.
# Provides metrics and track history for analysis.
    
class PersonTracker:
    """
    YOLO-based person tracker with IoU tracking algorithm.
    """
    
    def __init__(
        self, 
        model_name: str = "models/ft_yolo11s.pt",
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.35,
        device: Optional[str] = None,
        max_misses: int = 20,
        min_hits: int = 10,
        new_track_thresh: float = 0.75
    ):
        """
        Initialize the person tracker

        Args:
            model_name: Path to YOLO model
            confidence_threshold: Minimum confidence for detections
            iou_threshold: Minimum IoU for association
            device: Device for YOLO ('cpu' or 'gpu')
            max_misses: Max consecutive misses
            min_hits: Min consecutive hits
            new_track_thresh: threshold for creating new tracks
        """

        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.max_misses = max_misses
        self.min_hits = min_hits
        self.new_track_thresh = new_track_thresh
        self.model = YOLO(model_name)
        self.person_class_id = 0
        self.tracks: dict[int, Track] = {}
        self.next_track_id = 1
        self.track_history: dict[int, list[tuple[float, float]]] = {}
        self.frame_times: list[float] = []
    
    @staticmethod
    def compute_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """
        Compute Intersection over Union between two bounding boxes.

        Args:
            bbox1: First bounding box.
            bbox2: Second bounding box.
        Returns:
            float: IoU value.
        """

        # Original bounding boxes
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Intersect bounding boxes
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        # Intersect area
        inter_width = max(0, inter_x_max - inter_x_min)
        inter_height = max(0, inter_y_max - inter_y_min)
        inter_area = inter_width * inter_height
        
        # Union area
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = bbox1_area + bbox2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def detection_track_assoc(
            self,
            detections: list[tuple[np.ndarray, float]]
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """
        Associates detections to existing tracks using IoU greedy matching

        Args:
            detections: Tuples (bbox, confidence) from YOLO
        Returns:
            tuple:
                matches: List of (track_id, detection_id) pairs
                unmatched_tracks: List of track_ids with no match
                unmatched_detections: List of detection indices with no match
        """

        # No tracks to match too
        if len(self.tracks) == 0:
            return [], [], list(range(len(detections)))

        # No detections to match
        if len(detections) == 0:
            return [], list(self.tracks.keys()), []

        # IoU Matrix: tracks * detections
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        track_ids = list(self.tracks.keys())    
        
        # Fill in IoU matrix
        for t_idx, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            predicted_bbox = track.predict()
            
            # Compute IOU for predicted track and detection positions
            for d_idx, (detected_bbox, _) in enumerate(detections):
                iou_matrix[t_idx, d_idx] = self.compute_iou(predicted_bbox, detected_bbox)
        
        matches = []
        unmatched_tracks = set(range(len(track_ids)))
        unmatched_detections = set(range(len(detections)))
        
        # Find pairs above thresholds
        iou_pairs = []
        for t_idx in range(len(track_ids)):
            for d_idx in range(len(detections)):
                if iou_matrix[t_idx, d_idx] >= self.iou_threshold:
                    iou_pairs.append((iou_matrix[t_idx, d_idx], t_idx, d_idx))
        
        iou_pairs.sort(reverse=True)
        
        # Assign matches greedily (highest first)
        for _, t_idx, d_idx in iou_pairs:
            if t_idx in unmatched_tracks and d_idx in unmatched_detections:
                matches.append((track_ids[t_idx], d_idx))
                unmatched_tracks.discard(t_idx)
                unmatched_detections.discard(d_idx)
        
        unmatched_track_ids = [track_ids[i] for i in unmatched_tracks]
        
        return matches, unmatched_track_ids, list(unmatched_detections)
    
    def update_tracks(
        self, 
        detections: list[tuple[np.ndarray, float]], 
        frame_number: int
    ) -> FrameDetections:
        """
        Update tracks with new detections

        Args:
            detections: Tuples (bbox, confidence) from YOLO
            frame_number: Frame index
        Returns:
            FrameDetections: Confirmed tracks for this frame.
        """

        # Associate detections & tracks
        matches, unmatched_tracks, unmatched_detections = self.detection_track_assoc(detections)
        
        # Update matches
        for track_id, det_idx in matches:
            bbox, confidence = detections[det_idx]
            self.tracks[track_id].update(bbox, confidence)
        
        # Mark misses
        for track_id in unmatched_tracks:
            self.tracks[track_id].mark_missed()
        
        # Create new tracks for unmatched detections (only above thresh)
        for det_idx in unmatched_detections:
            bbox, confidence = detections[det_idx]
            if confidence >= self.new_track_thresh:
                track = Track(self.next_track_id, bbox, confidence)
                self.tracks[self.next_track_id] = track
                self.next_track_id += 1
        
        # Remove dead tracks
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if not track.is_active(self.max_misses):
                tracks_to_remove.append(track_id)
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
        
        # Build FrameDetections from confirmed tracks
        persons: list[TrackedPerson] = []
        for track_id, track in self.tracks.items():
            # Only include tracks with minimum hits
            if track.hits >= self.min_hits:
                person = TrackedPerson(
                    track_id=track_id,
                    bbox=tuple(track.bbox),
                    confidence=track.confidence
                )
                persons.append(person)
                
                # Update track history
                if track_id not in self.track_history:
                    self.track_history[track_id] = []
                self.track_history[track_id].append(track.centroid)
        
        return FrameDetections(
            frame_number=frame_number,
            persons=persons
        )
    
    def detect_persons(self, frame: np.ndarray) -> list[tuple[np.ndarray, float]]:
        """
        Run YOLO detection and extract person bounding boxes

        Args:
            frame: Input image.
        Returns:
            list: Tuples (bbox, confidence) for detected persons
        """

        # Run YOLO prediction
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            classes=[self.person_class_id],
            device=self.device,
            verbose=False
        )
        
        # Get bboxes from prediction
        detections = []
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy()
                confidence = float(boxes.conf[i]) if boxes.conf is not None else 0.0
                detections.append((bbox, confidence))
        
        return detections
    
    def track_video(
        self,
        video_path: Path | str,
        return_frames: bool = False
    ) -> Iterator[FrameDetections]:
        """
        Track persons in a video file
    
        Args:
            video_path: Path to the video file
            return_frames: If True, include frame images in output
        Yields:
            FrameDetections: Tracking results for each frame
        """
        
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Reset
        self.tracks.clear()
        self.track_history.clear()
        self.frame_times.clear()
        self.next_track_id = 1
        
        cap = cv2.VideoCapture(str(video_path))
        frame_number = 0
    
        while cap.isOpened():
            read, frame = cap.read()

            # No more frames
            if not read:
                break
            
            start_time = time.time()
            
            # Detect persons
            detections = self.detect_persons(frame)
            
            # Update tracks
            frame_detections = self.update_tracks(detections, frame_number)
            
            frame_time = time.time() - start_time
            self.frame_times.append(frame_time)
            
            if return_frames:
                frame_detections.frame_image = frame
            
            # Generator: sequence of results over time
            yield frame_detections
            
            frame_number += 1
        
        cap.release()
    
    def get_track_trajectory(self, track_id: int) -> list[tuple[float, float]]:
        """
        Get the list of center points for a track ID.

        Args:
            track_id: Track identifier
        Returns:
            list: List of centroid positions
        """

        return self.track_history.get(track_id, [])
    
    def get_metrics(self, min_track_length: int = 5) -> dict:
        """
        Get evaluation metrics for the tracked video
    
        Args:
            min_track_length (int): Minimum length for a track to be considered valid
        Returns:
            dict: Dictionary of tracking statistics and metrics
        """
        
        if not self.frame_times:
            return {}
        
        # Calculate FPS
        total_frames = len(self.frame_times)
        total_time = sum(self.frame_times)
        avg_time_per_frame = total_time / total_frames if total_frames > 0 else 0
        fps = 1 / avg_time_per_frame if avg_time_per_frame > 0 else 0
        
        # Get tracks and corresponding lengths
        all_tracks = len(self.track_history)
        all_track_lengths = [len(trajectory) for trajectory in self.track_history.values()]
        
        # Sort valid/invalid tracks
        valid_tracks = {
            track_id: trajectory
            for track_id, trajectory in self.track_history.items()
            if len(trajectory) >= min_track_length
        }
        valid_track_lengths = [len(trajectory) for trajectory in valid_tracks.values()]
        
        metrics = {
            'total_frames_processed': total_frames,
            'total_processing_time': total_time,
            'avg_time_per_frame': avg_time_per_frame,
            'fps': fps,
            'total_tracks': all_tracks,
            'valid_tracks': len(valid_tracks),
            'short_tracks': all_tracks - len(valid_tracks),
            'avg_track_length': np.mean(all_track_lengths) if all_track_lengths else 0,
            'avg_valid_track_length': np.mean(valid_track_lengths) if valid_track_lengths else 0,
            'max_track_length': max(all_track_lengths) if all_track_lengths else 0,
            'min_track_length': min(all_track_lengths) if all_track_lengths else 0,
        }
        
        return metrics
