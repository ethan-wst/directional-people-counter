"""
helpers.py

Utility classes and functions for person tracking and visualization

Contains:
    TrackedPerson: Data class for a tracked person
    FrameDetections: Data class for all detections in a frame
    Track: Internal state for a single track
    draw_detections: Visualization utility
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import cv2


@dataclass
class TrackedPerson:
    """
    Represents a single person tracked in a frame (GUI)
    
    Attributes:
        track_id: Unique track identifier
        bbox: Bounding box (x1, y1, x2, y2)
        confidence: Detection confidence score
    """

    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float

    @property
    def centroid(self) -> tuple[float, float]:
        """
        Returns the center coordinate of a bbox
        """

        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
@dataclass
class FrameDetections:
    """
    Contains all tracked persons in a frame
    
    Attributes:
        frame_number: Frame index of video file
        persons: List of tracked persons
        frame_image: Optional image for visualization (GUI)
    """

    frame_number: int
    persons: list[TrackedPerson] = field(default_factory=list)
    frame_image: Optional[np.ndarray] = None

class Track:
    """
    State and logic for single tracked person
    Handles prediction, update, and continues track managment
    
    Attributes:
        track_id: Unique track identifier
        bbox: Current bounding box (x1, y1, x2, y2)
        confidence: Most recent confidence score of track
        hits: Number of consecutive matches
        misses: Number of consecutive misses
        age: Total frames since initiation
        velocity: Estimated velocity for prediction
        history: List of previous bounding boxes
    """

    def __init__(self, track_id: int, bbox: np.ndarray, confidence: float):
        self.track_id = track_id
        self.bbox = bbox
        self.confidence = confidence
        self.hits = 1
        self.misses = 0
        self.age = 1
        self.velocity = np.array([0.0, 0.0, 0.0, 0.0])
        self.history = [bbox.copy()]

    @property
    def centroid(self) -> tuple[float, float]:
        """
        Returns the center coordinate of the bbox
        """

        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def is_active(self, max_misses: int = 30) -> bool:
        """
        True if misses < max_misses
        """

        return self.misses < max_misses

    def predict(self) -> np.ndarray:
        """
        Predicts the next bbox position using velocity
        """

        return self.bbox + self.velocity

    def update(self, bbox: np.ndarray, confidence: float):
        """
        Updates the track with new detection

        Args:
            bbox: New bounding box
            confidence: Detection confidence
        """

        bbox_delta = bbox - self.bbox
        self.velocity = 0.7 * self.velocity + 0.3 * bbox_delta
        self.bbox = bbox
        self.confidence = confidence
        self.hits += 1
        self.misses = 0
        self.age += 1
        self.history.append(bbox.copy())

    def mark_missed(self):
        """
        Marks track as missed for current frame and predicts next position
        """

        self.bbox = self.predict()
        self.misses += 1
        self.age += 1


def draw_detections(
    frame: np.ndarray,
    detections: FrameDetections
) -> np.ndarray:
    """
    Draws tracked detections on a frame

    Args:
        frame: Image to draw on
        detections: Detections to visualize
    Returns:
        np.ndarray: The frame with visualizations
    """

    frame = frame.copy()
    for person in detections.persons:
        # Get bounding box & centroid
        x1, y1, x2, y2 = [int(v) for v in person.bbox]
        cx, cy = [int(v) for v in person.centroid]

        # Draw bounding box 
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label
        label = f"ID:{person.track_id} ({person.confidence:.2f})"
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            frame, 
            (x1, y1 - text_height - 10), 
            (x1 + text_width, y1),
            (0, 255, 0), -1
        )
        cv2.putText(
            frame,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        # Draw centroid
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    return frame


