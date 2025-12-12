"""
directional_counter.py

Manages the counting of tracked persons upon crossing boundary.
"""

from typing import Literal

class DirectionalCounter:
    """
    Manages directional counting

    Attributes:
        line_position: Normalized position of the couting line
        orientation: 'horizontal' or 'vertical' of boundary line
        count_direction: Direction to count crosses ('up', 'down', 'left', 'right')
        min_track_age: Minimum track age to be counted
        count: Current count
        crosses: Total crossing events
        counted_tracks: Set of trackIDs counted
        track_ages: TrackID to trajectory length mapping
    """
    
    def __init__(
        self,
        line_position: float = 0.5,
        orientation: str = 'vertical',
        count_direction: str = 'right',
        min_track_age: int = 5
    ):
        """
        Initialize directional counter.

        Args:
            line_position: Normalized position of the couting line
            orientation: 'horizontal' or 'vertical' of boundary line
            count_direction: Direction to count crosses ('up', 'down', 'left', 'right')
            min_track_age: Minimum track age to be counted
        """

        self.line_position = line_position
        self.orientation = orientation
        self.count_direction = count_direction
        self.min_track_age = min_track_age
        self.count = 0
        self.crosses = 0
        self.counted_tracks = set()
        self.track_ages = {}
    
    def check_crossing(self, track_id: int, trajectory: list[tuple[float, float]]) -> bool:
        """
        Check if a track crossed boundary and in what direction

        Args:
            track_id: Track identifier
            trajectory: List of centroid positions
        Returns:
            bool: True if a crossing event was detected
        """

        # Check if track is old enough
        self.track_ages[track_id] = len(trajectory)
        if len(trajectory) < self.min_track_age:
            return False
        
        # Check if already counted
        if track_id in self.counted_tracks:
            return False
        
        # Get last two positions
        prev_x, prev_y = trajectory[-2]
        curr_x, curr_y = trajectory[-1]
        
        # Check crossing by orientation
        if self.orientation == 'horizontal':
            line_coord = self.line_position
            
            # Entry: up -> down
            if self.count_direction == 'down':
                if prev_y < line_coord <= curr_y:
                    self.counted_tracks.add(track_id)
                    self.count += 1
                    self.crosses += 1
                    return True
                if prev_y > line_coord >= curr_y:
                    self.counted_tracks.add(track_id)
                    self.count -= 1
                    self.crosses += 1
                    return True
            
            # Entry: down -> up
            elif self.count_direction == 'up':
                if prev_y > line_coord >= curr_y:
                    self.counted_tracks.add(track_id)
                    self.count += 1
                    self.crosses += 1
                    return True
                if prev_y < line_coord <= curr_y:
                    self.counted_tracks.add(track_id)
                    self.count -= 1
                    self.crosses += 1
                    return True
                
        
        elif self.orientation == 'vertical':
            line_coord = self.line_position
            # Entry: right -> left
            if self.count_direction == 'right':
                if prev_x < line_coord <= curr_x:
                    self.counted_tracks.add(track_id)
                    self.count += 1
                    self.crosses += 1
                    return True
                if prev_x > line_coord >= curr_x:
                    self.counted_tracks.add(track_id)
                    self.count -= 1
                    self.crosses += 1
                    return True
                
            # Entry: left -> right
            elif self.count_direction == 'left':
                if prev_x > line_coord >= curr_x:
                    self.counted_tracks.add(track_id)
                    self.count += 1
                    self.crosses += 1
                    return True
                if prev_x < line_coord <= curr_x:
                    self.counted_tracks.add(track_id)
                    self.count -= 1
                    self.crosses += 1
                    return True
        return False
    
    def get_stats(self) -> dict:
        """
        Get current statistics and configuration

        Returns:
            dict: Dictionary of statistics and configuration values
        """

        return {
            'total_count': self.count,
            'total_crosses': self.crosses,
            'unique_counted_tracks': len(self.counted_tracks),
            'active_tracks': len(self.track_ages),
            'min_track_age': self.min_track_age,
            'line_position': self.line_position,
            'orientation': self.orientation,
            'count_direction': self.count_direction
        }
