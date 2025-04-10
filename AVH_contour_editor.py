import numpy as np
import cv2
import json
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Any, Optional

@dataclass
class Point:
    x: float
    y: float
    
@dataclass
class Contour:
    points: List[Point] = field(default_factory=list)
    is_selected: bool = False
    
@dataclass
class Segment:
    id: int
    contours: List[Contour] = field(default_factory=list)
    color: Tuple[float, float, float] = (0.0, 0.0, 1.0)  # RGB, default blue
    is_visible: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

class ContourEditor:
    """
    Backend for web-based contour editing functionality.
    Provides methods for manipulating contours that can be exposed via API.
    """
    def __init__(self):
        self.segments = []
        self.history = []  # For undo/redo functionality
        self.history_position = -1
        self.max_history = 20
        
    def load_segments(self, segments_data):
        """
        Load segments from SAM output or other sources.
        
        Args:
            segments_data: List of segment dictionaries
            
        Returns:
            success: Boolean indicating if loading was successful
        """
        try:
            self.segments = []
            for idx, segment in enumerate(segments_data):
                # Extract color or use default
                color = segment.get('color', (0.0, 0.0, 1.0))
                
                # Create a new segment
                new_segment = Segment(id=idx)
                new_segment.color = color
                
                # Process geometric elements
                for element in segment.get('geometric_elements', []):
                    if 'points' in element:
                        contour = Contour()
                        for x, y in element['points']:
                            contour.points.append(Point(x=x, y=y))
                        new_segment.contours.append(contour)
                
                self.segments.append(new_segment)
                
            # Save initial state in history
            self._save_to_history()
            return True
            
        except Exception as e:
            print(f"Error loading segments: {e}")
            return False
            
    def get_segments(self):
        """
        Get all segments in a format suitable for frontend.
        
        Returns:
            segments_data: List of segment dictionaries
        """
        result = []
        for segment in self.segments:
            segment_dict = {
                'id': segment.id,
                'color': segment.color,
                'is_visible': segment.is_visible,
                'geometric_elements': []
            }
            
            # Convert contours to geometric elements
            for contour in segment.contours:
                points = [(point.x, point.y) for point in contour.points]
                segment_dict['geometric_elements'].append({
                    'type': 'contour',
                    'points': points
                })
                
            result.append(segment_dict)
            
        return result
    
    def add_point(self, segment_id, contour_idx, position, point_position):
        """
        Add a new point to a contour at the specified position.
        
        Args:
            segment_id: ID of the segment
            contour_idx: Index of the contour within the segment
            position: Position in the points list to insert the new point
            point_position: (x, y) coordinates of the new point
            
        Returns:
            success: Boolean indicating if adding was successful
        """
        try:
            # Find the segment
            segment = next((s for s in self.segments if s.id == segment_id), None)
            if not segment or contour_idx >= len(segment.contours):
                return False
                
            # Get the contour
            contour = segment.contours[contour_idx]
            
            # Add the point
            x, y = point_position
            new_point = Point(x=x, y=y)
            if position >= len(contour.points):
                contour.points.append(new_point)
            else:
                contour.points.insert(position, new_point)
                
            # Save state for undo
            self._save_to_history()
            return True
            
        except Exception as e:
            print(f"Error adding point: {e}")
            return False
    
    def remove_point(self, segment_id, contour_idx, point_idx):
        """
        Remove a point from a contour.
        
        Args:
            segment_id: ID of the segment
            contour_idx: Index of the contour within the segment
            point_idx: Index of the point to remove
            
        Returns:
            success: Boolean indicating if removal was successful
        """
        try:
            # Find the segment
            segment = next((s for s in self.segments if s.id == segment_id), None)
            if not segment or contour_idx >= len(segment.contours):
                return False
                
            # Get the contour
            contour = segment.contours[contour_idx]
            
            # Ensure we have enough points (at least 3 for a closed contour)
            if len(contour.points) <= 3:
                return False
                
            # Remove the point
            if point_idx < len(contour.points):
                contour.points.pop(point_idx)
                # Save state for undo
                self._save_to_history()
                return True
                
            return False
            
        except Exception as e:
            print(f"Error removing point: {e}")
            return False
    
    def move_point(self, segment_id, contour_idx, point_idx, new_position):
        """
        Move a point to a new position.
        
        Args:
            segment_id: ID of the segment
            contour_idx: Index of the contour within the segment
            point_idx: Index of the point to move
            new_position: (x, y) coordinates of the new position
            
        Returns:
            success: Boolean indicating if moving was successful
        """
        try:
            # Find the segment
            segment = next((s for s in self.segments if s.id == segment_id), None)
            if not segment or contour_idx >= len(segment.contours):
                return False
                
            # Get the contour
            contour = segment.contours[contour_idx]
            
            # Move the point
            if point_idx < len(contour.points):
                x, y = new_position
                contour.points[point_idx] = Point(x=x, y=y)
                # Save state for undo
                self._save_to_history()
                return True
                
            return False
            
        except Exception as e:
            print(f"Error moving point: {e}")
            return False
    
    def delete_segment(self, segment_id):
        """
        Delete a segment.
        
        Args:
            segment_id: ID of the segment to delete
            
        Returns:
            success: Boolean indicating if deletion was successful
        """
        try:
            # Find the segment index
            segment_idx = next((i for i, s in enumerate(self.segments) if s.id == segment_id), None)
            if segment_idx is None:
                return False
                
            # Remove the segment
            self.segments.pop(segment_idx)
            
            # Save state for undo
            self._save_to_history()
            return True
            
        except Exception as e:
            print(f"Error deleting segment: {e}")
            return False
    
    def smooth_contour(self, segment_id, contour_idx, smoothness=0.2):
        """
        Apply smoothing to a contour.
        
        Args:
            segment_id: ID of the segment
            contour_idx: Index of the contour within the segment
            smoothness: Smoothing factor (0.0-1.0)
            
        Returns:
            success: Boolean indicating if smoothing was successful
        """
        try:
            # Find the segment
            segment = next((s for s in self.segments if s.id == segment_id), None)
            if not segment or contour_idx >= len(segment.contours):
                return False
                
            # Get the contour
            contour = segment.contours[contour_idx]
            
            # Ensure there are enough points
            if len(contour.points) < 3:
                return False
                
            # Convert points to numpy array
            points = np.array([(p.x, p.y) for p in contour.points])
            
            # Apply Chaikin's corner cutting algorithm
            smoothed_points = self._chaikin_smooth(points, smoothness)
            
            # Update the contour with smoothed points
            contour.points = [Point(x=x, y=y) for x, y in smoothed_points]
            
            # Save state for undo
            self._save_to_history()
            return True
            
        except Exception as e:
            print(f"Error smoothing contour: {e}")
            return False
    
    def simplify_contour(self, segment_id, contour_idx, tolerance=1.0):
        """
        Simplify a contour using Douglas-Peucker algorithm.
        
        Args:
            segment_id: ID of the segment
            contour_idx: Index of the contour within the segment
            tolerance: Simplification tolerance
            
        Returns:
            success: Boolean indicating if simplification was successful
        """
        try:
            # Find the segment
            segment = next((s for s in self.segments if s.id == segment_id), None)
            if not segment or contour_idx >= len(segment.contours):
                return False
                
            # Get the contour
            contour = segment.contours[contour_idx]
            
            # Ensure there are enough points
            if len(contour.points) < 3:
                return False
                
            # Convert points to numpy array for OpenCV
            points = np.array([(p.x, p.y) for p in contour.points]).reshape(-1, 1, 2).astype(np.float32)
            
            # Apply Douglas-Peucker algorithm
            epsilon = tolerance * cv2.arcLength(points, True)
            simplified = cv2.approxPolyDP(points, epsilon, True)
            
            # Update the contour with simplified points
            simplified_points = simplified.reshape(-1, 2)
            contour.points = [Point(x=x, y=y) for x, y in simplified_points]
            
            # Save state for undo
            self._save_to_history()
            return True
            
        except Exception as e:
            print(f"Error simplifying contour: {e}")
            return False
    
    def merge_segments(self, segment_ids):
        """
        Merge multiple segments into one.
        
        Args:
            segment_ids: List of segment IDs to merge
            
        Returns:
            success: Boolean indicating if merging was successful
        """
        try:
            if len(segment_ids) < 2:
                return False
                
            # Find the segments
            segments_to_merge = [s for s in self.segments if s.id in segment_ids]
            if len(segments_to_merge) < 2:
                return False
                
            # Create a new segment with contours from all merged segments
            new_segment = Segment(id=max(s.id for s in self.segments) + 1)
            # Use color from first segment
            new_segment.color = segments_to_merge[0].color
            
            # Add all contours
            for segment in segments_to_merge:
                new_segment.contours.extend(segment.contours)
                
            # Remove old segments
            self.segments = [s for s in self.segments if s.id not in segment_ids]
            
            # Add new merged segment
            self.segments.append(new_segment)
            
            # Save state for undo
            self._save_to_history()
            return True
            
        except Exception as e:
            print(f"Error merging segments: {e}")
            return False
    
    def undo(self):
        """
        Undo the last operation.
        
        Returns:
            success: Boolean indicating if undo was successful
        """
        if self.history_position > 0:
            self.history_position -= 1
            self._restore_from_history()
            return True
        return False
    
    def redo(self):
        """
        Redo the last undone operation.
        
        Returns:
            success: Boolean indicating if redo was successful
        """
        if self.history_position < len(self.history) - 1:
            self.history_position += 1
            self._restore_from_history()
            return True
        return False
    
    def _chaikin_smooth(self, points, ratio=0.25):
        """
        Apply Chaikin's smoothing algorithm to points.
        
        Args:
            points: NumPy array of points
            ratio: Smoothing ratio
            
        Returns:
            smoothed_points: NumPy array of smoothed points
        """
        # Ensure we're working with a closed contour by adding first point as last
        if np.any(points[0] != points[-1]):
            points = np.vstack([points, points[0:1]])
            
        n_points = len(points)
        smoothed = []
        
        for i in range(n_points - 1):
            p0 = points[i]
            p1 = points[(i + 1) % n_points]
            
            # Calculate points at ratio and 1-ratio along the line
            q0 = p0 * (1 - ratio) + p1 * ratio
            q1 = p0 * ratio + p1 * (1 - ratio)
            
            smoothed.extend([q0, q1])
            
        # Return all but last point (which should be first point again)
        return np.array(smoothed[:-1])
    
    def _save_to_history(self):
        """Save current state to history for undo/redo"""
        # Convert segments to serializable form
        state = [asdict(s) for s in self.segments]
        
        # If we made changes after undoing, trim history
        if self.history_position < len(self.history) - 1:
            self.history = self.history[:self.history_position + 1]
            
        # Add state to history
        self.history.append(state)
        self.history_position = len(self.history) - 1
        
        # Limit history size
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
            self.history_position = len(self.history) - 1
    
    def _restore_from_history(self):
        """Restore state from history"""
        if 0 <= self.history_position < len(self.history):
            state = self.history[self.history_position]
            
            # Convert back to objects
            self.segments = []
            for segment_dict in state:
                segment = Segment(id=segment_dict['id'])
                segment.color = tuple(segment_dict['color'])
                segment.is_visible = segment_dict['is_visible']
                segment.metadata = segment_dict['metadata']
                
                for contour_dict in segment_dict['contours']:
                    contour = Contour()
                    contour.is_selected = contour_dict['is_selected']
                    contour.points = [
                        Point(x=p['x'], y=p['y']) for p in contour_dict['points']
                    ]
                    segment.contours.append(contour)
                    
                self.segments.append(segment)