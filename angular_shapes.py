from base_shapes import BaseShape
from typing import List, Tuple, Dict, Any
import random
import math

class AngularShape(BaseShape):
    """Generator for non-smooth curves with sharp corners (V-shapes, L-shapes, zigzags, etc).
    
    Following the SketchAgent pattern where corner points are duplicated with adjacent t-values
    to create sharp corners instead of smooth curves.
    """
    
    # Angular shape constants
    MIN_SEGMENT_LENGTH = 8  # Minimum length for each segment
    MAX_SEGMENTS = 5       # Maximum number of segments
    MIN_ANGLE_CHANGE = 30  # Minimum angle change at corners (degrees)
    MAX_ANGLE_CHANGE = 150 # Maximum angle change at corners (degrees)
    
    def __init__(self, grid_size: int = BaseShape.DEFAULT_GRID_SIZE):
        super().__init__(grid_size)
        self.shape_id = "angular"
        
        # Calculate angular shape parameters
        self.min_segment_length = max(self.MIN_SEGMENT_LENGTH, int(self.grid_size * 0.08))
        self.max_segment_length = int(self.grid_size * 0.3)
        self.margin = max(self.edge_margin, 10)
    
    def generate_sample(self, apply_random_transforms: bool = True) -> Dict[str, Any]:
        """Generate a complete angular shape sample."""
        
        # Randomly choose angular shape type
        shape_types = ['v_shape', 'l_shape', 'zigzag', 'step']
        shape_type = random.choice(shape_types)
        
        # Generate the shape
        points, t_values, shape_info = self._generate_angular_shape(shape_type)
        
        # Apply transformations if requested
        if apply_random_transforms:
            points, t_values_transformed, transformations = self.apply_random_transformations(points, t_values)
        else:
            transformations = {'scale': 1.0, 'shift_x': 0, 'shift_y': 0, 'rotation_deg': 0.0}
            t_values_transformed = t_values
        
        # Format points for output
        formatted_points = self.format_points_for_prompt(points)
        
        return {
            'shape_id': self.shape_id,
            'shape_type': shape_type,
            'raw_points': points,
            'points': formatted_points,
            't_values': t_values_transformed,
            'transformations': transformations,
            'shape_info': shape_info
        }
    
    def generate_shape_type(self, shape_type: str) -> Tuple[List[Tuple[int, int]], List[float]]:
        """Generate a specific angular shape type."""
        points, t_values, _ = self._generate_angular_shape(shape_type)
        return points, t_values
    
    def _generate_angular_shape(self, shape_type: str) -> Tuple[List[Tuple[int, int]], List[float], Dict[str, Any]]:
        """Generate an angular shape with sharp corners."""
        
        if shape_type == 'v_shape':
            return self._generate_v_shape()
        elif shape_type == 'l_shape':
            return self._generate_l_shape()
        elif shape_type == 'zigzag':
            return self._generate_zigzag()
        elif shape_type == 'step':
            return self._generate_step_pattern()
        else:
            return self._generate_v_shape()
    
    def _generate_v_shape(self) -> Tuple[List[Tuple[int, int]], List[float], Dict[str, Any]]:
        """Generate a V-shape (upward or downward)."""
        
        # Choose orientation
        upward = random.choice([True, False])
        
        # Ensure we have enough space for minimum segment lengths
        required_space = self.min_segment_length * 2
        if self.grid_size - 2 * self.margin < required_space:
            # Fallback to smaller segment length if grid is too small
            min_seg = max(5, (self.grid_size - 2 * self.margin) // 3)
        else:
            min_seg = self.min_segment_length
        
        # Generate three points: start, corner (duplicated), end
        if upward:
            # Upward V: start low, peak high, end low
            max_start_y = self.grid_size // 2 - min_seg
            start_x = random.randint(self.margin, self.grid_size // 2)
            start_y = random.randint(self.margin, max(self.margin, max_start_y))
            
            # Ensure corner_x has valid range
            min_corner_x = start_x + min_seg
            max_corner_x = self.grid_size - self.margin
            if min_corner_x >= max_corner_x:
                corner_x = min_corner_x
            else:
                corner_x = random.randint(min_corner_x, max_corner_x)
            corner_y = random.randint(start_y + min_seg, self.grid_size - self.margin)
            
            # Ensure end_x has valid range
            min_end_x = corner_x + min_seg // 2
            max_end_x = self.grid_size - self.margin
            if min_end_x >= max_end_x:
                end_x = min_end_x
            else:
                end_x = random.randint(min_end_x, max_end_x)
            
            # Ensure valid range for end_y
            max_end_y = corner_y - min_seg
            end_y = random.randint(self.margin, max(self.margin, max_end_y))
        else:
            # Downward V: start high, valley low, end high
            min_start_y = self.grid_size // 2 + min_seg
            start_x = random.randint(self.margin, self.grid_size // 2)
            start_y = random.randint(min(min_start_y, self.grid_size - self.margin), self.grid_size - self.margin)
            
            # Ensure corner_x has valid range
            min_corner_x = start_x + min_seg
            max_corner_x = self.grid_size - self.margin
            if min_corner_x >= max_corner_x:
                corner_x = min_corner_x
            else:
                corner_x = random.randint(min_corner_x, max_corner_x)
            
            # Ensure valid range for corner_y
            max_corner_y = start_y - min_seg
            corner_y = random.randint(self.margin, max(self.margin, max_corner_y))
            
            # Ensure end_x has valid range
            min_end_x = corner_x + min_seg // 2
            max_end_x = self.grid_size - self.margin
            if min_end_x >= max_end_x:
                end_x = min_end_x
            else:
                end_x = random.randint(min_end_x, max_end_x)
            end_y = random.randint(corner_y + min_seg, self.grid_size - self.margin)
        
        # Create points with duplicated corner (following SketchAgent pattern)
        points = [
            (start_x, start_y),
            (corner_x, corner_y),  # First corner
            (corner_x, corner_y),  # Duplicated corner for sharp angle
            (end_x, end_y)
        ]
        
        # Create t-values with adjacent values for corner (following SketchAgent pattern)
        t1 = random.uniform(0.4, 0.6)
        t2 = t1 - random.uniform(0.02, 0.1)  # Adjacent value slightly less
        t_values = [0.0, t1, t2, 1.0]
        
        shape_info = {
            'orientation': 'upward' if upward else 'downward',
            'corner_count': 1,
            'segments': 2,
            'type': 'v_shape'
        }
        
        return points, t_values, shape_info
    
    def _generate_l_shape(self) -> Tuple[List[Tuple[int, int]], List[float], Dict[str, Any]]:
        """Generate an L-shape in various orientations."""
        
        # Choose orientation (4 possible L orientations)
        orientation = random.choice(['bottom_left', 'bottom_right', 'top_left', 'top_right'])
        
        if orientation == 'bottom_left':
            # L starting from bottom, going up, then right
            start_x = random.randint(self.margin, self.grid_size // 2)
            start_y = random.randint(self.margin, self.grid_size // 2)
            
            corner_x = start_x
            min_corner_y = start_y + self.min_segment_length
            corner_y = random.randint(min_corner_y, self.grid_size - self.margin)
            
            end_x = random.randint(corner_x + self.min_segment_length, self.grid_size - self.margin)
            end_y = corner_y
            
        elif orientation == 'bottom_right':
            # L starting from bottom, going up, then left
            start_x = random.randint(self.grid_size // 2, self.grid_size - self.margin)
            start_y = random.randint(self.margin, self.grid_size // 2)
            
            corner_x = start_x
            min_corner_y = start_y + self.min_segment_length
            corner_y = random.randint(min_corner_y, self.grid_size - self.margin)
            
            max_end_x = corner_x - self.min_segment_length
            end_x = random.randint(self.margin, max(self.margin, max_end_x))
            end_y = corner_y
            
        elif orientation == 'top_left':
            # L starting from top, going down, then right
            start_x = random.randint(self.margin, self.grid_size // 2)
            start_y = random.randint(self.grid_size // 2, self.grid_size - self.margin)
            
            corner_x = start_x
            max_corner_y = start_y - self.min_segment_length
            corner_y = random.randint(self.margin, max(self.margin, max_corner_y))
            
            end_x = random.randint(corner_x + self.min_segment_length, self.grid_size - self.margin)
            end_y = corner_y
            
        else:  # top_right
            # L starting from top, going down, then left
            start_x = random.randint(self.grid_size // 2, self.grid_size - self.margin)
            start_y = random.randint(self.grid_size // 2, self.grid_size - self.margin)
            
            corner_x = start_x
            max_corner_y = start_y - self.min_segment_length
            corner_y = random.randint(self.margin, max(self.margin, max_corner_y))
            
            max_end_x = corner_x - self.min_segment_length
            end_x = random.randint(self.margin, max(self.margin, max_end_x))
            end_y = corner_y
        
        # Create points with duplicated corner
        points = [
            (start_x, start_y),
            (corner_x, corner_y),  # First corner
            (corner_x, corner_y),  # Duplicated corner for sharp 90-degree angle
            (end_x, end_y)
        ]
        
        # Create t-values with adjacent values for corner
        t1 = random.uniform(0.45, 0.55)
        t2 = t1 - random.uniform(0.02, 0.08)
        t_values = [0.0, t1, t2, 1.0]
        
        shape_info = {
            'orientation': orientation,
            'corner_count': 1,
            'segments': 2,
            'type': 'l_shape'
        }
        
        return points, t_values, shape_info
    
    def _generate_zigzag(self) -> Tuple[List[Tuple[int, int]], List[float], Dict[str, Any]]:
        """Generate a zigzag pattern with multiple sharp corners."""
        
        # Choose number of peaks (2-4)
        num_peaks = random.randint(2, 4)
        
        # Generate zigzag from left to right
        width = self.grid_size - 2 * self.margin
        segment_width = width // (num_peaks + 1)
        
        points = []
        
        # Start point
        start_x = self.margin
        start_y = random.randint(self.margin + 20, self.grid_size - self.margin - 20)
        points.append((start_x, start_y))
        
        current_x = start_x
        up = True  # Start going up
        
        for i in range(num_peaks):
            # Next peak/valley
            current_x += segment_width
            if up:
                # Going up - ensure we have room above start_y
                min_peak_y = start_y + 15
                peak_y = random.randint(min_peak_y, self.grid_size - self.margin)
            else:
                # Going down - ensure we have room below start_y
                max_peak_y = start_y - 15
                peak_y = random.randint(self.margin, max(self.margin, max_peak_y))
            
            # Add corner point twice
            points.append((current_x, peak_y))
            points.append((current_x, peak_y))  # Duplicate for sharp corner
            
            up = not up  # Alternate direction
        
        # End point
        end_x = self.grid_size - self.margin
        end_y = random.randint(self.margin + 20, self.grid_size - self.margin - 20)
        points.append((end_x, end_y))
        
        # Generate t-values with adjacent pairs for corners
        t_values = []
        num_segments = len(points)
        
        for i in range(num_segments):
            if i == 0:
                t_values.append(0.0)
            elif i == num_segments - 1:
                t_values.append(1.0)
            elif i % 2 == 1:  # First of corner pair
                t = i / (num_segments - 1)
                t_values.append(t + random.uniform(0.01, 0.05))
            else:  # Second of corner pair
                t = i / (num_segments - 1)
                t_values.append(t - random.uniform(0.01, 0.05))
        
        shape_info = {
            'peaks': num_peaks,
            'corner_count': num_peaks,
            'segments': num_peaks + 1,
            'type': 'zigzag'
        }
        
        return points, t_values, shape_info
    
    def _generate_step_pattern(self) -> Tuple[List[Tuple[int, int]], List[float], Dict[str, Any]]:
        """Generate a step pattern (stairs-like shape)."""
        
        # Choose number of steps (2-4)
        num_steps = random.randint(2, 4)
        
        # Choose direction (up/down and left/right)
        going_up = random.choice([True, False])
        going_right = random.choice([True, False])
        
        points = []
        
        # Start point
        if going_right:
            start_x = self.margin
            if going_up:
                start_y = self.margin
            else:
                start_y = self.grid_size - self.margin
        else:
            start_x = self.grid_size - self.margin
            if going_up:
                start_y = self.margin
            else:
                start_y = self.grid_size - self.margin
        
        points.append((start_x, start_y))
        
        current_x, current_y = start_x, start_y
        step_size_x = (self.grid_size - 2 * self.margin) // (num_steps * 2)
        step_size_y = (self.grid_size - 2 * self.margin) // (num_steps * 2)
        
        for i in range(num_steps):
            # Horizontal segment
            if going_right:
                current_x += step_size_x
            else:
                current_x -= step_size_x
            
            # Add corner point twice
            points.append((current_x, current_y))
            points.append((current_x, current_y))
            
            # Vertical segment
            if going_up:
                current_y += step_size_y
            else:
                current_y -= step_size_y
            
            # Add corner point twice
            points.append((current_x, current_y))
            points.append((current_x, current_y))
        
        # Generate t-values
        t_values = []
        for i in range(len(points)):
            if i == 0:
                t_values.append(0.0)
            elif i == len(points) - 1:
                t_values.append(1.0)
            else:
                base_t = i / (len(points) - 1)
                if i % 2 == 1:  # First of corner pair
                    t_values.append(base_t + random.uniform(0.01, 0.03))
                else:  # Second of corner pair
                    t_values.append(base_t - random.uniform(0.01, 0.03))
        
        shape_info = {
            'steps': num_steps,
            'direction': ('up' if going_up else 'down') + '_' + ('right' if going_right else 'left'),
            'corner_count': num_steps * 2,
            'segments': num_steps * 2 + 1,
            'type': 'step_pattern'
        }
        
        return points, t_values, shape_info 