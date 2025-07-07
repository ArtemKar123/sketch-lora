from base_shapes import BaseShape
from typing import List, Tuple, Dict, Any
import random
import math
import sys
import os

class CircularShape(BaseShape):
    """Generator for circular shapes - circles and ellipses as single-stroke closed curves.
    
    Following the SketchAgent pattern where circles start and end at the same point
    with evenly distributed intermediate points around the circumference.
    """
    
    # Circular shape constants
    MIN_RADIUS = 8  # Minimum radius
    MAX_RADIUS_RATIO = 0.4  # Maximum radius as ratio of grid
    MIN_SEPARATION = 5  # Minimum separation from edges
    DEFAULT_POINT_COUNT = 9  # Including start/end point (8 intermediate + 1 duplicate)
    
    def __init__(self, grid_size: int = BaseShape.DEFAULT_GRID_SIZE):
        super().__init__(grid_size)
        self.shape_id = "circular"
        
        # Calculate circular shape parameters
        self.min_radius = max(self.MIN_RADIUS, int(self.grid_size * 0.08))
        self.max_radius = int(self.grid_size * self.MAX_RADIUS_RATIO)
        self.margin = max(self.edge_margin, self.MIN_SEPARATION)
    
    def generate_sample(self, apply_random_transforms: bool = True, start_point: Tuple[int, int] = None, x_range: Tuple[int, int] = None, y_range: Tuple[int, int] = None) -> Dict[str, Any]:
        """Generate a complete circular shape sample.
        
        Args:
            apply_random_transforms: Whether to apply random transformations
            start_point: Optional center point for the shape
            x_range: Optional (min_x, max_x) constraint for all points
            y_range: Optional (min_y, max_y) constraint for all points
        """
        
        # Randomly choose circle type
        shape_type = random.choice(['circle', 'ellipse'])
        
        # Generate the shape
        points, t_values, shape_info = self._generate_circular_shape(shape_type, start_point, x_range, y_range)
        
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
    
    def generate_shape_type(self, shape_type: str, start_point: Tuple[int, int] = None) -> Tuple[List[Tuple[int, int]], List[float]]:
        """Generate a specific circular shape type."""
        points, t_values, shape_info = self._generate_circular_shape(shape_type, start_point)
        
        # Store the shape_info for later use
        self._last_shape_info = shape_info
        
        return points, t_values
    
    def get_last_shape_info(self) -> Dict[str, Any]:
        """Get the shape_info from the last generated shape."""
        return getattr(self, '_last_shape_info', {})
    
    def _generate_circular_shape(self, shape_type: str, start_point: Tuple[int, int] = None, x_range: Tuple[int, int] = None, y_range: Tuple[int, int] = None) -> Tuple[List[Tuple[int, int]], List[float], Dict[str, Any]]:
        """Generate a circular shape with evenly distributed points.
        
        Args:
            shape_type: Type of circular shape ('circle' or 'ellipse')
            start_point: Optional center point for the shape
            x_range: Optional (min_x, max_x) constraint for all points
            y_range: Optional (min_y, max_y) constraint for all points
        """
        
        if shape_type == 'circle':
            return self._generate_circle(start_point, x_range, y_range)
        elif shape_type == 'ellipse':
            return self._generate_ellipse(start_point, x_range, y_range)
        else:
            # Default to circle
            return self._generate_circle(start_point, x_range, y_range)
    
    def _generate_circle(self, start_point: Tuple[int, int] = None, x_range: Tuple[int, int] = None, y_range: Tuple[int, int] = None) -> Tuple[List[Tuple[int, int]], List[float], Dict[str, Any]]:
        """Generate a perfect circle with evenly distributed points.
        
        Args:
            start_point: Optional center point for the circle
            x_range: Optional (min_x, max_x) constraint for all points
            y_range: Optional (min_y, max_y) constraint for all points
        """
        
        # Calculate effective bounds considering margins and optional ranges
        margin = self.margin
        
        # Default bounds
        min_x, max_x = margin, self.grid_size - margin
        min_y, max_y = margin, self.grid_size - margin
        
        # Apply x_range constraint if provided
        if x_range is not None:
            min_x = max(min_x, x_range[0])
            max_x = min(max_x, x_range[1])
        
        # Apply y_range constraint if provided  
        if y_range is not None:
            min_y = max(min_y, y_range[0])
            max_y = min(max_y, y_range[1])
        
        # Calculate maximum possible radius within constraints
        available_width = max_x - min_x
        available_height = max_y - min_y
        max_possible_radius = min(available_width//2, available_height//2, self.max_radius)
        
        # Choose circle radius
        radius = random.randint(self.min_radius, max(self.min_radius, max_possible_radius))
        
        # Choose center position to ensure circle fits within constraints
        if start_point:
            center_x, center_y = start_point
            # Constrain center to ensure circle fits within bounds
            center_x = max(min_x + radius, min(max_x - radius, center_x))
            center_y = max(min_y + radius, min(max_y - radius, center_y))
        else:
            center_min_x = min_x + radius
            center_max_x = max_x - radius
            center_min_y = min_y + radius
            center_max_y = max_y - radius
            
            # Ensure we have valid center ranges
            if center_min_x >= center_max_x:
                center_x = (min_x + max_x) // 2
            else:
                center_x = random.randint(center_min_x, center_max_x)
                
            if center_min_y >= center_max_y:
                center_y = (min_y + max_y) // 2
            else:
                center_y = random.randint(center_min_y, center_max_y)
        
        # Generate points around the circle
        points = []
        num_segments = self.DEFAULT_POINT_COUNT - 1  # Exclude duplicate start/end point
        
        for i in range(self.DEFAULT_POINT_COUNT):
            # Calculate angle for this point (full circle = 2π)
            if i == self.DEFAULT_POINT_COUNT - 1:
                # Last point is same as first (closed loop)
                angle = 0
            else:
                angle = (2 * math.pi * i) / num_segments
            
            # Calculate point coordinates
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            
            # Ensure points are within specified bounds
            x = max(min_x, min(max_x, int(round(x))))
            y = max(min_y, min(max_y, int(round(y))))
            
            points.append((x, y))
        
        # Generate evenly distributed t_values
        t_values = [i / (self.DEFAULT_POINT_COUNT - 1) for i in range(self.DEFAULT_POINT_COUNT)]
        
        shape_info = {
            'radius': radius,
            'center': (center_x, center_y),
            'point_count': self.DEFAULT_POINT_COUNT,
            'type': 'perfect_circle'
        }
        
        return points, t_values, shape_info
    
    def _generate_ellipse(self, start_point: Tuple[int, int] = None, x_range: Tuple[int, int] = None, y_range: Tuple[int, int] = None) -> Tuple[List[Tuple[int, int]], List[float], Dict[str, Any]]:
        """Generate an ellipse with evenly distributed points.
        
        Args:
            start_point: Optional center point for the ellipse
            x_range: Optional (min_x, max_x) constraint for all points
            y_range: Optional (min_y, max_y) constraint for all points
        """
        
        # Calculate effective bounds considering margins and optional ranges
        margin = self.margin
        
        # Default bounds
        min_x, max_x = margin, self.grid_size - margin
        min_y, max_y = margin, self.grid_size - margin
        
        # Apply x_range constraint if provided
        if x_range is not None:
            min_x = max(min_x, x_range[0])
            max_x = min(max_x, x_range[1])
        
        # Apply y_range constraint if provided  
        if y_range is not None:
            min_y = max(min_y, y_range[0])
            max_y = min(max_y, y_range[1])
        
        # Calculate maximum possible radii within constraints
        available_width = max_x - min_x
        available_height = max_y - min_y
        max_possible_radius_x = min(available_width//2, self.max_radius)
        max_possible_radius_y = min(available_height//2, self.max_radius)
        
        # Choose ellipse radii (different for width and height)
        radius_x = random.randint(self.min_radius, max(self.min_radius, max_possible_radius_x))
        radius_y = random.randint(self.min_radius, max(self.min_radius, max_possible_radius_y))
        
        # Ensure it's actually an ellipse (not a circle)
        attempts = 0
        while abs(radius_x - radius_y) < 3 and attempts < 10:  # Minimum difference for ellipse
            if random.choice([True, False]) and max_possible_radius_x > self.min_radius:
                radius_x = random.randint(self.min_radius, max_possible_radius_x)
            elif max_possible_radius_y > self.min_radius:
                radius_y = random.randint(self.min_radius, max_possible_radius_y)
            attempts += 1
        
        # Choose center position to ensure ellipse fits within constraints
        if start_point:
            center_x, center_y = start_point
            # Constrain center to ensure ellipse fits within bounds
            center_x = max(min_x + radius_x, min(max_x - radius_x, center_x))
            center_y = max(min_y + radius_y, min(max_y - radius_y, center_y))
        else:
            center_min_x = min_x + radius_x
            center_max_x = max_x - radius_x
            center_min_y = min_y + radius_y
            center_max_y = max_y - radius_y
            
            # Ensure we have valid center ranges
            if center_min_x >= center_max_x:
                center_x = (min_x + max_x) // 2
            else:
                center_x = random.randint(center_min_x, center_max_x)
                
            if center_min_y >= center_max_y:
                center_y = (min_y + max_y) // 2
            else:
                center_y = random.randint(center_min_y, center_max_y)
        
        # Generate points around the ellipse
        points = []
        num_segments = self.DEFAULT_POINT_COUNT - 1  # Exclude duplicate start/end point
        
        for i in range(self.DEFAULT_POINT_COUNT):
            # Calculate angle for this point
            if i == self.DEFAULT_POINT_COUNT - 1:
                # Last point is same as first (closed loop)
                angle = 0
            else:
                angle = (2 * math.pi * i) / num_segments
            
            # Calculate point coordinates for ellipse
            x = center_x + radius_x * math.cos(angle)
            y = center_y + radius_y * math.sin(angle)
            
            # Ensure points are within specified bounds
            x = max(min_x, min(max_x, int(round(x))))
            y = max(min_y, min(max_y, int(round(y))))
            
            points.append((x, y))
        
        # Generate evenly distributed t_values
        t_values = [i / (self.DEFAULT_POINT_COUNT - 1) for i in range(self.DEFAULT_POINT_COUNT)]
        
        shape_info = {
            'radius_x': radius_x,
            'radius_y': radius_y,
            'center': (center_x, center_y),
            'point_count': self.DEFAULT_POINT_COUNT,
            'type': 'ellipse'
        }
        
        return points, t_values, shape_info 