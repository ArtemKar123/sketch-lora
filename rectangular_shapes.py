from base_shapes import BaseShape
from typing import List, Tuple, Dict, Any
import random
import math
import sys
import os

class RectangularShape(BaseShape):
    """Generator for rectangular shapes - squares, rectangles, and rhombus (8-point shapes with corner duplication)."""
    
    # Rectangular shape constants
    MIN_SIZE = 8  # Minimum side length
    MAX_SIZE_RATIO = 0.6  # Maximum size as ratio of grid
    MIN_SEPARATION = 3  # Minimum separation from edges
    
    # T-value patterns for rectangular shapes (corners need duplication)
    T_VALUE_PATTERNS = {
        'even_corners': lambda: [0.00, 0.25, 0.25, 0.50, 0.50, 0.75, 0.75, 1.00],  # more rounded shapes
        'varied_corners': lambda: [0.00, 0.23, 0.27, 0.5, 0.5, 0.73, 0.77, 1.00],   # bent inward but more sharp
    }
    
    def __init__(self, grid_size: int = BaseShape.DEFAULT_GRID_SIZE):
        super().__init__(grid_size)
        self.shape_id = "rectangular"
        
        # Calculate rectangular shape parameters
        self.min_size = max(self.MIN_SIZE, int(self.grid_size * 0.08))
        self.max_size = int(self.grid_size * self.MAX_SIZE_RATIO)
        self.margin = max(self.edge_margin, self.MIN_SEPARATION)
    
    def generate_sample(self, apply_random_transforms: bool = True, start_point: Tuple[int, int] = None, x_range: Tuple[int, int] = None, y_range: Tuple[int, int] = None) -> Dict[str, Any]:
        """Generate a random rectangular shape sample.
        
        Args:
            apply_random_transforms: Whether to apply random transformations
            start_point: Optional starting point (bottom-left corner) for the shape
            x_range: Optional (min_x, max_x) constraint for all points
            y_range: Optional (min_y, max_y) constraint for all points
        """
        
        # Choose shape type
        shape_type = random.choice(['square', 'rectangle', 'rhombus'])
        
        # Generate the shape
        points, t_values, shape_info = self._generate_rectangular_shape(shape_type, start_point, x_range, y_range)
        
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
        """Generate a specific rectangular shape type."""
        points, t_values, _ = self._generate_rectangular_shape(shape_type)
        return points, t_values
    
    def _generate_rectangular_shape(self, shape_type: str, start_point: Tuple[int, int] = None, x_range: Tuple[int, int] = None, y_range: Tuple[int, int] = None) -> Tuple[List[Tuple[int, int]], List[float], Dict[str, Any]]:
        """Generate a rectangular shape with proper corner duplication.
        
        Args:
            shape_type: Type of rectangular shape to generate
            start_point: Optional starting point (bottom-left corner)
            x_range: Optional (min_x, max_x) constraint for all points
            y_range: Optional (min_y, max_y) constraint for all points
        """
        
        if shape_type == 'square':
            return self._generate_square(start_point, x_range, y_range)
        elif shape_type == 'rectangle':
            return self._generate_rectangle(start_point, x_range, y_range)
        elif shape_type == 'rhombus':
            return self._generate_rhombus(start_point, x_range, y_range)
        else:
            # Default to square
            return self._generate_square(start_point, x_range, y_range)
    
    def _generate_square(self, start_point: Tuple[int, int] = None, x_range: Tuple[int, int] = None, y_range: Tuple[int, int] = None) -> Tuple[List[Tuple[int, int]], List[float], Dict[str, Any]]:
        """Generate a square with duplicated corners for shape variation.
        
        Args:
            start_point: Optional starting point (bottom-left corner)
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
        
        # Calculate maximum possible square size within constraints
        max_width = max_x - min_x
        max_height = max_y - min_y
        max_possible_size = min(max_width, max_height, self.max_size)
        
        # Choose square size
        side_length = random.randint(self.min_size, max(self.min_size, max_possible_size))
        
        if start_point is not None:
            # Use provided start point, constrained within bounds
            start_x, start_y = start_point
            start_x = max(min_x, min(max_x - side_length, start_x))
            start_y = max(min_y, min(max_y - side_length, start_y))
        else:
            # Choose position randomly within constraints
            max_start_x = max_x - side_length
            max_start_y = max_y - side_length
        
            start_x = random.randint(min_x, max(min_x, max_start_x))
            start_y = random.randint(min_y, max(min_y, max_start_y))
        
        # Calculate corner coordinates
        x1, y1 = start_x, start_y                    # Bottom-left
        x2, y2 = start_x + side_length, start_y     # Bottom-right  
        x3, y3 = start_x + side_length, start_y + side_length  # Top-right
        x4, y4 = start_x, start_y + side_length     # Top-left
        
        # Create points with corner duplication (following prompts.py format)
        points = [
            (x1, y1),  # Start at bottom-left
            (x2, y2), (x2, y2),  # Bottom-right corner (duplicated)
            (x3, y3), (x3, y3),  # Top-right corner (duplicated)
            (x4, y4), (x4, y4),  # Top-left corner (duplicated)
            (x1, y1)   # Back to start
        ]
        
        # Choose t-values pattern
        pattern_name = random.choice(list(self.T_VALUE_PATTERNS.keys()))
        t_values = self.T_VALUE_PATTERNS[pattern_name]()
        
        shape_info = {
            'side_length': side_length,
            'center': (start_x + side_length//2, start_y + side_length//2),
            'pattern': pattern_name,
            'corner_style': 'rounded' if pattern_name == 'even_corners' else 'sharp_inward'
        }
        
        return points, t_values, shape_info
    
    def _generate_rectangle(self, start_point: Tuple[int, int] = None, x_range: Tuple[int, int] = None, y_range: Tuple[int, int] = None) -> Tuple[List[Tuple[int, int]], List[float], Dict[str, Any]]:
        """Generate a rectangle with duplicated corners for shape variation.
        
        Args:
            start_point: Optional starting point (bottom-left corner)
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
        
        # Calculate maximum possible rectangle dimensions within constraints
        max_width = max_x - min_x
        max_height = max_y - min_y
        max_possible_width = min(max_width, self.max_size)
        max_possible_height = min(max_height, self.max_size)
        
        # Choose rectangle dimensions (width != height)
        width = random.randint(self.min_size, max(self.min_size, max_possible_width))
        height = random.randint(self.min_size, max(self.min_size, max_possible_height))
        
        # Ensure it's actually a rectangle (not a square)
        attempts = 0
        while abs(width - height) < 3 and attempts < 10:  # Minimum difference to distinguish from square
            if random.choice([True, False]) and max_possible_width > self.min_size:
                width = random.randint(self.min_size, max_possible_width)
            elif max_possible_height > self.min_size:
                height = random.randint(self.min_size, max_possible_height)
            attempts += 1
        
        if start_point is not None:
            # Use provided start point, constrained within bounds
            start_x, start_y = start_point
            start_x = max(min_x, min(max_x - width, start_x))
            start_y = max(min_y, min(max_y - height, start_y))
        else:
            # Choose position randomly within constraints
            max_start_x = max_x - width
            max_start_y = max_y - height
        
            start_x = random.randint(min_x, max(min_x, max_start_x))
            start_y = random.randint(min_y, max(min_y, max_start_y))
        
        # Calculate corner coordinates
        x1, y1 = start_x, start_y              # Bottom-left
        x2, y2 = start_x + width, start_y     # Bottom-right
        x3, y3 = start_x + width, start_y + height   # Top-right
        x4, y4 = start_x, start_y + height    # Top-left
        
        # Create points with corner duplication (following prompts.py format)
        points = [
            (x1, y1),  # Start at bottom-left
            (x2, y2), (x2, y2),  # Bottom-right corner (duplicated)
            (x3, y3), (x3, y3),  # Top-right corner (duplicated)
            (x4, y4), (x4, y4),  # Top-left corner (duplicated)
            (x1, y1)   # Back to start
        ]
        
        # Choose t-values pattern
        pattern_name = random.choice(list(self.T_VALUE_PATTERNS.keys()))
        t_values = self.T_VALUE_PATTERNS[pattern_name]()
        
        shape_info = {
            'width': width,
            'height': height,
            'center': (start_x + width//2, start_y + height//2),
            'pattern': pattern_name,
            'corner_style': 'rounded' if pattern_name == 'even_corners' else 'sharp_inward'
        }
        
        return points, t_values, shape_info
    
    def _generate_rhombus(self, start_point: Tuple[int, int] = None, x_range: Tuple[int, int] = None, y_range: Tuple[int, int] = None) -> Tuple[List[Tuple[int, int]], List[float], Dict[str, Any]]:
        """Generate a rhombus with duplicated corners for shape variation.
        
        Args:
            start_point: Optional starting point (center of rhombus)
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
        
        # Choose rhombus parameters
        # For a rhombus, we need equal side lengths but non-90 degree angles
        available_width = max_x - min_x
        available_height = max_y - min_y
        max_size_constrained = min(available_width//2, available_height//2, self.max_size, self.grid_size // 3)
        
        side_length = random.randint(self.min_size, max(self.min_size, max_size_constrained))
        
        # Choose center position within constraints
        center_margin = side_length
        center_min_x = max(min_x + center_margin, margin + center_margin)
        center_max_x = min(max_x - center_margin, self.grid_size - margin - center_margin)
        center_min_y = max(min_y + center_margin, margin + center_margin)
        center_max_y = min(max_y - center_margin, self.grid_size - margin - center_margin)
        
        if start_point is not None:
            # Use provided start point as center, constrained within bounds
            center_x, center_y = start_point
            center_x = max(center_min_x, min(center_max_x, center_x))
            center_y = max(center_min_y, min(center_max_y, center_y))
        else:
            # Ensure we have valid center ranges
            if center_min_x >= center_max_x:
                center_x = (min_x + max_x) // 2
            else:
                center_x = random.randint(center_min_x, center_max_x)
                
            if center_min_y >= center_max_y:
                center_y = (min_y + max_y) // 2
            else:
                center_y = random.randint(center_min_y, center_max_y)
        
        # Choose rhombus angle (not 90 degrees to distinguish from square)
        # Angle between 30-60 degrees or 120-150 degrees for clear rhombus shape
        angle_deg = random.choice([
            random.randint(30, 60),
            random.randint(120, 150)
        ])
        angle_rad = math.radians(angle_deg)
        
        # Calculate rhombus vertices using angle and side length
        # We'll create a rhombus by placing vertices at calculated offsets from center
        half_diagonal1 = side_length * math.cos(angle_rad / 2)
        half_diagonal2 = side_length * math.sin(angle_rad / 2)
        
        # Calculate corner coordinates (diamond orientation)
        x1 = int(center_x - half_diagonal1)  # Left vertex
        y1 = int(center_y)
        
        x2 = int(center_x)                   # Bottom vertex
        y2 = int(center_y - half_diagonal2)
        
        x3 = int(center_x + half_diagonal1)  # Right vertex
        y3 = int(center_y)
        
        x4 = int(center_x)                   # Top vertex
        y4 = int(center_y + half_diagonal2)
        
        # Ensure all points are within the specified ranges
        x1 = max(min_x, min(max_x, x1))
        x2 = max(min_x, min(max_x, x2))
        x3 = max(min_x, min(max_x, x3))
        x4 = max(min_x, min(max_x, x4))
        
        y1 = max(min_y, min(max_y, y1))
        y2 = max(min_y, min(max_y, y2))
        y3 = max(min_y, min(max_y, y3))
        y4 = max(min_y, min(max_y, y4))
        
        # Create points with corner duplication (following prompts.py format)
        points = [
            (x1, y1),  # Start at left vertex
            (x2, y2), (x2, y2),  # Bottom vertex (duplicated)
            (x3, y3), (x3, y3),  # Right vertex (duplicated)
            (x4, y4), (x4, y4),  # Top vertex (duplicated)
            (x1, y1)   # Back to start
        ]
        
        # Choose t-values pattern
        pattern_name = random.choice(list(self.T_VALUE_PATTERNS.keys()))
        t_values = self.T_VALUE_PATTERNS[pattern_name]()
        
        shape_info = {
            'side_length': side_length,
            'angle_degrees': angle_deg,
            'center': (center_x, center_y),
            'pattern': pattern_name,
            'corner_style': 'rounded' if pattern_name == 'even_corners' else 'sharp_inward'
        }
        
        return points, t_values, shape_info 