from base_shapes import BaseShape
from typing import List, Tuple
import random

class StraightLine(BaseShape):
    """Generator for straight lines with various orientations and lengths."""
    
    # Line-specific constants
    MIN_LINE_LENGTH = 5
    MAX_SLOPE_MAGNITUDE = 2.0  # Avoid very steep slopes
    
    def __init__(self, grid_size: int = BaseShape.DEFAULT_GRID_SIZE):
        super().__init__(grid_size)
        self.shape_id = "straight_line"
        
        # Calculate line-specific parameters based on grid size
        self.min_line_length = max(self.MIN_LINE_LENGTH, int(self.grid_size * 0.05))
        self.max_line_length = int(self.grid_size * 0.8)  # 80% of grid size
        self.diagonal_margin = max(15, int(self.grid_size * 0.15))  # Extra margin for diagonal calculations
    
    def generate_base_shape(self, start_point: Tuple[int, int] = None, x_range: Tuple[int, int] = None, y_range: Tuple[int, int] = None) -> Tuple[List[Tuple[int, int]], List[float]]:
        """Generate a straight line from one point to another.
        
        Args:
            start_point: Optional starting point for the line
            x_range: Optional (min_x, max_x) constraint for all points
            y_range: Optional (min_y, max_y) constraint for all points
        """
        
        # Calculate effective bounds considering margins and optional ranges
        margin = self.edge_margin
        
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
        
        # Ensure we have valid ranges
        min_x = min(min_x, max_x - 1)
        min_y = min(min_y, max_y - 1)
        
        if start_point is not None:
            # Use provided start point, but constrain within bounds
            x1, y1 = start_point
            x1 = max(min_x, min(max_x, x1))
            y1 = max(min_y, min(max_y, y1))
        else:
            # Generate random start point within bounds
            x1 = random.randint(min_x, max_x)
            y1 = random.randint(min_y, max_y)
        
        # Generate random end point within bounds
        x2 = random.randint(min_x, max_x)
        y2 = random.randint(min_y, max_y)
        
        # Ensure the points are different and create a meaningful line
        min_distance = self.min_line_length
        attempts = 0
        max_attempts = 20
        
        while ((x1, y1) == (x2, y2) or ((x2 - x1)**2 + (y2 - y1)**2)**0.5 < min_distance) and attempts < max_attempts:
            x2 = random.randint(min_x, max_x)
            y2 = random.randint(min_y, max_y)
            attempts += 1
        
        # If we couldn't find a good end point within constraints, just move away from start
        if (x1, y1) == (x2, y2):
            # Try to move in available direction within constraints
            if x2 < max_x:
                x2 = min(max_x, x1 + self.min_line_length)
            elif x2 > min_x:
                x2 = max(min_x, x1 - self.min_line_length)
            elif y2 < max_y:
                y2 = min(max_y, y1 + self.min_line_length)
            elif y2 > min_y:
                y2 = max(min_y, y1 - self.min_line_length)
        
        points = [(x1, y1), (x2, y2)]
        t_values = [0.0, 1.0]
        
        return points, t_values
    
    def generate_specific_line(self, 
                             start_point: Tuple[int, int], 
                             end_point: Tuple[int, int]) -> Tuple[List[Tuple[int, int]], List[float]]:
        """Generate a line between two specific points."""
        points = [start_point, end_point]
        t_values = [0.0, 1.0]
        return points, t_values
    
    def generate_horizontal_line(self, length: int = None) -> Tuple[List[Tuple[int, int]], List[float]]:
        """Generate a horizontal line of specified length (or random if None)."""
        if length is None:
            length = random.randint(self.min_line_length, self.max_line_length)
        
        # Ensure line fits within grid
        length = min(length, self.grid_size - 2 * self.edge_margin)
        
        # Random starting y position
        y = random.randint(self.edge_margin, self.grid_size - self.edge_margin)
        
        # Calculate x positions ensuring line fits
        max_start_x = self.grid_size - self.edge_margin - length
        x1 = random.randint(self.edge_margin, max(self.edge_margin, max_start_x))
        x2 = x1 + length
        
        points = [(x1, y), (x2, y)]
        t_values = [0.0, 1.0]
        return points, t_values
    
    def generate_vertical_line(self, length: int = None) -> Tuple[List[Tuple[int, int]], List[float]]:
        """Generate a vertical line of specified length (or random if None)."""
        if length is None:
            length = random.randint(self.min_line_length, self.max_line_length)
        
        # Ensure line fits within grid
        length = min(length, self.grid_size - 2 * self.edge_margin)
        
        # Random starting x position
        x = random.randint(self.edge_margin, self.grid_size - self.edge_margin)
        
        # Calculate y positions ensuring line fits
        max_start_y = self.grid_size - self.edge_margin - length
        y1 = random.randint(self.edge_margin, max(self.edge_margin, max_start_y))
        y2 = y1 + length
        
        points = [(x, y1), (x, y2)]
        t_values = [0.0, 1.0]
        return points, t_values
    
    def generate_diagonal_line(self, length: int = None, slope: float = None) -> Tuple[List[Tuple[int, int]], List[float]]:
        """Generate a diagonal line with specified length and slope."""
        if length is None:
            length = random.randint(self.min_line_length, self.max_line_length)
        
        if slope is None:
            # Random slope avoiding very steep slopes
            slope = random.uniform(-self.MAX_SLOPE_MAGNITUDE, self.MAX_SLOPE_MAGNITUDE)
        
        # Starting point with extra margin for diagonal calculations
        margin = self.diagonal_margin
        max_start_coord = self.grid_size - margin
        
        x1 = random.randint(margin, max_start_coord)
        y1 = random.randint(margin, max_start_coord)
        
        # Calculate end point based on slope and length
        # length = sqrt((x2-x1)^2 + (y2-y1)^2)
        # slope = (y2-y1)/(x2-x1)
        # So: y2-y1 = slope * (x2-x1)
        # And: length^2 = (x2-x1)^2 + (slope * (x2-x1))^2
        # Therefore: (x2-x1) = length / sqrt(1 + slope^2)
        
        dx = length / (1 + slope**2)**0.5
        dy = slope * dx
        
        x2 = int(round(x1 + dx))
        y2 = int(round(y1 + dy))
        
        # Ensure points are within grid bounds
        x2 = max(self.MIN_COORDINATE, min(self.grid_size, x2))
        y2 = max(self.MIN_COORDINATE, min(self.grid_size, y2))
        
        points = [(x1, y1), (x2, y2)]
        t_values = [0.0, 1.0]
        return points, t_values 