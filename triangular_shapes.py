from base_shapes import MultiStrokeShape
from typing import List, Tuple, Dict, Any
import random
import math
import sys
import os

class TriangularShape(MultiStrokeShape):
    """Generator for triangular shapes using natural multi-stroke drawing approach.
    
    Triangles are drawn with two strokes:
    1. First stroke: Two sides of the triangle (V-shape or L-shape)
    2. Second stroke: Closing line to complete the triangle
    
    This approach is much more natural than forcing a triangle into a single curve.
    """
    
    # Triangular shape constants
    MIN_SIZE = 8  # Minimum side length
    MAX_SIZE_RATIO = 0.6  # Maximum size as ratio of grid
    MIN_SEPARATION = 3  # Minimum separation from edges
    MIN_AREA_RATIO = 0.003  # Minimum area as ratio of grid area (prevents line-like triangles)
    
    def __init__(self, grid_size: int = MultiStrokeShape.DEFAULT_GRID_SIZE):
        super().__init__(grid_size)
        self.shape_id = "triangular"
        
        # Calculate triangular shape parameters
        self.min_size = max(self.MIN_SIZE, int(self.grid_size * 0.08))
        self.max_size = int(self.grid_size * self.MAX_SIZE_RATIO)
        self.margin = max(self.edge_margin, self.MIN_SEPARATION)
    
        # Calculate minimum area to prevent line-like triangles
        grid_area = self.grid_size * self.grid_size
        self.min_area = max(20, int(grid_area * self.MIN_AREA_RATIO))  # At least 20 square units
    
    def _calculate_triangle_area(self, vertex1: Tuple[int, int], vertex2: Tuple[int, int], vertex3: Tuple[int, int]) -> float:
        """Calculate the area of a triangle using the shoelace formula."""
        x1, y1 = vertex1
        x2, y2 = vertex2
        x3, y3 = vertex3
        
        # Shoelace formula: Area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
        area = 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        return area
    
    def generate_sample(self, apply_random_transforms: bool = True, start_point: Tuple[int, int] = None) -> Dict[str, Any]:
        """Generate a random triangular shape sample."""
        
        # Clear any existing strokes
        self.clear_strokes()
        
        # Choose triangle type
        triangle_type = random.choice(['equilateral', 'isosceles', 'right', 'scalene'])
        
        # Generate the triangle using multiple strokes
        shape_info = self._generate_triangular_shape(triangle_type, start_point)
        
        # Apply transformations if requested
        if apply_random_transforms:
            transformations = self._generate_random_transformations()
            self.apply_transformations_to_all_strokes(transformations)
        else:
            transformations = {'scale': 1.0, 'shift_x': 0, 'shift_y': 0, 'rotation_deg': 0.0}
        
        return {
            'shape_id': self.shape_id,
            'shape_type': triangle_type,
            'transformations': transformations,
            'shape_info': shape_info,
            'strokes': [{'points': stroke.points, 't_values': stroke.t_values, 'formatted_points': stroke.formatted_points, 'stroke_id': stroke.stroke_id} for stroke in self.strokes],
            'stroke_count': self.get_stroke_count()
        }
    
    def generate_shape_type(self, shape_type: str) -> Tuple[List[Tuple[int, int]], List[float]]:
        """Generate a specific triangular shape type - returns strokes separately."""
        self.clear_strokes()
        self._generate_triangular_shape(shape_type)
        
        # Return first stroke for legacy compatibility (this method shouldn't be used for multi-stroke)
        if self.strokes:
            return self.strokes[0].points, self.strokes[0].t_values
        else:
            return [], []
    
    def _generate_triangular_shape(self, triangle_type: str, start_point: Tuple[int, int] = None) -> Dict[str, Any]:
        """Generate a triangular shape using natural multi-stroke approach."""
        
        if triangle_type == 'equilateral':
            return self._generate_equilateral_triangle(start_point)
        elif triangle_type == 'isosceles':
            return self._generate_isosceles_triangle(start_point)
        elif triangle_type == 'right':
            return self._generate_right_triangle(start_point)
        elif triangle_type == 'scalene':
            return self._generate_scalene_triangle(start_point)
        else:
            # Default to equilateral
            return self._generate_equilateral_triangle(start_point)
    
    def _generate_equilateral_triangle(self, start_point: Tuple[int, int] = None) -> Dict[str, Any]:
        """Generate an equilateral triangle using two natural strokes."""
        
        max_attempts = 20
        for attempt in range(max_attempts):
            # Choose triangle size
            max_possible_size = min(self.max_size, self.grid_size - 2 * self.margin)
            side_length = random.randint(self.min_size, max_possible_size)
            
            # Calculate height of equilateral triangle
            height = int(side_length * math.sqrt(3) / 2)
            
            if start_point is not None:
                # Use provided start point as bottom-left vertex
                bottom_left_x, bottom_left_y = start_point
                # Ensure the triangle fits within grid bounds
                bottom_left_x = max(self.margin, min(self.grid_size - self.margin - side_length, bottom_left_x))
                bottom_left_y = max(self.margin, min(self.grid_size - self.margin - height, bottom_left_y))
                
                # Calculate other vertices based on bottom-left position
                vertex_bottom_left = (bottom_left_x, bottom_left_y)
                vertex_bottom_right = (bottom_left_x + side_length, bottom_left_y)
                vertex_top = (bottom_left_x + side_length//2, bottom_left_y + height)
                center_x = bottom_left_x + side_length//2
                center_y = bottom_left_y + height//2
            else:
                # Choose center position to ensure triangle fits (random)
                center_x = random.randint(self.margin + side_length//2, self.grid_size - self.margin - side_length//2)
                center_y = random.randint(self.margin + height//2, self.grid_size - self.margin - height//2)
                
                # Calculate vertices for equilateral triangle (pointing up)
                vertex_top = (center_x, center_y + height//2)  # Top vertex
                vertex_bottom_left = (center_x - side_length//2, center_y - height//2)  # Bottom-left
                vertex_bottom_right = (center_x + side_length//2, center_y - height//2)  # Bottom-right
                
                # Ensure all vertices are within grid bounds
                vertices = []
                for x, y in [vertex_top, vertex_bottom_left, vertex_bottom_right]:
                    x = max(self.margin, min(self.grid_size - self.margin, x))
                    y = max(self.margin, min(self.grid_size - self.margin, y))
                    vertices.append((x, y))
                
                vertex_top, vertex_bottom_left, vertex_bottom_right = vertices
            
            # Check triangle area (for equilateral: area = side_length² * √3 / 4)
            area = self._calculate_triangle_area(vertex_bottom_left, vertex_bottom_right, vertex_top)
            if area >= self.min_area:
                break
        
        # If we couldn't generate a valid triangle, create a minimal valid one
        if attempt >= max_attempts - 1:
            # Create equilateral triangle with guaranteed minimum area
            safe_side = max(self.min_size, int(math.sqrt(self.min_area * 4 / math.sqrt(3))))
            safe_height = int(safe_side * math.sqrt(3) / 2)
            
            if start_point is not None:
                start_x, start_y = start_point
                start_x = max(self.margin, min(self.grid_size - self.margin - safe_side, start_x))
                start_y = max(self.margin, min(self.grid_size - self.margin - safe_height, start_y))
                
                vertex_bottom_left = (start_x, start_y)
                vertex_bottom_right = (start_x + safe_side, start_y)
                vertex_top = (start_x + safe_side//2, start_y + safe_height)
                center_x = start_x + safe_side//2
                center_y = start_y + safe_height//2
            else:
                center_x = self.grid_size // 2
                center_y = self.grid_size // 2
                
                vertex_bottom_left = (center_x - safe_side//2, center_y - safe_height//2)
                vertex_bottom_right = (center_x + safe_side//2, center_y - safe_height//2)
                vertex_top = (center_x, center_y + safe_height//2)
                
            vertices = [vertex_top, vertex_bottom_left, vertex_bottom_right]
            side_length = safe_side
            height = safe_height
        
        # Generate strokes following the example pattern
        # Stroke 1: V-shape from bottom-left to top to bottom-right (like the example)
        stroke1_points = [vertex_bottom_left, vertex_top, vertex_top, vertex_bottom_right]
        stroke1_t_values = [0.00, 0.50, 0.50, 1.00]  # Corner duplication at top
        
        # Stroke 2: Closing line from bottom-right back to bottom-left
        stroke2_points = [vertex_bottom_right, vertex_bottom_left]
        stroke2_t_values = [0.00, 1.00]
        
        # Add strokes to the shape
        self.add_stroke(stroke1_points, stroke1_t_values, 's1')
        self.add_stroke(stroke2_points, stroke2_t_values, 's2')
        
        shape_info = {
            'side_length': side_length,
            'height': height,
            'center': (center_x, center_y),
            'vertices': vertices,
            'stroke_pattern': 'v_shape_plus_closing_line',
            'area': self._calculate_triangle_area(vertex_bottom_left, vertex_bottom_right, vertex_top)
        }
        
        return shape_info
    
    def _generate_isosceles_triangle(self, start_point: Tuple[int, int] = None) -> Dict[str, Any]:
        """Generate an isosceles triangle using two natural strokes."""
        
        max_attempts = 20
        for attempt in range(max_attempts):
            # Choose triangle dimensions
            base_length = random.randint(self.min_size, min(self.max_size, self.grid_size - 2 * self.margin))
            height = random.randint(self.min_size, min(self.max_size, self.grid_size - 2 * self.margin))
            
            if start_point is not None:
                # Use provided start point as bottom-left vertex
                bottom_left_x, bottom_left_y = start_point
                # Ensure the triangle fits within grid bounds
                bottom_left_x = max(self.margin, min(self.grid_size - self.margin - base_length, bottom_left_x))
                bottom_left_y = max(self.margin, min(self.grid_size - self.margin - height, bottom_left_y))
                
                # Calculate vertices for isosceles triangle (pointing up)
                vertex_bottom_left = (bottom_left_x, bottom_left_y)
                vertex_bottom_right = (bottom_left_x + base_length, bottom_left_y)
                vertex_top = (bottom_left_x + base_length//2, bottom_left_y + height)
            else:
                # Choose position randomly to ensure triangle fits
                center_x = random.randint(self.margin + base_length//2, self.grid_size - self.margin - base_length//2)
                center_y = random.randint(self.margin + height//2, self.grid_size - self.margin - height//2)
                
                # Calculate vertices for isosceles triangle (pointing up)
                vertex_top = (center_x, center_y + height//2)
                vertex_bottom_left = (center_x - base_length//2, center_y - height//2)
                vertex_bottom_right = (center_x + base_length//2, center_y - height//2)
                
                # Ensure all vertices are within bounds
                vertices = []
                for x, y in [vertex_top, vertex_bottom_left, vertex_bottom_right]:
                    x = max(self.margin, min(self.grid_size - self.margin, x))
                    y = max(self.margin, min(self.grid_size - self.margin, y))
                    vertices.append((x, y))
                
                vertex_top, vertex_bottom_left, vertex_bottom_right = vertices
            
            # Check triangle area to ensure it's not line-like
            area = self._calculate_triangle_area(vertex_bottom_left, vertex_bottom_right, vertex_top)
            if area >= self.min_area:
                break
        
        # If we couldn't generate a valid triangle, create a minimal valid one
        if attempt >= max_attempts - 1:
            # Create a triangle with guaranteed minimum area
            safe_base = max(self.min_size, int(math.sqrt(self.min_area * 2)))
            safe_height = max(self.min_size, int(self.min_area * 2 / safe_base))
            
            if start_point is not None:
                start_x, start_y = start_point
                start_x = max(self.margin, min(self.grid_size - self.margin - safe_base, start_x))
                start_y = max(self.margin, min(self.grid_size - self.margin - safe_height, start_y))
                
                vertex_bottom_left = (start_x, start_y)
                vertex_bottom_right = (start_x + safe_base, start_y)
                vertex_top = (start_x + safe_base//2, start_y + safe_height)
            else:
                center_x = self.grid_size // 2
                center_y = self.grid_size // 2
                
                vertex_bottom_left = (center_x - safe_base//2, center_y - safe_height//2)
                vertex_bottom_right = (center_x + safe_base//2, center_y - safe_height//2)
                vertex_top = (center_x, center_y + safe_height//2)
                
            vertices = [vertex_top, vertex_bottom_left, vertex_bottom_right]
        
        # Generate strokes
        # Stroke 1: V-shape from bottom-left to top to bottom-right
        stroke1_points = [vertex_bottom_left, vertex_top, vertex_top, vertex_bottom_right]
        stroke1_t_values = [0.00, 0.50, 0.50, 1.00]
        
        # Stroke 2: Closing line
        stroke2_points = [vertex_bottom_right, vertex_bottom_left]
        stroke2_t_values = [0.00, 1.00]
        
        self.add_stroke(stroke1_points, stroke1_t_values, 's1')
        self.add_stroke(stroke2_points, stroke2_t_values, 's2')
        
        shape_info = {
            'base_length': base_length,
            'height': height,
            'center': (center_x, center_y),
            'vertices': vertices,
            'stroke_pattern': 'v_shape_plus_closing_line',
            'area': self._calculate_triangle_area(vertex_bottom_left, vertex_bottom_right, vertex_top)
        }
        
        return shape_info
    
    def _generate_right_triangle(self, start_point: Tuple[int, int] = None) -> Dict[str, Any]:
        """Generate a right triangle using two natural strokes."""
        
        max_attempts = 20
        for attempt in range(max_attempts):
            # Choose triangle dimensions
            base_length = random.randint(self.min_size, min(self.max_size, self.grid_size - 2 * self.margin))
            height = random.randint(self.min_size, min(self.max_size, self.grid_size - 2 * self.margin))
            
            if start_point is not None:
                # Use provided start point as the right angle vertex
                start_x, start_y = start_point
                # Ensure the triangle fits within grid bounds
                start_x = max(self.margin, min(self.grid_size - self.margin - base_length, start_x))
                start_y = max(self.margin, min(self.grid_size - self.margin - height, start_y))
            else:
                # Choose position randomly to ensure triangle fits
                start_x = random.randint(self.margin, self.grid_size - self.margin - base_length)
                start_y = random.randint(self.margin, self.grid_size - self.margin - height)
            
            # Calculate vertices for right triangle (right angle at bottom-left)
            vertex_right_angle = (start_x, start_y)  # Bottom-left (right angle)
            vertex_bottom_right = (start_x + base_length, start_y)  # Bottom-right
            vertex_top_left = (start_x, start_y + height)  # Top-left
            
            # Check triangle area to ensure it's not line-like
            area = self._calculate_triangle_area(vertex_right_angle, vertex_bottom_right, vertex_top_left)
            if area >= self.min_area:
                break
        
        # If we couldn't generate a valid triangle, create a minimal valid one
        if attempt >= max_attempts - 1:
            # Create a right triangle with guaranteed minimum area (area = 0.5 * base * height)
            safe_base = max(self.min_size, int(math.sqrt(self.min_area * 2)))
            safe_height = max(self.min_size, int(self.min_area * 2 / safe_base))
            
            if start_point is not None:
                start_x, start_y = start_point
                start_x = max(self.margin, min(self.grid_size - self.margin - safe_base, start_x))
                start_y = max(self.margin, min(self.grid_size - self.margin - safe_height, start_y))
            else:
                start_x = self.grid_size // 2 - safe_base // 2
                start_y = self.grid_size // 2 - safe_height // 2
            
            vertex_right_angle = (start_x, start_y)
            vertex_bottom_right = (start_x + safe_base, start_y)
            vertex_top_left = (start_x, start_y + safe_height)
        
        # Generate strokes - for right triangle, use L-shape pattern
        # Stroke 1: L-shape from bottom-right to bottom-left (right angle) to top-left
        stroke1_points = [vertex_bottom_right, vertex_right_angle, vertex_right_angle, vertex_top_left]
        stroke1_t_values = [0.00, 0.50, 0.50, 1.00]  # Corner duplication at right angle
        
        # Stroke 2: Closing line (hypotenuse) from top-left back to bottom-right
        stroke2_points = [vertex_top_left, vertex_bottom_right]
        stroke2_t_values = [0.00, 1.00]
        
        self.add_stroke(stroke1_points, stroke1_t_values, 's1')
        self.add_stroke(stroke2_points, stroke2_t_values, 's2')
        
        shape_info = {
            'base_length': base_length,
            'height': height,
            'right_angle_at': vertex_right_angle,
            'vertices': [vertex_right_angle, vertex_bottom_right, vertex_top_left],
            'stroke_pattern': 'l_shape_plus_hypotenuse',
            'area': self._calculate_triangle_area(vertex_right_angle, vertex_bottom_right, vertex_top_left)
        }
        
        return shape_info
    
    def _generate_scalene_triangle(self, start_point: Tuple[int, int] = None) -> Dict[str, Any]:
        """Generate a scalene triangle using two natural strokes."""
        
        if start_point is not None:
            # Use provided start point as first vertex
            first_vertex = start_point
            # Generate two additional vertices to form a scalene triangle
            vertices = [first_vertex]
            
            # Generate second vertex within reasonable distance
            min_dist = self.min_size
            max_dist = min(self.max_size, self.grid_size // 2)
            
            for _ in range(2):  # Generate 2 more vertices
                attempts = 0
                while attempts < 10:
                    x = random.randint(self.margin, self.grid_size - self.margin)
                    y = random.randint(self.margin, self.grid_size - self.margin)
                    
                    # Check distance from existing vertices
                    valid = True
                    for existing_vertex in vertices:
                        dist = math.sqrt((x - existing_vertex[0])**2 + (y - existing_vertex[1])**2)
                        if dist < min_dist or dist > max_dist:
                            valid = False
                            break
                    
                    if valid:
                        vertices.append((x, y))
                        break
                    attempts += 1
                
                # If we couldn't find a good vertex, use a constructed one
                if len(vertices) == 1:  # Still only have first vertex
                    angle = random.uniform(0, 2 * math.pi)
                    dist = random.randint(min_dist, max_dist)
                    x = int(first_vertex[0] + dist * math.cos(angle))
                    y = int(first_vertex[1] + dist * math.sin(angle))
                    x = max(self.margin, min(self.grid_size - self.margin, x))
                    y = max(self.margin, min(self.grid_size - self.margin, y))
                    vertices.append((x, y))
                elif len(vertices) == 2:  # Have first two vertices
                    # Create third vertex that makes a scalene triangle
                    angle = random.uniform(0, 2 * math.pi)
                    dist = random.randint(min_dist, max_dist)
                    # Use first vertex as reference
                    x = int(first_vertex[0] + dist * math.cos(angle))
                    y = int(first_vertex[1] + dist * math.sin(angle))
                    x = max(self.margin, min(self.grid_size - self.margin, x))
                    y = max(self.margin, min(self.grid_size - self.margin, y))
                    vertices.append((x, y))
                    
            # Check if generated triangle has sufficient area
            if len(vertices) == 3:
                area = self._calculate_triangle_area(vertices[0], vertices[1], vertices[2])
                if area < self.min_area:
                    # Regenerate with larger guaranteed area
                    vertices = self._create_fallback_scalene_triangle(start_point)
        else:
            # Generate three vertices with different distances to ensure scalene (original logic)
            attempts = 0
            max_attempts = 20
            valid_triangle_found = False
            
            while attempts < max_attempts and not valid_triangle_found:
                # Choose three random points within bounds
                vertices = []
                for _ in range(3):
                    x = random.randint(self.margin, self.grid_size - self.margin)
                    y = random.randint(self.margin, self.grid_size - self.margin)
                    vertices.append((x, y))
                
                # Calculate side lengths
                side_lengths = []
                for i in range(3):
                    j = (i + 1) % 3
                    dist = math.sqrt((vertices[j][0] - vertices[i][0])**2 + (vertices[j][1] - vertices[i][1])**2)
                    side_lengths.append(dist)
                
                # Calculate area
                area = self._calculate_triangle_area(vertices[0], vertices[1], vertices[2])
                
                # Check if triangle is valid, scalene, and has sufficient area
                min_side_diff = 3
                if (all(side > self.min_size for side in side_lengths) and
                    all(abs(side_lengths[i] - side_lengths[j]) > min_side_diff 
                        for i in range(3) for j in range(i+1, 3)) and
                    side_lengths[0] + side_lengths[1] > side_lengths[2] and
                    side_lengths[1] + side_lengths[2] > side_lengths[0] and
                    side_lengths[0] + side_lengths[2] > side_lengths[1] and
                    area >= self.min_area):
                    valid_triangle_found = True
                    break
                
                attempts += 1
            
            # If we couldn't generate a good scalene triangle, create a constructed one
            if not valid_triangle_found:
                vertices = self._create_fallback_scalene_triangle()
        
        # Generate strokes - use the first vertex as starting point
        # Stroke 1: Two sides forming a V or L shape
        stroke1_points = [vertices[0], vertices[1], vertices[1], vertices[2]]
        stroke1_t_values = [0.00, 0.50, 0.50, 1.00]
        
        # Stroke 2: Closing line
        stroke2_points = [vertices[2], vertices[0]]
        stroke2_t_values = [0.00, 1.00]
        
        self.add_stroke(stroke1_points, stroke1_t_values, 's1')
        self.add_stroke(stroke2_points, stroke2_t_values, 's2')
        
        # Calculate final side lengths for info
        final_side_lengths = []
        for i in range(3):
            j = (i + 1) % 3
            dist = math.sqrt((vertices[j][0] - vertices[i][0])**2 + (vertices[j][1] - vertices[i][1])**2)
            final_side_lengths.append(dist)
        
        shape_info = {
            'side_lengths': final_side_lengths,
            'vertices': vertices,
            'stroke_pattern': 'two_sides_plus_closing_line',
            'area': self._calculate_triangle_area(vertices[0], vertices[1], vertices[2])
        }
        
        return shape_info 
    
    def _create_fallback_scalene_triangle(self, start_point: Tuple[int, int] = None) -> List[Tuple[int, int]]:
        """Create a scalene triangle with guaranteed minimum area."""
        # Calculate safe dimensions that ensure minimum area
        safe_base = max(self.min_size, int(math.sqrt(self.min_area * 2)))
        safe_height = max(self.min_size, int(self.min_area * 2 / safe_base))
        
        if start_point is not None:
            start_x, start_y = start_point
            start_x = max(self.margin, min(self.grid_size - self.margin - safe_base, start_x))
            start_y = max(self.margin, min(self.grid_size - self.margin - safe_height, start_y))
        else:
            start_x = self.grid_size // 2 - safe_base // 2
            start_y = self.grid_size // 2 - safe_height // 2
        
        # Create scalene by offsetting the top vertex significantly
        offset = safe_base // 3 + random.randint(1, safe_base // 4)  # Ensure meaningful offset
        
        vertices = [
            (start_x, start_y),  # Bottom-left
            (start_x + safe_base, start_y),  # Bottom-right
            (max(self.margin, min(self.grid_size - self.margin, start_x + safe_base//2 + offset)), start_y + safe_height)  # Top (offset)
        ]
        
        return vertices 