from base_shapes import BaseShape, MultiStrokeShape
from straight_line import StraightLine
from curves import Curve
from rectangular_shapes import RectangularShape
from triangular_shapes import TriangularShape
from circular_shapes import CircularShape
from typing import List, Tuple, Dict, Any, Optional
import random
import math
import sys
import os
import ast

# Import utils from parent directory
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import SketchAgent.utils as utils

class SpatialRelationsGenerator(MultiStrokeShape):
    """Generator for spatial relations tasks - adding connected strokes to existing shapes.
    
    Task 1: Generate a base shape, then add a connected stroke in one of 8 directions:
    top, top-right, top-left, bottom, bottom-right, bottom-left, left, right
    """
    
    # 8-directional vectors (dx, dy) - normalized directions
    DIRECTIONS = {
        'top': (0, 1),
        'top_right': (1, 1), 
        'top_left': (-1, 1),
        'bottom': (0, -1),
        'bottom_right': (1, -1),
        'bottom_left': (-1, -1),
        'right': (1, 0),
        'left': (-1, 0),
        'inside': (0, 0)  # Special case: shape placed inside base shape
    }
    
    # Stroke length parameters
    MIN_STROKE_LENGTH = 15
    MAX_STROKE_LENGTH = 40
    
    def __init__(self, grid_size: int = MultiStrokeShape.DEFAULT_GRID_SIZE):
        super().__init__(grid_size)
        self.shape_id = "spatial_relation"
        
        # Initialize base shape generators (keeping for potential future use)
        self.generators = {
            'rectangle': RectangularShape(grid_size),
            'triangle': TriangularShape(grid_size),
            'circle': CircularShape(grid_size),
        }
        self.stroke_types = ['line', 'curve', 'rectangle', 'triangle', 'circle', 'arc', 'ellipse', 'rhombus', 'square']
        
    
    def generate_sample(self, apply_random_transforms: bool = True, base_shape_type: str = None, direction: str = None, stroke_type: str = None) -> Dict[str, Any]:
        """Generate a spatial relations sample with base shape + connected stroke."""
        apply_random_transforms = False
        # Clear any existing strokes to prevent accumulation
        self.clear_strokes()
        
        # Generate base shape with controlled size
        if base_shape_type is None:
            base_shape_type = random.choice(list(self.generators.keys()))
        base_generator = self.generators[base_shape_type]
        base_sample = base_generator.generate_sample(apply_random_transforms=False)
        
        # Select direction for connected stroke
        if direction is None:
            direction = random.choice(list(self.DIRECTIONS.keys()))
        
        # Apply controlled scaling to make base shape smaller
        # This leaves more room for connected strokes
        base_scale_factor = random.uniform(0.6, 0.8) if direction != 'inside' else 1.0 # Make base shape 60-80% of original size
        base_sample = self._apply_base_shape_scaling(base_sample, base_scale_factor)
        
        
        # Randomly choose stroke type to attach
        if stroke_type is None:
            stroke_type = random.choice(self.stroke_types)
        
        # Calculate intersection point and generate connected stroke
        intersection_point, connected_stroke = self._generate_connected_stroke(
            base_sample, direction, stroke_type
        )
        
        # Add strokes to multi-stroke shape
        if base_sample.get('strokes'):
            # Multi-stroke base shape - add all base strokes
            for i, stroke in enumerate(base_sample['strokes']):
                stroke_id = f's{i+1}'
                self.add_stroke(stroke['points'], stroke['t_values'], stroke_id)
            next_stroke_id = f's{len(base_sample["strokes"])+1}'
        else:
            # Single-stroke base shape
            self.add_stroke(base_sample['raw_points'], base_sample['t_values'], 's1')
            next_stroke_id = 's2'
        
        # Add connected stroke
        if connected_stroke.get('is_multistroke', False):
            # Handle multistroke inscribed shapes
            for i, stroke_data in enumerate(connected_stroke['strokes']):
                stroke_id = f's{len(base_sample.get("strokes", [])) + i + 1}' if base_sample.get('strokes') else f's{i + 2}'
                self.add_stroke(stroke_data['points'], stroke_data['t_values'], stroke_id)
        else:
            if len(connected_stroke['points']) != len(connected_stroke['t_values']):
                connected_stroke['t_values'] = [i / (len(connected_stroke['points']) - 1) for i in range(len(connected_stroke['points']))]
            self.add_stroke(connected_stroke['points'], connected_stroke['t_values'], next_stroke_id)
        
        transformations = {'scale': 1.0, 'shift_x': 0, 'shift_y': 0, 'rotation_deg': 0.0}
        
        # Format output
        return {
            'shape_id': self.shape_id,
            'shape_type': f'{base_shape_type}_with_{direction}_stroke',
            'strokes': self.format_strokes_for_prompt(),
            'stroke_count': len(self.strokes),
            'transformations': transformations,
            'spatial_info': {
                'base_stroke_count': max(1, len(base_sample.get('strokes', []))),
                'base_shape_type': base_shape_type,
                'direction': direction,
                'intersection_point': intersection_point,
                'base_shape_info': base_sample.get('shape_info', {}),
                'connected_stroke_length': connected_stroke['length'],
                'connected_stroke_type': connected_stroke['stroke_type'],
                'is_multistroke_inscribed': connected_stroke.get('is_multistroke', False)
            }
        }
    
    def _generate_connected_stroke(self, base_sample: Dict[str, Any], direction: str, stroke_type: str) -> Tuple[Tuple[int, int], Dict[str, Any]]:
        """Generate a connected stroke in the specified direction from the base shape."""
        
        # Special handling for 'inside' direction
        if direction == 'inside':
            return self._generate_inscribed_stroke(base_sample, stroke_type)
        
        # Step 1: Calculate center of base shape
        center = self._calculate_shape_center(base_sample)
        
        # Step 2: Find intersection point using curve fitting for more precise detection
        intersection_point = self._find_intersection_point(base_sample, center, direction)
        
        # Step 3: Generate connected stroke
        connected_stroke = self._generate_shape_stroke(intersection_point, direction, stroke_type)
        
        return intersection_point, connected_stroke
    
    def _apply_base_shape_scaling(self, base_sample: Dict[str, Any], scale_factor: float) -> Dict[str, Any]:
        """Apply controlled scaling to base shape to make it smaller."""
        
        if base_sample.get('strokes'):
            # Multi-stroke shape - scale each stroke's points
            scaled_strokes = []
            for stroke_data in base_sample['strokes']:
                # Convert formatted points back to coordinate tuples
                raw_points = stroke_data['points']
                
                # Apply scaling using BaseShape's transformation method
                # We'll use a temporary BaseShape instance for the transformation
                temp_shape = BaseShape(self.grid_size)
                scaled_points = temp_shape.apply_transformations(
                    raw_points, 
                    scale=scale_factor, 
                    shift_x=0, 
                    shift_y=0, 
                    rotation_deg=0.0
                )
                
                # Update stroke data
                scaled_stroke = stroke_data.copy()
                scaled_stroke['points'] = scaled_points
                scaled_stroke['formatted_points'] = [f'x{x}y{y}' for x, y in scaled_points]
                scaled_strokes.append(scaled_stroke)
            
            # Update base sample
            base_sample['strokes'] = scaled_strokes
            
        else:
            # Single-stroke shape - scale raw_points
            if 'raw_points' in base_sample:
                temp_shape = BaseShape(self.grid_size)
                scaled_points = temp_shape.apply_transformations(
                    base_sample['raw_points'],
                    scale=scale_factor,
                    shift_x=0,
                    shift_y=0, 
                    rotation_deg=0.0
                )
                
                # Update base sample
                base_sample['raw_points'] = scaled_points
                base_sample['points'] = [f'x{x}y{y}' for x, y in scaled_points]
        
        return base_sample
    
    def _generate_shape_stroke(self, intersection_point: Tuple[int, int], direction: str, stroke_type: str) -> Dict[str, Any]:
        """Generate a connected stroke using the new clean start_point approach with directional constraints."""
        
        # Calculate directional ranges based on intersection point and direction
        x_range, y_range = self._calculate_directional_ranges(intersection_point, direction)
        
        try:
            # Generate the attached shape directly at the intersection point using start_point and ranges
            if stroke_type == 'line':
                generator = StraightLine(self.grid_size)
                points, t_values = generator.generate_base_shape(
                    start_point=intersection_point,
                    x_range=x_range,
                    y_range=y_range
                )
                
                return {
                    'points': points,
                    't_values': t_values,
                    'length': ((points[1][0] - points[0][0])**2 + (points[1][1] - points[0][1])**2)**0.5,
                    'stroke_type': 'line'
                }
                
            elif stroke_type == 'curve':
                generator = Curve(self.grid_size)
                sample = generator.generate_sample(
                    apply_random_transforms=False,
                    start_point=intersection_point,
                    x_range=x_range,
                    y_range=y_range
                )
                
                return {
                    'points': sample['raw_points'],
                    't_values': sample['t_values'],
                    'length': 20,  # Approximate length
                    'stroke_type': 'curve'
                }
                
            elif stroke_type == 'rectangle':
                generator = RectangularShape(self.grid_size)
                sample = generator.generate_sample(
                    apply_random_transforms=False,
                    start_point=intersection_point,
                    x_range=x_range,
                    y_range=y_range
                )
                
                return {
                    'points': sample['raw_points'],
                    't_values': sample['t_values'],
                    'length': 30,  # Approximate length
                    'stroke_type': 'rectangle'
                }
                
            elif stroke_type == 'circle':
                generator = CircularShape(self.grid_size)
                sample = generator.generate_sample(
                    apply_random_transforms=False,
                    start_point=intersection_point,  # Used as center for circles
                    x_range=x_range,
                    y_range=y_range
                )
                
                return {
                    'points': sample['raw_points'],
                    't_values': sample['t_values'],
                    'length': 25,  # Approximate length
                    'stroke_type': 'circle'
                }
                
            elif stroke_type == 'triangle':
                # For triangles, we'll generate a simple single-stroke triangle as fallback
                # since the full multi-stroke triangle is more complex to constrain
                # This creates a triangular path that respects the directional constraints
                return self._generate_simple_triangle_stroke(intersection_point, direction, x_range, y_range)
                
            elif stroke_type == 'arc':
                # Generate an arc using the curve generator with directional constraints support
                generator = Curve(self.grid_size)
                points, t_values = generator._generate_constrained_arc(intersection_point, x_range, y_range)
                
                return {
                    'points': points,
                    't_values': t_values,
                    'length': 15,  # Approximate length
                    'stroke_type': 'arc'
                }
                
            elif stroke_type == 'ellipse':
                # Generate an ellipse using the circular generator with ellipse generation
                generator = CircularShape(self.grid_size)
                points, t_values, shape_info = generator._generate_ellipse(intersection_point, x_range, y_range)
                
                return {
                    'points': points,
                    't_values': t_values,
                    'length': 25,
                    'stroke_type': 'ellipse'
                }
                
            elif stroke_type == 'rhombus':
                # Generate a rhombus using the rectangular generator with rhombus type
                generator = RectangularShape(self.grid_size)
                points, t_values, shape_info = generator._generate_rhombus(intersection_point, x_range, y_range)
                
                return {
                    'points': points,
                    't_values': t_values,
                    'length': 35,  # Approximate length
                    'stroke_type': 'rhombus'
                }
                
            elif stroke_type == 'square':
                # Generate a square using the rectangular generator with square type
                generator = RectangularShape(self.grid_size)
                points, t_values, shape_info = generator._generate_square(intersection_point, x_range, y_range)
                
                return {
                    'points': points,
                    't_values': t_values,
                    'length': 30,  # Approximate length
                    'stroke_type': 'square'
                }
                
            else:
                # Fallback to line for any other stroke type
                return self._generate_straight_stroke(intersection_point, direction)
                
        except Exception as e:
            print(f"Error generating {stroke_type} at {intersection_point}: {e}")
            # Fallback to simple line
            return self._generate_straight_stroke(intersection_point, direction)
    
    def _calculate_directional_ranges(self, intersection_point: Tuple[int, int], direction: str) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Calculate x_range and y_range based on intersection point and desired direction.
        
        Direction logic:
        - up: any X, only Y > start_y
        - right: X > start_x, any Y  
        - top-right: X > start_x, Y > start_y
        - down: any X, only Y < start_y
        - left: X < start_x, any Y
        - bottom-left: X < start_x, Y < start_y
        - etc.
        """
        start_x, start_y = intersection_point
        margin = self.edge_margin
        
        # Default full ranges
        full_x_range = (margin, self.grid_size - margin)
        full_y_range = (margin, self.grid_size - margin)
        
        # Direction-specific constraints
        if direction == 'top':
            # up: any X, only Y > start_y
            x_range = full_x_range
            y_range = (start_y, self.grid_size - margin)
            
        elif direction == 'bottom':
            # down: any X, only Y < start_y
            x_range = full_x_range
            y_range = (margin, start_y)
            
        elif direction == 'right':
            # right: X > start_x, any Y
            x_range = (start_x, self.grid_size - margin)
            y_range = full_y_range
            
        elif direction == 'left':
            # left: X < start_x, any Y
            x_range = (margin, start_x)
            y_range = full_y_range
            
        elif direction == 'top_right':
            # top-right: X > start_x, Y > start_y
            x_range = (start_x, self.grid_size - margin)
            y_range = (start_y, self.grid_size - margin)
            
        elif direction == 'top_left':
            # top-left: X < start_x, Y > start_y
            x_range = (margin, start_x)
            y_range = (start_y, self.grid_size - margin)
            
        elif direction == 'bottom_right':
            # bottom-right: X > start_x, Y < start_y
            x_range = (start_x, self.grid_size - margin)
            y_range = (margin, start_y)
            
        elif direction == 'bottom_left':
            # bottom-left: X < start_x, Y < start_y
            x_range = (margin, start_x)
            y_range = (margin, start_y)
            
        else:
            # Unknown direction - use full ranges
            x_range = full_x_range
            y_range = full_y_range
        
        # Ensure ranges are valid (min < max)
        x_range = (min(x_range[0], x_range[1] - 1), x_range[1])
        y_range = (min(y_range[0], y_range[1] - 1), y_range[1])
        
        return x_range, y_range
    
    def _calculate_shape_center(self, base_sample: Dict[str, Any]) -> Tuple[int, int]:
        """Calculate the center point of the base shape."""
        
        if base_sample.get('strokes'):
            # Multi-stroke shape - use all points
            all_points = []
            for stroke in base_sample['strokes']:
                all_points.extend(stroke['points'])
        else:
            # Single-stroke shape
            all_points = base_sample['raw_points']
        
        if not all_points:
            return (self.grid_size // 2, self.grid_size // 2)
        
        # Calculate centroid
        center_x = sum(point[0] for point in all_points) // len(all_points)
        center_y = sum(point[1] for point in all_points) // len(all_points)
        
        return (center_x, center_y)
    
    def _find_intersection_point(self, base_sample: Dict[str, Any], center: Tuple[int, int], direction: str) -> Tuple[int, int]:
        """Find intersection point using curve fitting for more precise detection."""
        
        direction_vector = self.DIRECTIONS[direction]
        
        try:
            # Step 1: Convert base_sample to XML format for utils processing
            xml_content = self._create_sketch_xml(base_sample)
            
            # Step 2: Parse with utils to get strokes and t_values
            strokes_list_str, t_values_str = utils.parse_xml_string(xml_content, self.grid_size)
            strokes_list, t_values = ast.literal_eval(strokes_list_str), ast.literal_eval(t_values_str)
            
            # Step 3: Get control points using utils
            cell_size = 6  # Use same as prompt_visualizer_app.py  
            cells_to_pixels_map = utils.cells_to_pixels(self.grid_size, cell_size, header_size=cell_size)
            all_control_points = utils.get_control_points(strokes_list, t_values, cells_to_pixels_map)
            
            # Step 4: Sample 10-20 points along each curve and find best point in direction
            all_sampled_points = []
            
            for stroke_control_points in all_control_points:
                for curve_control_points in stroke_control_points:  # Handle sub-curves
                    # Sample 15 points along this curve
                    for i in range(15):
                        t = i / 14.0  # t from 0 to 1
                        if len(curve_control_points) >= 4:
                            sampled_point = utils.bezier_point(curve_control_points, t)
                            # Convert back from pixels to grid coordinates
                            grid_point = self._pixels_to_grid_coords(sampled_point, cells_to_pixels_map)
                            if grid_point:
                                all_sampled_points.append(grid_point)
                        elif len(curve_control_points) == 2:
                            sampled_point = curve_control_points[0] + (curve_control_points[1] - curve_control_points[0]) * t
                            grid_point = self._pixels_to_grid_coords(sampled_point, cells_to_pixels_map)
                            if grid_point:
                                all_sampled_points.append(grid_point)
                        else:
                            raise ValueError(f"Invalid number of control points: {len(curve_control_points)}")
            
            # Step 5: Find the best sampled point in the given direction
            best_point = self._find_best_point_in_direction(all_sampled_points, center, direction_vector)
            return best_point
            
        except Exception as e:
            print(f"Error in curve-based intersection finding: {e}")
            raise e
            # Fallback to original method
            return self._find_intersection_point_fallback(base_sample, center, direction)
    
    def _create_sketch_xml(self, base_sample: Dict[str, Any]) -> str:
        """Convert base_sample to XML format expected by utils."""
        
        # Check if this is a multi-stroke shape
        if 'strokes' in base_sample and base_sample.get('stroke_count', 0) > 1:
            # Multi-stroke format
            strokes_xml = []
            for i, stroke_data in enumerate(base_sample['strokes'], 1):
                points_str = "'" + "', '".join([f"x{x}y{y}" for x, y in stroke_data['points']]) + "'"
                t_values_str = ",".join([f"{t:.2f}" for t in stroke_data['t_values']])
                
                stroke_xml = f"""    <s{i}>
        <points>{points_str}</points>
        <t_values>{t_values_str}</t_values>
        <id>{base_sample.get('shape_id', 'shape')}</id>
    </s{i}>"""
                strokes_xml.append(stroke_xml)
            
            xml_content = f"""<strokes>
{chr(10).join(strokes_xml)}
</strokes>"""
        else:
            # Single-stroke format
            if 'raw_points' in base_sample:
                points_str = "'" + "', '".join([f"x{x}y{y}" for x, y in base_sample['raw_points']]) + "'"
            else:
                points_str = "'" + "', '".join(base_sample.get('points', [])) + "'"
            t_values_str = ",".join([f"{t:.2f}" for t in base_sample.get('t_values', [])])
            shape_id = base_sample.get('shape_id', 'shape')
            
            xml_content = f"""<strokes>
    <s1>
        <points>{points_str}</points>
        <t_values>{t_values_str}</t_values>
        <id>{shape_id}</id>
    </s1>
</strokes>"""
        
        return xml_content
    
    def _pixels_to_grid_coords(self, pixel_point: Tuple[float, float], cells_to_pixels_map: Dict[str, Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """Convert pixel coordinates back to grid coordinates."""
        px, py = pixel_point
        
        # Find the closest grid cell
        min_distance = float('inf')
        closest_coords = None
        
        for grid_key, (grid_px, grid_py) in cells_to_pixels_map.items():
            distance = math.sqrt((px - grid_px)**2 + (py - grid_py)**2)
            if distance < min_distance:
                min_distance = distance
                # Extract x,y from grid_key format "x1y2"
                import re
                match = re.match(r'x(\d+)y(\d+)', grid_key)
                if match:
                    x, y = int(match.group(1)), int(match.group(2))
                    closest_coords = (x, y)
        
        return closest_coords
    
    def _find_best_point_in_direction(self, sampled_points: List[Tuple[int, int]], center: Tuple[int, int], direction_vector: Tuple[float, float]) -> Tuple[int, int]:
        """Find the best sampled point in the given direction from center."""
        
        if not sampled_points:
            return center
        
        best_point = center
        best_score = -1
        
        for point in sampled_points:
            # Calculate vector from center to this point
            to_point = (point[0] - center[0], point[1] - center[1])
            
            # Skip if point is at center
            if to_point == (0, 0):
                continue
            
            # Calculate distance from center
            distance = math.sqrt(to_point[0]**2 + to_point[1]**2)
            
            # Normalize the vector
            normalized_to_point = (to_point[0] / distance, to_point[1] / distance)
            
            # Calculate dot product to measure alignment with desired direction
            dot_product = normalized_to_point[0] * direction_vector[0] + normalized_to_point[1] * direction_vector[1]
            
            # Score combines direction alignment and distance (prefer points that are aligned and not too close)
            score = dot_product * min(distance / 5.0, 1.0)  # Normalize distance factor
            
            if score > best_score and distance > 2:  # Avoid points too close to center
                best_score = score
                best_point = point
        
        return best_point
    
    def _find_intersection_point_fallback(self, base_sample: Dict[str, Any], center: Tuple[int, int], direction: str) -> Tuple[int, int]:
        """Fallback intersection finding using the original straight-line method."""
        
        direction_vector = self.DIRECTIONS[direction]
        
        # Get all shape points
        if base_sample.get('strokes'):
            all_points = []
            for stroke in base_sample['strokes']:
                all_points.extend(stroke['points'])
        else:
            all_points = base_sample['raw_points']
        
        if not all_points:
            return center
        
        # Cast ray from center in direction and find closest intersection
        best_distance = float('inf')
        best_point = center
        
        # Check intersection with each edge of the shape
        for i in range(len(all_points) - 1):
            p1 = all_points[i]
            p2 = all_points[i + 1]
            
            # Skip if points are the same (duplicated corners)
            if p1 == p2:
                continue
            
            intersection = self._ray_line_intersection(center, direction_vector, p1, p2)
            if intersection:
                distance = math.sqrt((intersection[0] - center[0])**2 + (intersection[1] - center[1])**2)
                if distance < best_distance and distance > 1:  # Avoid intersection at center
                    best_distance = distance
                    best_point = intersection
        
        return best_point
    
    def format_strokes_for_prompt(self) -> List[Dict[str, Any]]:
        """Format strokes for prompt generation, similar to triangular shapes."""
        return [
            {
                'points': stroke.points, 
                't_values': stroke.t_values, 
                'formatted_points': stroke.formatted_points, 
                'stroke_id': stroke.stroke_id
            } 
            for stroke in self.strokes
        ] 

    def _generate_straight_stroke(self, intersection_point: Tuple[int, int], direction: str) -> Dict[str, Any]:
        """Generate a simple straight line stroke as fallback."""
        direction_vector = self.DIRECTIONS[direction]
        stroke_length = random.randint(self.MIN_STROKE_LENGTH, self.MAX_STROKE_LENGTH)
        
        end_x = intersection_point[0] + int(direction_vector[0] * stroke_length)
        end_y = intersection_point[1] + int(direction_vector[1] * stroke_length)
        
        # Ensure end point is within grid bounds
        end_x = max(self.edge_margin, min(self.grid_size - self.edge_margin, end_x))
        end_y = max(self.edge_margin, min(self.grid_size - self.edge_margin, end_y))
        
        return {
            'points': [intersection_point, (end_x, end_y)],
            't_values': [0.0, 1.0],
            'length': stroke_length,
            'stroke_type': 'line'
        } 
    
    def _ray_line_intersection(self, ray_origin: Tuple[int, int], ray_direction: Tuple[float, float], 
                              line_p1: Tuple[int, int], line_p2: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Calculate intersection between a ray and a line segment."""
        
        # Ray: ray_origin + t * ray_direction (t >= 0)
        # Line: line_p1 + s * (line_p2 - line_p1) (0 <= s <= 1)
        
        rx, ry = ray_origin
        dx, dy = ray_direction
        x1, y1 = line_p1
        x2, y2 = line_p2
        
        # Line direction vector
        lx, ly = x2 - x1, y2 - y1
        
        # Solve the system: ray_origin + t * ray_direction = line_p1 + s * line_direction
        # rx + t * dx = x1 + s * lx
        # ry + t * dy = y1 + s * ly
        
        denominator = dx * ly - dy * lx
        if abs(denominator) < 1e-10:  # Lines are parallel
            return None
        
        # Calculate parameters
        t = ((x1 - rx) * ly - (y1 - ry) * lx) / denominator
        s = ((x1 - rx) * dy - (y1 - ry) * dx) / denominator
        
        # Check if intersection is valid
        if t >= 0 and 0 <= s <= 1:
            # Calculate intersection point
            ix = int(rx + t * dx)
            iy = int(ry + t * dy)
            return (ix, iy)
        
        return None 

    def _generate_simple_triangle_stroke(self, intersection_point: Tuple[int, int], direction: str, x_range: Tuple[int, int], y_range: Tuple[int, int]) -> Dict[str, Any]:
        """Generate a simple triangular path as a single stroke with directional constraints."""
        
        start_x, start_y = intersection_point
        min_x, max_x = x_range
        min_y, max_y = y_range
        
        # Calculate minimum area needed for a meaningful triangle
        min_area = 20  # Minimum triangle area in grid units
        
        max_attempts = 10
        for attempt in range(max_attempts):
            # Generate a triangle with better size control
            triangle_size = random.randint(15, 30)  # Increased minimum size
            
            # Constrain the triangle vertices within the directional range
            vertex1 = (start_x, start_y)  # Start at intersection
            
            # Second vertex - extend meaningfully in one direction within constraints
            vertex2_x = max(min_x, min(max_x, start_x + random.randint(-triangle_size, triangle_size)))
            vertex2_y = max(min_y, min(max_y, start_y + random.randint(-triangle_size//2, triangle_size)))
            vertex2 = (vertex2_x, vertex2_y)
            
            # Third vertex - ensure it forms a meaningful triangle, not a line
            # Make sure the third vertex is not collinear with the first two
            min_height = triangle_size // 3  # Minimum perpendicular distance
            vertex3_x = max(min_x, min(max_x, start_x + random.randint(-triangle_size, triangle_size)))
            vertex3_y = max(min_y, min(max_y, start_y + random.randint(-triangle_size, triangle_size)))
            
            # Ensure vertex3 is not too close to the line formed by vertex1 and vertex2
            # Calculate perpendicular distance from vertex3 to line vertex1-vertex2
            if vertex2_x != vertex1[0] or vertex2_y != vertex1[1]:  # Avoid identical points
                # Use the shoelace formula to calculate triangle area
                area = 0.5 * abs(vertex1[0] * (vertex2_y - vertex3_y) + 
                               vertex2_x * (vertex3_y - vertex1[1]) + 
                               vertex3_x * (vertex1[1] - vertex2_y))
                
                if area >= min_area:
                    vertex3 = (vertex3_x, vertex3_y)
                    break
        
        # Fallback: create a guaranteed non-degenerate triangle
        if attempt >= max_attempts - 1:
            # Create a triangle with fixed proportions that guarantee minimum area
            base_size = max(15, int(math.sqrt(min_area * 2)))
            height = max(10, int(min_area * 2 / base_size))
            
            vertex1 = (start_x, start_y)
            vertex2 = (max(min_x, min(max_x, start_x + base_size)), start_y)
            vertex3 = (max(min_x, min(max_x, start_x + base_size//2)), 
                      max(min_y, min(max_y, start_y + height)))
        
        # Create triangular path: start -> vertex2 -> vertex3 -> back to start
        points = [vertex1, vertex2, vertex3, vertex1]
        t_values = [0.0, 0.33, 0.67, 1.0]
        
        return {
            'points': points,
            't_values': t_values,
            'length': triangle_size * 3,  # Approximate perimeter
            'stroke_type': 'triangle'
        } 

    def _generate_inscribed_stroke(self, base_sample: Dict[str, Any], stroke_type: str) -> Tuple[Tuple[int, int], Dict[str, Any]]:
        """Generate a stroke inscribed inside the base shape using uniform scaling.
        
        Args:
            base_sample: The base shape data
            stroke_type: Type of stroke to generate inside
            
        Returns:
            Tuple of (center_point, stroke_dict)
        """
        
        # Step 1: Find a point inside the base shape to serve as the inscribed shape's center
        inside_center = self._find_point_inside_base_shape(base_sample)
        
        # Step 2: Generate the stroke normally without constraints to preserve its properties
        inscribed_stroke = self._generate_unconstrained_stroke(stroke_type)
        
        # Step 3: Handle multistroke shapes differently
        if inscribed_stroke.get('is_multistroke', False):
            # For multistroke shapes, process all strokes
            final_stroke = self._process_multistroke_inscribed_shape(inscribed_stroke, inside_center, base_sample)
        else:
            # Step 3: Calculate the original center of the generated stroke
            original_center = self._calculate_stroke_center(inscribed_stroke['points'])
            
            # Step 4: Translate the stroke to be centered at the inside point
            translated_stroke = self._translate_stroke(inscribed_stroke, original_center, inside_center)
            
            # Step 5: Scale the stroke down until it fits completely inside the base shape
            final_stroke = self._scale_stroke_to_fit_inside(translated_stroke, inside_center, base_sample)
        
        return inside_center, final_stroke
    
    def _find_point_inside_base_shape(self, base_sample: Dict[str, Any]) -> Tuple[int, int]:
        """Find a point inside the base shape using geometric center.
        
        Args:
            base_sample: The base shape data
            
        Returns:
            A point inside the base shape (geometric center)
        """
        
        # For most reasonable shapes, the geometric center is inside the shape
        # This is much simpler than complex ray casting and works well in practice
        return self._calculate_shape_center(base_sample)

    def _calculate_stroke_center(self, points: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Calculate the geometric center of a stroke.
        
        Args:
            points: List of points defining the stroke
            
        Returns:
            The center point as (x, y)
        """
        
        if not points:
            return (50, 50)  # Default center
        
        center_x = sum(point[0] for point in points) // len(points)
        center_y = sum(point[1] for point in points) // len(points)
        
        return (center_x, center_y)
    
    def _translate_stroke(self, stroke: Dict[str, Any], from_center: Tuple[int, int], to_center: Tuple[int, int]) -> Dict[str, Any]:
        """Translate a stroke from one center position to another.
        
        Args:
            stroke: The stroke data with points and t_values
            from_center: Original center position
            to_center: Target center position
            
        Returns:
            New stroke data with translated points
        """
        
        dx = to_center[0] - from_center[0]
        dy = to_center[1] - from_center[1]
        
        # Translate all points
        translated_points = [(x + dx, y + dy) for x, y in stroke['points']]
        
        # Create new stroke with translated points
        translated_stroke = stroke.copy()
        translated_stroke['points'] = translated_points
        
        return translated_stroke 

    def _scale_stroke_to_fit_inside(self, stroke: Dict[str, Any], center: Tuple[int, int], base_sample: Dict[str, Any]) -> Dict[str, Any]:
        """Scale down a stroke until it fits completely inside the base shape using ray-casting approach.
        
        Algorithm:
        1. For each sampled point P in the inscribed stroke (not just control points)
        2. Cast ray from center C to point P  
        3. Find intersection I with base shape boundary
        4. If distance CP > CI, scale the entire stroke by CI/CP ratio (with margin)
        
        Args:
            stroke: The stroke to scale
            center: The center point (guaranteed inside base shape)
            base_sample: The base shape data
            
        Returns:
            Scaled stroke that fits inside the base shape
        """
        
        # Get sampled points from base shape for intersection testing
        base_points = self._get_base_shape_boundary_points(base_sample)
        
        if not base_points:
            return stroke  # No base shape points available
        
        # Get sampled points from inscribed shape for accurate checking
        inscribed_points = self._get_inscribed_shape_sampled_points(stroke)
        
        # Find the minimum scaling factor needed
        min_scale_factor = 1.0
        
        for point in inscribed_points:
            # Skip if point is at center (no scaling needed)
            if point == center:
                continue
                
            # Calculate ray direction from center to point
            ray_direction = (point[0] - center[0], point[1] - center[1])
            
            # Find intersection with base shape boundary
            closest_intersection = self._find_ray_boundary_intersection(center, ray_direction, base_points)
            
            if closest_intersection:
                # Calculate distances
                cp_distance = ((point[0] - center[0])**2 + (point[1] - center[1])**2)**0.5
                ci_distance = ((closest_intersection[0] - center[0])**2 + (closest_intersection[1] - center[1])**2)**0.5
                
                # If point is outside boundary, calculate required scale factor
                if cp_distance > ci_distance and ci_distance > 0:
                    required_scale = ci_distance / cp_distance
                    min_scale_factor = min(min_scale_factor, required_scale)
        
        # Apply safety margin
        final_scale_factor = min_scale_factor * 0.5
        
        # Scale the stroke if needed
        if final_scale_factor < 1.0:
            return self._apply_uniform_scale(stroke, center, final_scale_factor)
        else:
            return stroke

    def _get_base_shape_boundary_points(self, base_sample: Dict[str, Any]) -> List[Tuple[int, int]]:
        """Extract boundary points from the base shape for intersection testing.
        
        Args:
            base_sample: The base shape data
            
        Returns:
            List of points defining the base shape boundary
        """
        
        try:
            # Use the same approach as _find_point_inside_base_shape
            xml_content = self._create_sketch_xml(base_sample)
            strokes_list_str, t_values_str = utils.parse_xml_string(xml_content, self.grid_size)
            strokes_list, t_values = ast.literal_eval(strokes_list_str), ast.literal_eval(t_values_str)
            
            cell_size = 6
            cells_to_pixels_map = utils.cells_to_pixels(self.grid_size, cell_size, header_size=cell_size)
            all_control_points = utils.get_control_points(strokes_list, t_values, cells_to_pixels_map)
            
            # Sample points along each curve
            boundary_points = []
            
            for stroke_control_points in all_control_points:
                for curve_control_points in stroke_control_points:
                    # Sample more densely for better boundary representation
                    for i in range(25):
                        t = i / 24.0
                        if len(curve_control_points) >= 4:
                            sampled_point = utils.bezier_point(curve_control_points, t)
                            grid_point = self._pixels_to_grid_coords(sampled_point, cells_to_pixels_map)
                            if grid_point:
                                boundary_points.append(grid_point)
                        elif len(curve_control_points) == 2:
                            sampled_point = curve_control_points[0] + (curve_control_points[1] - curve_control_points[0]) * t
                            grid_point = self._pixels_to_grid_coords(sampled_point, cells_to_pixels_map)
                            if grid_point:
                                boundary_points.append(grid_point)
                        else:
                            raise ValueError(f"Invalid number of control points: {len(curve_control_points)}")
            
            return boundary_points
            
        except Exception as e:
            print(f"Error getting base shape boundary points: {e}")
            # Fallback: use raw_points if available
            if 'raw_points' in base_sample:
                return base_sample['raw_points']
            return []
    
    def _find_ray_boundary_intersection(self, center: Tuple[int, int], ray_direction: Tuple[float, float], boundary_points: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """Find the intersection between a ray from center to a point and the base shape boundary."""
        
        for i in range(len(boundary_points)):
            edge_p1 = boundary_points[i]
            edge_p2 = boundary_points[(i + 1) % len(boundary_points)]
            
            intersection = self._ray_line_intersection(center, ray_direction, edge_p1, edge_p2)
            if intersection:
                return intersection
        
        return None
    
    def _apply_uniform_scale(self, stroke: Dict[str, Any], center: Tuple[int, int], scale_factor: float) -> Dict[str, Any]:
        """Apply uniform scaling to a stroke around a center point.
        
        Args:
            stroke: The stroke to scale
            center: The center point for scaling
            scale_factor: The scaling factor
            
        Returns:
            New stroke with scaled points
        """
        
        scaled_points = []
        
        for point in stroke['points']:
            # Vector from center to point
            dx = point[0] - center[0]
            dy = point[1] - center[1]
            
            # Scale the vector
            scaled_dx = dx * scale_factor
            scaled_dy = dy * scale_factor
            
            # New point position
            new_point = (
                int(center[0] + scaled_dx),
                int(center[1] + scaled_dy)
            )
            scaled_points.append(new_point)
        
        # Create new stroke with scaled points
        scaled_stroke = stroke.copy()
        scaled_stroke['points'] = scaled_points
        
        return scaled_stroke 

    def _process_multistroke_inscribed_shape(self, inscribed_stroke: Dict[str, Any], inside_center: Tuple[int, int], base_sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process a multistroke inscribed shape by scaling all strokes together as a unit.
        
        Args:
            inscribed_stroke: The multistroke shape data
            inside_center: The center point inside the base shape
            base_sample: The base shape data
            
        Returns:
            Processed multistroke shape with all strokes scaled together
        """
        
        # Step 1: Get all strokes from the inscribed shape
        strokes = inscribed_stroke['strokes']
        
        # Step 2: Calculate the overall center of the multistroke shape
        all_points = []
        for stroke_data in strokes:
            all_points.extend(stroke_data['points'])
        
        if not all_points:
            return inscribed_stroke  # Fallback if no points
        
        original_center = self._calculate_stroke_center(all_points)
        
        # Step 3: Translate all strokes to center the shape at inside_center
        translated_strokes = []
        dx = inside_center[0] - original_center[0] 
        dy = inside_center[1] - original_center[1]
        
        for stroke_data in strokes:
            translated_points = [(x + dx, y + dy) for x, y in stroke_data['points']]
            translated_stroke = stroke_data.copy()
            translated_stroke['points'] = translated_points
            translated_strokes.append(translated_stroke)
        
        # Step 4: Scale all strokes together as a unit
        base_points = self._get_base_shape_boundary_points(base_sample)
        if not base_points:
            inscribed_stroke['strokes'] = translated_strokes
            return inscribed_stroke
        
        # Find the minimum scaling factor needed across all strokes
        min_scale_factor = 1.0
        
        # Get all sampled points from all inscribed strokes
        all_inscribed_points = []
        for stroke_data in translated_strokes:
            stroke_sampled_points = self._get_inscribed_shape_sampled_points(stroke_data)
            all_inscribed_points.extend(stroke_sampled_points)
        
        # Check all sampled points from all strokes
        for point in all_inscribed_points:
            # Skip if point is at center (no scaling needed)
            if point == inside_center:
                continue
                
            # Calculate ray direction from center to point
            ray_direction = (point[0] - inside_center[0], point[1] - inside_center[1])
            
            # Find intersection with base shape boundary
            closest_intersection = self._find_ray_boundary_intersection(inside_center, ray_direction, base_points)
            
            if closest_intersection:
                # Calculate distances
                cp_distance = ((point[0] - inside_center[0])**2 + (point[1] - inside_center[1])**2)**0.5
                ci_distance = ((closest_intersection[0] - inside_center[0])**2 + (closest_intersection[1] - inside_center[1])**2)**0.5
                
                # If point is outside boundary, calculate required scale factor
                if cp_distance > ci_distance and ci_distance > 0:
                    required_scale = ci_distance / cp_distance
                    min_scale_factor = min(min_scale_factor, required_scale)
        
        # Apply safety margin and scale all strokes if needed
        final_scale_factor = min_scale_factor * 0.5
        
        if final_scale_factor < 1.0:
            # Scale all strokes around the inside_center
            final_strokes = []
            for stroke_data in translated_strokes:
                scaled_stroke = self._apply_uniform_scale(stroke_data, inside_center, final_scale_factor)
                final_strokes.append(scaled_stroke)
            inscribed_stroke['strokes'] = final_strokes
        else:
            inscribed_stroke['strokes'] = translated_strokes
        
        return inscribed_stroke 

    def _generate_unconstrained_stroke(self, stroke_type: str) -> Dict[str, Any]:
        """Generate a stroke without positional or directional constraints.
        
        Args:
            stroke_type: Type of stroke to generate
            
        Returns:
            Dictionary containing stroke data - for multistroke shapes, includes all strokes
        """
        
        try:
            if stroke_type == 'line':
                generator = StraightLine(self.grid_size)
                points, t_values = generator.generate_base_shape()
                return {
                    'points': points,
                    't_values': t_values,
                    'length': ((points[1][0] - points[0][0])**2 + (points[1][1] - points[0][1])**2)**0.5,
                    'stroke_type': 'line',
                    'is_multistroke': False
                }
                
            elif stroke_type == 'curve':
                generator = Curve(self.grid_size)
                sample = generator.generate_sample(apply_random_transforms=False)
                return {
                    'points': sample['raw_points'],
                    't_values': sample['t_values'],
                    'length': 20,
                    'stroke_type': 'curve',
                    'is_multistroke': False
                }
                
            elif stroke_type == 'rectangle':
                generator = RectangularShape(self.grid_size)
                sample = generator.generate_sample(apply_random_transforms=False)
                return {
                    'points': sample['raw_points'],
                    't_values': sample['t_values'], 
                    'length': 30,
                    'stroke_type': 'rectangle',
                    'is_multistroke': False
                }
                
            elif stroke_type == 'circle':
                generator = CircularShape(self.grid_size)
                # Force circle generation 
                points, t_values, shape_info = generator._generate_circle()
                return {
                    'points': points,
                    't_values': t_values,
                    'length': 25,
                    'stroke_type': 'circle',
                    'is_multistroke': False
                }
                
            elif stroke_type == 'triangle':
                generator = TriangularShape(self.grid_size)
                sample = generator.generate_sample(apply_random_transforms=False)
                
                # Handle multistroke triangle properly
                if 'strokes' in sample and len(sample['strokes']) > 1:
                    return {
                        'strokes': sample['strokes'],  # All strokes that make up the triangle
                        'length': 35,
                        'stroke_type': 'triangle',
                        'is_multistroke': True
                    }
                else:
                    # Fallback for single-stroke triangle
                    return {
                        'points': sample['raw_points'],
                        't_values': sample['t_values'],
                        'length': 35,
                        'stroke_type': 'triangle',
                        'is_multistroke': False
                    }
                
            elif stroke_type == 'arc':
                generator = Curve(self.grid_size)
                points, t_values = generator._generate_arc()
                return {
                    'points': points,
                    't_values': t_values,
                    'length': 15,
                    'stroke_type': 'arc',
                    'is_multistroke': False
                }
                
            elif stroke_type == 'ellipse':
                generator = CircularShape(self.grid_size)
                # Force ellipse generation
                points, t_values, shape_info = generator._generate_ellipse()
                return {
                    'points': points,
                    't_values': t_values,
                    'length': 25,
                    'stroke_type': 'ellipse',
                    'is_multistroke': False
                }
                
            elif stroke_type == 'rhombus':
                generator = RectangularShape(self.grid_size)
                points, t_values, shape_info = generator._generate_rhombus()
                return {
                    'points': points,
                    't_values': t_values,
                    'length': 35,
                    'stroke_type': 'rhombus',
                    'is_multistroke': False
                }
                
            elif stroke_type == 'square':
                generator = RectangularShape(self.grid_size)
                points, t_values, shape_info = generator._generate_square()
                return {
                    'points': points,
                    't_values': t_values,
                    'length': 30,
                    'stroke_type': 'square',
                    'is_multistroke': False
                }
                
            else:
                # Fallback to line if unknown type
                generator = StraightLine(self.grid_size)
                points, t_values = generator.generate_base_shape()
                return {
                    'points': points,
                    't_values': t_values,
                    'length': 20,
                    'stroke_type': 'line',
                    'is_multistroke': False
                }
                
        except Exception as e:
            print(f"Error generating unconstrained stroke of type {stroke_type}: {e}")
            # Fallback to simple line
            generator = StraightLine(self.grid_size)
            points, t_values = generator.generate_base_shape()
            return {
                'points': points,
                't_values': t_values,
                'length': 15,
                'stroke_type': 'line',
                'is_multistroke': False
            } 

    def _get_inscribed_shape_sampled_points(self, stroke: Dict[str, Any]) -> List[Tuple[int, int]]:
        """Extract densely sampled points from the inscribed shape using Bezier sampling.
        
        Args:
            stroke: The inscribed stroke data
            
        Returns:
            List of sampled points along the inscribed shape curves
        """
        
        try:
            # Create a temporary sample in the same format as base_sample
            temp_sample = {
                'raw_points': stroke['points'],
                't_values': stroke['t_values'],
                'shape_id': 'inscribed_stroke'
            }
            
            # Convert to XML format for utils processing
            xml_content = self._create_sketch_xml(temp_sample)
            
            # Parse with utils to get strokes and t_values
            strokes_list_str, t_values_str = utils.parse_xml_string(xml_content, self.grid_size)
            strokes_list, t_values = ast.literal_eval(strokes_list_str), ast.literal_eval(t_values_str)
            
            # Get control points using utils
            cell_size = 6
            cells_to_pixels_map = utils.cells_to_pixels(self.grid_size, cell_size, header_size=cell_size)
            all_control_points = utils.get_control_points(strokes_list, t_values, cells_to_pixels_map)
            
            # Sample points along each curve (same density as base shape)
            sampled_points = []
            
            for stroke_control_points in all_control_points:
                for curve_control_points in stroke_control_points:
                    # Sample same density as base shape (50 points)
                    for i in range(50):
                        t = i / 49.0
                        if len(curve_control_points) >= 4:
                            sampled_point = utils.bezier_point(curve_control_points, t)
                            grid_point = self._pixels_to_grid_coords(sampled_point, cells_to_pixels_map)
                            if grid_point:
                                sampled_points.append(grid_point)
                        elif len(curve_control_points) == 2:
                            sampled_point = curve_control_points[0] + (curve_control_points[1] - curve_control_points[0]) * t
                            grid_point = self._pixels_to_grid_coords(sampled_point, cells_to_pixels_map)
                            if grid_point:
                                sampled_points.append(grid_point)
                        else:
                            raise ValueError(f"Invalid number of control points: {len(curve_control_points)}")
            
            return sampled_points if sampled_points else stroke['points']  # Fallback to control points
            
        except Exception as e:
            print(f"Error getting inscribed shape sampled points: {e}")
            # Fallback: use original control points
            return stroke['points'] 