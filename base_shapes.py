import numpy as np
from typing import List, Tuple, Dict, Any
import random
import math

class BaseShape:
    """Base class for all geometric shapes and primitives."""
    
    # Grid and transformation constants
    DEFAULT_GRID_SIZE = 100
    MIN_COORDINATE = 1
    
    # Transformation default ranges
    DEFAULT_SCALE_RANGE = (1.0, 1.0)
    DEFAULT_SHIFT_RANGE = (0, 0)
    DEFAULT_ROTATION_RANGE = (0, 360)
    
    def __init__(self, grid_size: int = DEFAULT_GRID_SIZE):
        self.grid_size = grid_size
        self.points = []
        self.t_values = []
        self.shape_id = ""
        
        # Calculate dynamic margins based on grid size
        self.edge_margin = max(5, int(self.grid_size * 0.05))  # 5% of grid size, minimum 5
        self.transform_buffer = max(10, int(self.grid_size * 0.1))  # 10% of grid size for transformations
        
    def generate_base_shape(self) -> Tuple[List[Tuple[int, int]], List[float]]:
        """Generate the base shape coordinates and t_values.
        Should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement generate_base_shape")
    
    def apply_transformations(self, 
                            points: List[Tuple[int, int]], 
                            scale: float = 1.0,
                            shift_x: int = 0, 
                            shift_y: int = 0,
                            rotation_deg: float = 0.0) -> List[Tuple[int, int]]:
        """Apply scale, shift, and rotation transformations to points."""
        transformed_points = []
        
        # Convert to numpy for easier math
        points_array = np.array(points, dtype=float)
        
        # Apply scaling
        points_array *= scale
        
        # Apply rotation if specified
        if rotation_deg != 0:
            angle_rad = math.radians(rotation_deg)
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
            
            # Rotate around center of grid
            center_x, center_y = points_array[:, 0].mean(), points_array[:, 1].mean() #self.grid_size / 2, self.grid_size / 2
            
            # Translate to origin, rotate, translate back
            points_array[:, 0] -= center_x
            points_array[:, 1] -= center_y
            
            rotated_x = points_array[:, 0] * cos_a - points_array[:, 1] * sin_a
            rotated_y = points_array[:, 0] * sin_a + points_array[:, 1] * cos_a
            
            points_array[:, 0] = rotated_x + center_x
            points_array[:, 1] = rotated_y + center_y
        
        # Apply shift
        points_array[:, 0] += shift_x
        points_array[:, 1] += shift_y
        
        # Clamp to grid bounds with enhanced margins and convert to integers
        # Use a margin to ensure coordinates stay well within canvas bounds
        display_margin = max(3, int(self.grid_size * 0.03))  # 3% margin for display
        min_coord = self.MIN_COORDINATE + display_margin
        max_coord = self.grid_size - display_margin
        
        for point in points_array:
            x = max(min_coord, min(max_coord, int(round(point[0]))))
            y = max(min_coord, min(max_coord, int(round(point[1]))))
            transformed_points.append((x, y))
            
        return transformed_points
    
    def format_points_for_prompt(self, points: List[Tuple[int, int]]) -> List[str]:
        """Convert (x, y) tuples to 'xNyM' format expected by the prompt."""
        return [f'x{x}y{y}' for x, y in points]
    
    def apply_random_transformations(self, points: List[Tuple[int, int]], t_values: List[float]) -> Tuple[List[Tuple[int, int]], List[float], Dict[str, Any]]:
        """Apply random transformations to points and return transformed points, t_values, and transformation info."""
        
        # Generate random transformation parameters
        scale = random.uniform(*self.DEFAULT_SCALE_RANGE)
        shift_x = random.randint(*self.DEFAULT_SHIFT_RANGE)
        shift_y = random.randint(*self.DEFAULT_SHIFT_RANGE)
        rotation = random.uniform(*self.DEFAULT_ROTATION_RANGE)
        
        # Apply transformations
        transformed_points = self.apply_transformations(points, scale, shift_x, shift_y, rotation)
        
        # t_values typically don't change with geometric transformations
        transformed_t_values = t_values.copy()
        
        transformations = {
            'scale': scale,
            'shift_x': shift_x,
            'shift_y': shift_y,
            'rotation_deg': rotation
        }
        
        return transformed_points, transformed_t_values, transformations
    
    def generate_sample(self, 
                       scale_range: Tuple[float, float] = None,
                       shift_range: Tuple[int, int] = None,
                       rotation_range: Tuple[float, float] = None,
                       apply_random_transforms: bool = True) -> Dict[str, Any]:
        """Generate a sample with random transformations."""
        
        # Use defaults if not specified
        if scale_range is None:
            scale_range = self.DEFAULT_SCALE_RANGE
        if shift_range is None:
            shift_range = self.DEFAULT_SHIFT_RANGE
        if rotation_range is None:
            rotation_range = self.DEFAULT_ROTATION_RANGE
        
        # Generate base shape
        base_points, t_values = self.generate_base_shape()
        
        if apply_random_transforms:
            # Random transformations
            scale = random.uniform(*scale_range)
            shift_x = random.randint(*shift_range)
            shift_y = random.randint(*shift_range)
            rotation = random.uniform(*rotation_range)
        else:
            scale, shift_x, shift_y, rotation = 1.0, 0, 0, 0.0
        
        # Apply transformations
        transformed_points = self.apply_transformations(
            base_points, scale, shift_x, shift_y, rotation
        )
        
        # Format for prompt
        formatted_points = self.format_points_for_prompt(transformed_points)
        
        return {
            'points': formatted_points,
            't_values': t_values,
            'shape_id': self.shape_id,
            'transformations': {
                'scale': scale,
                'shift_x': shift_x,
                'shift_y': shift_y,
                'rotation_deg': rotation
            },
            'raw_points': transformed_points
        } 

class Stroke:
    """Represents a single stroke with its own points and t_values."""
    
    def __init__(self, points: List[Tuple[int, int]], t_values: List[float], stroke_id: str = None):
        if len(points) != len(t_values):
            raise ValueError("Points and t_values must have the same length")
        
        self.points = points
        self.t_values = t_values
        self.stroke_id = stroke_id
        self.formatted_points = [f'x{x}y{y}' for x, y in points]
    
    def apply_transformations(self, 
                            scale: float = 1.0,
                            shift_x: int = 0, 
                            shift_y: int = 0,
                            rotation_deg: float = 0.0,
                            grid_size: int = 100,
                            edge_margin: int = 5) -> 'Stroke':
        """Apply transformations to this stroke and return a new transformed stroke."""
        
        transformed_points = []
        
        for x, y in self.points:
            # Apply scale
            new_x = x * scale
            new_y = y * scale
            
            # Apply rotation
            if rotation_deg != 0:
                angle_rad = math.radians(rotation_deg)
                cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
                
                # Rotate around center of grid
                center_x, center_y = grid_size / 2, grid_size / 2
                new_x -= center_x
                new_y -= center_y
                
                rotated_x = new_x * cos_a - new_y * sin_a
                rotated_y = new_x * sin_a + new_y * cos_a
                
                new_x = rotated_x + center_x
                new_y = rotated_y + center_y
            
            # Apply shift
            new_x += shift_x
            new_y += shift_y
            
            # Ensure points stay within grid bounds
            new_x = max(edge_margin, min(grid_size - edge_margin, int(new_x)))
            new_y = max(edge_margin, min(grid_size - edge_margin, int(new_y)))
            
            transformed_points.append((new_x, new_y))
        
        return Stroke(transformed_points, self.t_values.copy(), self.stroke_id)

class MultiStrokeShape(BaseShape):
    """Base class for shapes that require multiple strokes to draw naturally.
    
    Each stroke is handled as an individual Stroke object, allowing for proper
    separation of concerns and natural multi-stroke rendering.
    """
    
    def __init__(self, grid_size: int = BaseShape.DEFAULT_GRID_SIZE):
        super().__init__(grid_size)
        self.strokes = []  # List of Stroke objects
    
    def add_stroke(self, points: List[Tuple[int, int]], t_values: List[float], stroke_id: str = None):
        """Add a stroke to the multi-stroke shape."""
        stroke = Stroke(points, t_values, stroke_id or f's{len(self.strokes) + 1}')
        self.strokes.append(stroke)
    
    def clear_strokes(self):
        """Clear all strokes."""
        self.strokes = []
    
    def get_stroke_count(self) -> int:
        """Get the number of strokes in this shape."""
        return len(self.strokes)
    
    def apply_transformations_to_all_strokes(self, transformations: Dict[str, Any]):
        """Apply transformations to all strokes independently."""
        transformed_strokes = []
        
        for stroke in self.strokes:
            transformed_stroke = stroke.apply_transformations(
                scale=transformations['scale'],
                shift_x=transformations['shift_x'],
                shift_y=transformations['shift_y'],
                rotation_deg=transformations['rotation_deg'],
                grid_size=self.grid_size,
                edge_margin=self.edge_margin
            )
            transformed_strokes.append(transformed_stroke)
        
        self.strokes = transformed_strokes
    
    def _generate_random_transformations(self) -> Dict[str, Any]:
        """Generate random transformation parameters."""
        return {
            'scale': random.uniform(*self.DEFAULT_SCALE_RANGE),
            'shift_x': random.randint(*self.DEFAULT_SHIFT_RANGE),
            'shift_y': random.randint(*self.DEFAULT_SHIFT_RANGE),
            'rotation_deg': random.uniform(*self.DEFAULT_ROTATION_RANGE)
        }
    
    def generate_sample(self, apply_random_transforms: bool = True) -> Dict[str, Any]:
        """Generate a complete multi-stroke shape sample. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement generate_sample method")
    
    def format_multi_stroke_output(self, concept: str = None) -> str:
        """Format the output for multi-stroke shapes with multiple <s_i> tags."""
        if concept is None:
            concept = self.shape_id.replace('_', ' ').title()
        
        if not self.strokes:
            raise ValueError("No strokes available to format")
        
        # Build the strokes XML
        strokes_xml = []
        for stroke in self.strokes:
            points_str = "'" + "', '".join(stroke.formatted_points) + "'"
            t_values_str = ",".join([f"{t:.2f}" for t in stroke.t_values])
            stroke_id = stroke.stroke_id or f's{len(strokes_xml) + 1}'
            
            stroke_xml = f"""    <{stroke_id}>
        <points>{points_str}</points>
        <t_values>{t_values_str}</t_values>
        <id>{self.shape_id}</id>
    </{stroke_id}>"""
            strokes_xml.append(stroke_xml)
        
        output = f"""<concept>{concept}</concept>
<strokes>
{chr(10).join(strokes_xml)}
</strokes>"""
        
        return output 