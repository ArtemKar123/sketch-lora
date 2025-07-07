from base_shapes import BaseShape
from typing import List, Tuple, Dict, Any
import random
import math
import numpy as np
import sys
import os

# Import utils from parent directory
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import SketchAgent.utils as utils

class Curve(BaseShape):
    """Generator for natural curves - arcs and simple human-like curves."""
    
    # Curve-specific constants
    MIN_CURVE_POINTS = 3
    MAX_CURVE_POINTS = 6
    MIN_CURVE_SPAN = 15  # Minimum distance between start and end
    MIN_POINT_SEPARATION = 3  # Minimum distance between consecutive points (both x and y)
    
    # T-value patterns for main stroke types
    T_VALUE_PATTERNS = {
        'even_spacing': lambda n: [i/(max(n-1, 1)) for i in range(n)],
        'faster_start': lambda n: [0.0] + [0.4 + 0.6*i/(max(n-3, 1)) for i in range(max(n-2, 0))] + [1.0] if n > 2 else [0.0, 1.0],
        'faster_end': lambda n: [0.0] + [0.6*i/(max(n-3, 1)) for i in range(max(n-2, 0))] + [1.0] if n > 2 else [0.0, 1.0],
        'smooth': lambda n: [(1 - math.cos(i * math.pi / max(n-1, 1))) / 2 for i in range(n)],
    }
    
    def __init__(self, grid_size: int = BaseShape.DEFAULT_GRID_SIZE):
        super().__init__(grid_size)
        self.shape_id = "curve"
        
        # Calculate curve-specific parameters
        self.min_curve_span = max(self.MIN_CURVE_SPAN, int(self.grid_size * 0.15))
        self.max_curve_span = int(self.grid_size * 0.6)
        self.curve_margin = max(self.edge_margin * 2, 10)
    
    def _clamp_point(self, x: float, y: float) -> Tuple[int, int]:
        """Safely clamp a point to grid bounds with enhanced margins."""
        margin = self.curve_margin
        clamped_x = max(margin, min(self.grid_size - margin, int(x)))
        clamped_y = max(margin, min(self.grid_size - margin, int(y)))
        return (clamped_x, clamped_y)
    
    def _safe_distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Calculate distance between points with minimum return value to prevent division by zero."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        distance = math.sqrt(dx*dx + dy*dy)
        return max(distance, 1.0)
    
    def _safe_atan2(self, y: float, x: float) -> float:
        """Safe atan2 that handles zero cases."""
        if abs(x) < 1e-10 and abs(y) < 1e-10:
            return 0.0
        return math.atan2(y, x)
    
    def generate_sample(self, apply_random_transforms: bool = True, start_point: Tuple[int, int] = None, x_range: Tuple[int, int] = None, y_range: Tuple[int, int] = None) -> Dict[str, Any]:
        """Generate a complete curve sample with proper curve fitting using utils.
        
        Args:
            apply_random_transforms: Whether to apply random transformations
            start_point: Optional starting point for the curve
            x_range: Optional (min_x, max_x) constraint for all points
            y_range: Optional (min_y, max_y) constraint for all points
        """
        
        # Step 1: Generate original points P and t_values using our logic
        original_points, t_values, pattern_used = self._generate_curve_with_pattern(start_point, x_range, y_range)
        
        # Step 2: Fit curve to original points and get fitted points P' using utils
        final_points = self._fit_and_get_fitted_points(original_points, t_values)
        
        # Apply transformations to final points if requested
        if apply_random_transforms:
            final_points, t_values_transformed, transformations = self.apply_random_transformations(final_points, t_values)
        else:
            transformations = {'scale': 1.0, 'shift_x': 0, 'shift_y': 0, 'rotation_deg': 0.0}
            t_values_transformed = t_values
        
        # Format final points for output
        formatted_points = self.format_points_for_prompt(final_points)
        
        return {
            'shape_id': self.shape_id,
            'raw_points': final_points,  # P' - fitted points for output
            'original_points': original_points,  # P - original points for visualization comparison
            'points': formatted_points,  # P' formatted
            't_values': t_values_transformed,
            'transformations': transformations,
            'pattern_type': pattern_used
        }
    
    def generate_shape_curve(self, shape_type: str) -> Tuple[List[Tuple[int, int]], List[float]]:
        """Generate natural curves with proper curve fitting."""
        
        if shape_type == 'arc_shape':
            original_points, t_values = self._generate_arc()
        else:  # 'curve' or default
            original_points, t_values = self._generate_curve()
        
        # Fit curve to original points and get fitted points
        final_points = self._fit_and_get_fitted_points(original_points, t_values)
        
        # Recalculate t_values if point count changed during fitting
        if len(final_points) != len(t_values):
            pattern_name = 'even_spacing'  # Default fallback
            t_values = self.T_VALUE_PATTERNS[pattern_name](len(final_points))
        
        return final_points, t_values
    
    def _generate_curve_with_pattern(self, start_point: Tuple[int, int] = None, x_range: Tuple[int, int] = None, y_range: Tuple[int, int] = None) -> Tuple[List[Tuple[int, int]], List[float], str]:
        """Generate a random natural curve and return the pattern used.
        
        Args:
            start_point: Optional starting point for the curve
            x_range: Optional (min_x, max_x) constraint for all points
            y_range: Optional (min_y, max_y) constraint for all points
        """
        
        # Randomly choose detail level
        num_points = random.randint(self.MIN_CURVE_POINTS, self.MAX_CURVE_POINTS)
        
        # Randomly choose t_value pattern for variety
        pattern_name = random.choice(list(self.T_VALUE_PATTERNS.keys()))
        
        # Generate t_values first
        t_values = self.T_VALUE_PATTERNS[pattern_name](num_points)
        
        # Generate points that match the t-value spacing
        points = self._generate_curve_points_for_t_values(t_values, start_point, x_range, y_range)
        
        # Validate point separation
        points = self._validate_point_separation(points)
        
        # Recalculate t_values if point count changed during validation
        if len(points) != len(t_values):
            t_values = self.T_VALUE_PATTERNS[pattern_name](len(points))
        
        return points, t_values, pattern_name
    
    def _generate_curve(self) -> Tuple[List[Tuple[int, int]], List[float]]:
        """Generate a natural, human-like curve."""
        num_points = random.choice([3, 4, 5])
        pattern_name = random.choice(['even_spacing', 'smooth'])
        t_values = self.T_VALUE_PATTERNS[pattern_name](num_points)
        
        points = self._generate_curve_points_for_t_values(t_values)
        points = self._validate_point_separation(points)
        
        if len(points) != len(t_values):
            t_values = self.T_VALUE_PATTERNS[pattern_name](len(points))
        
        return points, t_values
    
    def _generate_arc(self) -> Tuple[List[Tuple[int, int]], List[float]]:
        """Generate a simple arc."""
        num_points = random.choice([3, 4, 5])
        pattern_name = random.choice(['even_spacing', 'smooth'])
        t_values = self.T_VALUE_PATTERNS[pattern_name](num_points)
        
        points = self._generate_arc_points_for_t_values(t_values)
        points = self._validate_point_separation(points)
        
        if len(points) != len(t_values):
            t_values = self.T_VALUE_PATTERNS[pattern_name](len(points))
        
        return points, t_values
    
    def _generate_constrained_arc(self, start_point: Tuple[int, int] = None, x_range: Tuple[int, int] = None, y_range: Tuple[int, int] = None) -> Tuple[List[Tuple[int, int]], List[float]]:
        """Generate an arc with start point and directional constraints.
        
        Args:
            start_point: Optional starting point for the arc
            x_range: Optional (min_x, max_x) constraint for all points
            y_range: Optional (min_y, max_y) constraint for all points
        """
        num_points = random.choice([3, 4, 5])
        pattern_name = random.choice(['even_spacing', 'smooth'])
        t_values = self.T_VALUE_PATTERNS[pattern_name](num_points)
        
        points = self._generate_constrained_arc_points(t_values, start_point, x_range, y_range)
        points = self._validate_point_separation(points)
        
        if len(points) != len(t_values):
            t_values = self.T_VALUE_PATTERNS[pattern_name](len(points))
        
        return points, t_values
    
    def _fit_and_get_fitted_points(self, original_points: List[Tuple[int, int]], t_values: List[float]) -> List[Tuple[int, int]]:
        """Fit a curve to original points using utils and return fitted points."""
        
        if len(original_points) < 2:
            return original_points
        
        try:
            # Convert points to the format expected by utils (swap x,y to y,x for utils)
            utils_points = [[p[1], p[0]] for p in original_points]  # utils expects [y,x] format
            
            # Use utils to fit curve and get control points
            control_points_groups = utils.estimate_bezier_control_points(utils_points, t_values)
            
            # Sample fitted points from the curve at our t_values
            fitted_points = []
            
            for t in t_values:
                if len(control_points_groups) == 1:
                    # Single curve segment
                    control_points = control_points_groups[0]
                    
                    if len(control_points) == 1:
                        # Single point
                        point = control_points[0]
                    elif len(control_points) == 2:
                        # Linear interpolation
                        point = (1-t) * np.array(control_points[0]) + t * np.array(control_points[1])
                    else:
                        # Bezier curve - use utils function
                        point = utils.bezier_point(control_points, t)
                    
                    # Convert back from utils format [y,x] to our format (x,y)
                    x, y = int(point[1]), int(point[0])
                    clamped_point = self._clamp_point(x, y)
                    fitted_points.append(clamped_point)
                
                else:
                    # Multiple curve segments - choose segment based on t
                    segment_index = min(int(t * len(control_points_groups)), len(control_points_groups) - 1)
                    control_points = control_points_groups[segment_index]
                    
                    # Normalize t for this segment
                    segment_t = (t * len(control_points_groups)) % 1.0
                    
                    if len(control_points) == 1:
                        point = control_points[0]
                    elif len(control_points) == 2:
                        point = (1-segment_t) * np.array(control_points[0]) + segment_t * np.array(control_points[1])
                    else:
                        point = utils.bezier_point(control_points, segment_t)
                    
                    # Convert back from utils format [y,x] to our format (x,y)
                    x, y = int(point[1]), int(point[0])
                    clamped_point = self._clamp_point(x, y)
                    fitted_points.append(clamped_point)
            
            # Validate point separation
            fitted_points = self._validate_point_separation(fitted_points)
            
            return fitted_points
            
        except Exception as e:
            # print(f"Curve fitting failed: {e}, using original points")
            # Fallback to original points if fitting fails
            return original_points
    
    def _validate_point_separation(self, points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Ensure consecutive points have sufficient separation in both x and y directions."""
        if len(points) < 2:
            return points
        
        validated_points = [points[0]]  # Always keep the first point
        
        for i in range(1, len(points)):
            current_point = points[i]
            prev_point = validated_points[-1]
            
            # Check if current point is too close to previous point
            dx = abs(current_point[0] - prev_point[0])
            dy = abs(current_point[1] - prev_point[1])
            
            if dx < self.MIN_POINT_SEPARATION or dy < self.MIN_POINT_SEPARATION:
                # Adjust the current point to maintain minimum separation
                new_x = current_point[0]
                new_y = current_point[1]
                
                # Adjust x if too close
                if dx < self.MIN_POINT_SEPARATION:
                    if current_point[0] >= prev_point[0]:
                        new_x = prev_point[0] + self.MIN_POINT_SEPARATION
                    else:
                        new_x = prev_point[0] - self.MIN_POINT_SEPARATION
                
                # Adjust y if too close
                if dy < self.MIN_POINT_SEPARATION:
                    if current_point[1] >= prev_point[1]:
                        new_y = prev_point[1] + self.MIN_POINT_SEPARATION
                    else:
                        new_y = prev_point[1] - self.MIN_POINT_SEPARATION
                
                # Clamp to grid bounds and validate the adjusted point is still valid
                adjusted_point = self._clamp_point(new_x, new_y)
                
                # Double-check the adjusted point still has proper separation
                final_dx = abs(adjusted_point[0] - prev_point[0])
                final_dy = abs(adjusted_point[1] - prev_point[1])
                
                # Only add if we achieved proper separation, otherwise skip this point
                if final_dx >= self.MIN_POINT_SEPARATION and final_dy >= self.MIN_POINT_SEPARATION:
                    validated_points.append(adjusted_point)
            else:
                validated_points.append(current_point)
        
        # Ensure we have at least 2 points for a valid curve
        if len(validated_points) < 2:
            # If validation removed too many points, regenerate with larger separation
            return self._generate_fallback_points()
        
        return validated_points
    
    def _generate_fallback_points(self) -> List[Tuple[int, int]]:
        """Generate a simple fallback curve with guaranteed point separation."""
        margin = self.curve_margin
        
        # Generate simple 2-point curve with guaranteed separation
        start_x = random.randint(margin, self.grid_size - margin - self.MIN_POINT_SEPARATION * 2)
        start_y = random.randint(margin, self.grid_size - margin - self.MIN_POINT_SEPARATION * 2)
        
        end_x = start_x + random.randint(self.MIN_POINT_SEPARATION * 2, min(self.max_curve_span, self.grid_size - start_x - margin))
        end_y = start_y + random.randint(self.MIN_POINT_SEPARATION * 2, min(self.max_curve_span, self.grid_size - start_y - margin))
        
        end_x = max(margin, min(self.grid_size - margin, end_x))
        end_y = max(margin, min(self.grid_size - margin, end_y))
        
        return [(start_x, start_y), (end_x, end_y)]
    
    def _generate_curve_points_for_t_values(self, t_values: List[float], start_point: Tuple[int, int] = None, x_range: Tuple[int, int] = None, y_range: Tuple[int, int] = None) -> List[Tuple[int, int]]:
        """Generate curve points that match the given t-value spacing along a planned curve path.
        
        Args:
            t_values: List of t-values from 0 to 1
            start_point: Optional starting point for the curve
            x_range: Optional (min_x, max_x) constraint for all points
            y_range: Optional (min_y, max_y) constraint for all points
        """
        
        if len(t_values) < 2:
            raise ValueError("Need at least 2 t-values to generate a curve")
        
        # Calculate effective bounds considering margins and optional ranges
        margin = max(self.curve_margin, int(self.grid_size * 0.08))
        
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
            # Use provided start point, constrained within bounds
            start_x, start_y = start_point
            start_x = max(min_x, min(max_x, start_x))
            start_y = max(min_y, min(max_y, start_y))
        else:
            # Generate random start point within bounds
            start_x = random.randint(min_x, max_x)
            start_y = random.randint(min_y, max_y)
        
        # Ensure end point is sufficiently far from start and within bounds
        max_attempts = 50
        for attempt in range(max_attempts):
            end_x = random.randint(min_x, max_x)
            end_y = random.randint(min_y, max_y)
            distance = self._safe_distance((start_x, start_y), (end_x, end_y))
            if distance >= self.min_curve_span:
                break
        else:
            # Fallback: create a valid end point within constraints
            angle = random.uniform(0, 2*math.pi)
            end_x = start_x + int(self.min_curve_span * math.cos(angle))
            end_y = start_y + int(self.min_curve_span * math.sin(angle))
            end_x = max(min_x, min(max_x, end_x))
            end_y = max(min_y, min(max_y, end_y))
        
        # Define the planned curve path (a simple arc)
        line_angle = self._safe_atan2(end_y - start_y, end_x - start_x)
        curve_direction = random.choice([-1, 1])  # Left or right curve
        
        # Calculate curve control parameters
        line_length = self._safe_distance((start_x, start_y), (end_x, end_y))
        max_deviation = min(20, int(line_length * 0.3))  # Natural curvature
        
        points = []
        for i, t in enumerate(t_values):
            if t == 0.0:
                # Start point
                points.append((start_x, start_y))
            elif t == 1.0:
                # End point
                points.append((end_x, end_y))
            else:
                # Intermediate point positioned according to t-value
                # Base position along straight line
                base_x = start_x + t * (end_x - start_x)
                base_y = start_y + t * (end_y - start_y)
                
                # Add curvature perpendicular to the line based on t-value
                perp_angle = line_angle + math.pi / 2
                
                # Curvature strength follows a natural arc (sin curve)
                curve_strength = max_deviation * math.sin(t * math.pi) * curve_direction
                
                # Apply curvature
                curve_x = base_x + curve_strength * math.cos(perp_angle)
                curve_y = base_y + curve_strength * math.sin(perp_angle)
                
                # Constrain to our specific bounds (not just grid bounds)
                curve_x = max(min_x, min(max_x, int(curve_x)))
                curve_y = max(min_y, min(max_y, int(curve_y)))
                points.append((curve_x, curve_y))
        
        return points
    
    def _generate_arc_points_for_t_values(self, t_values: List[float]) -> List[Tuple[int, int]]:
        """Generate arc points that match the given t-value spacing."""
        
        if len(t_values) < 2:
            raise ValueError("Need at least 2 t-values to generate an arc")
        
        margin = self.curve_margin
        safe_area = self.grid_size - 2 * margin
        
        if safe_area <= 0:
            center = self.grid_size // 2
            return [(center, center)] * len(t_values)
        
        # Choose simple arc orientation
        orientation = random.choice(['horizontal', 'vertical'])
        
        # Simple arc dimensions - not too complex
        arc_width = random.randint(self.min_curve_span, min(safe_area // 2, self.max_curve_span//2))
        arc_height = random.randint(self.min_curve_span//2, min(safe_area // 3, self.max_curve_span//3))
        
        if orientation == 'horizontal':
            # Horizontal arc (like a smile or frown)
            center_x = self.grid_size // 2
            center_y = random.randint(margin + arc_height, self.grid_size - margin - arc_height)
            
            left_x = max(margin, center_x - arc_width//2)
            right_x = min(self.grid_size - margin, center_x + arc_width//2)
            
            # Randomly choose smile or frown
            arc_direction = random.choice([1, -1])  # 1 for up (smile), -1 for down (frown)
            
        else:  # vertical
            # Vertical arc (like a bracket)
            center_y = self.grid_size // 2
            center_x = random.randint(margin + arc_height, self.grid_size - margin - arc_height)
            
            top_y = max(margin, center_y - arc_width//2)
            bottom_y = min(self.grid_size - margin, center_y + arc_width//2)
            
            # Randomly choose left or right arc
            arc_direction = random.choice([1, -1])  # 1 for right, -1 for left
        
        # Generate points based on t_values
        points = []
        for i, t in enumerate(t_values):
            if orientation == 'horizontal':
                # Interpolate along horizontal arc
                x = left_x + t * (right_x - left_x)
                # Calculate y based on arc shape
                if t == 0.0 or t == 1.0:
                    y = center_y
                else:
                    # Use sine function for natural arc shape
                    arc_progress = math.sin(t * math.pi)
                    y = center_y - arc_direction * arc_height * arc_progress
                points.append(self._clamp_point(x, y))
            else:  # vertical
                # Interpolate along vertical arc
                y = top_y + t * (bottom_y - top_y)
                # Calculate x based on arc shape
                if t == 0.0 or t == 1.0:
                    x = center_x
                else:
                    # Use sine function for natural arc shape
                    arc_progress = math.sin(t * math.pi)
                    x = center_x + arc_direction * arc_height * arc_progress
                points.append(self._clamp_point(x, y))
        
        return points
    
    def _generate_constrained_arc_points(self, t_values: List[float], start_point: Tuple[int, int] = None, x_range: Tuple[int, int] = None, y_range: Tuple[int, int] = None) -> List[Tuple[int, int]]:
        """Generate arc points with start point and directional constraints.
        
        Args:
            t_values: List of t-values from 0 to 1
            start_point: Optional starting point for the arc
            x_range: Optional (min_x, max_x) constraint for all points
            y_range: Optional (min_y, max_y) constraint for all points
        """
        
        if len(t_values) < 2:
            raise ValueError("Need at least 2 t-values to generate an arc")
        
        # Calculate effective bounds considering margins and optional ranges
        margin = max(self.curve_margin, int(self.grid_size * 0.08))
        
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
            # Use provided start point, constrained within bounds
            start_x, start_y = start_point
            start_x = max(min_x, min(max_x, start_x))
            start_y = max(min_y, min(max_y, start_y))
        else:
            # Generate random start point within bounds
            start_x = random.randint(min_x, max_x)
            start_y = random.randint(min_y, max_y)
        
        # Determine arc dimensions based on available space
        available_width = max_x - min_x
        available_height = max_y - min_y
        
        # Calculate reasonable arc size within constraints
        max_arc_width = min(available_width, self.max_curve_span//2)
        max_arc_height = min(available_height, self.max_curve_span//3)
        
        arc_width = random.randint(self.min_curve_span, max(self.min_curve_span, max_arc_width))
        arc_height = random.randint(self.min_curve_span//2, max(self.min_curve_span//2, max_arc_height))
        
        # Determine arc orientation based on available space and constraints
        # Prefer horizontal arcs if we have more width, vertical if more height
        if available_width >= available_height:
            orientation = 'horizontal'
        else:
            orientation = 'vertical'
        
        # Override orientation if space is too constrained for it
        if orientation == 'horizontal' and arc_width > available_width:
            orientation = 'vertical'
        elif orientation == 'vertical' and arc_height > available_height:
            orientation = 'horizontal'
        
        # Calculate arc center and bounds
        if orientation == 'horizontal':
            # Horizontal arc (like a smile or frown)
            center_x = start_x
            center_y = start_y
            
            # Calculate left and right bounds within constraints
            left_x = max(min_x, center_x - arc_width//2)
            right_x = min(max_x, center_x + arc_width//2)
            
            # Adjust center if arc doesn't fit symmetrically
            if right_x - left_x < arc_width:
                actual_width = right_x - left_x
                center_x = left_x + actual_width//2
            
            # Choose arc direction (up/down) based on available space
            if center_y + arc_height <= max_y:
                arc_direction = 1  # Upward arc
            elif center_y - arc_height >= min_y:
                arc_direction = -1  # Downward arc
            else:
                # Use whatever fits better
                up_space = max_y - center_y
                down_space = center_y - min_y
                arc_direction = 1 if up_space >= down_space else -1
                arc_height = min(arc_height, max(up_space, down_space))
        
        else:  # vertical
            # Vertical arc (like a bracket)
            center_x = start_x
            center_y = start_y
            
            # Calculate top and bottom bounds within constraints
            top_y = max(min_y, center_y - arc_width//2)
            bottom_y = min(max_y, center_y + arc_width//2)
            
            # Adjust center if arc doesn't fit symmetrically
            if bottom_y - top_y < arc_width:
                actual_height = bottom_y - top_y
                center_y = top_y + actual_height//2
            
            # Choose arc direction (left/right) based on available space
            if center_x + arc_height <= max_x:
                arc_direction = 1  # Rightward arc
            elif center_x - arc_height >= min_x:
                arc_direction = -1  # Leftward arc
            else:
                # Use whatever fits better
                right_space = max_x - center_x
                left_space = center_x - min_x
                arc_direction = 1 if right_space >= left_space else -1
                arc_height = min(arc_height, max(right_space, left_space))
        
        # Generate points based on t_values and orientation
        points = []
        for i, t in enumerate(t_values):
            if orientation == 'horizontal':
                # Interpolate along horizontal arc
                x = left_x + t * (right_x - left_x)
                # Calculate y based on arc shape
                if t == 0.0 or t == 1.0:
                    y = center_y
                else:
                    # Use sine function for natural arc shape
                    arc_progress = math.sin(t * math.pi)
                    y = center_y + arc_direction * arc_height * arc_progress
                points.append(self._clamp_point(x, y))
            else:  # vertical
                # Interpolate along vertical arc
                y = top_y + t * (bottom_y - top_y)
                # Calculate x based on arc shape
                if t == 0.0 or t == 1.0:
                    x = center_x
                else:
                    # Use sine function for natural arc shape
                    arc_progress = math.sin(t * math.pi)
                    x = center_x + arc_direction * arc_height * arc_progress
                points.append(self._clamp_point(x, y))
        
        return points 