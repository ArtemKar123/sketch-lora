from typing import List, Dict, Any, Tuple
import random


class PromptGenerator:
    """Generates varied text prompts for shape drawing tasks."""
    
    # Constants for directional line detection
    DIRECTIONAL_TOLERANCE_RATIO = (
        0.02  # 2% of grid size for horizontal/vertical detection
    )
    MIN_DIRECTIONAL_TOLERANCE = 2
    
    def __init__(self, grid_size: int = 100):
        self.grid_size = grid_size
        self.directional_tolerance = max(
            self.MIN_DIRECTIONAL_TOLERANCE, 
            int(self.grid_size * self.DIRECTIONAL_TOLERANCE_RATIO),
        )
        
        # Line prompt templates - all with coordinates
        self.line_templates = [
            "Draw a straight line from ({x1}, {y1}) to ({x2}, {y2})",
            "Sketch a line from ({x1}, {y1}) to ({x2}, {y2})",
        ]
        
        self.horizontal_line_templates = [
            "Draw a horizontal line from ({x1}, {y1}) to ({x2}, {y2})",
            "Sketch a horizontal line from ({x1}, {y1}) to ({x2}, {y2})",
        ]
        
        self.vertical_line_templates = [
            "Draw a vertical line from ({x1}, {y1}) to ({x2}, {y2})",
            "Sketch a vertical line from ({x1}, {y1}) to ({x2}, {y2})",
        ]
        
        self.diagonal_line_templates = [
            "Draw a diagonal line from ({x1}, {y1}) to ({x2}, {y2})",
            "Sketch a diagonal line from ({x1}, {y1}) to ({x2}, {y2})",
        ]
        
        self.general_line_templates = [
            "Draw a straight line from ({x1}, {y1}) to ({x2}, {y2})",
            "Sketch a line from ({x1}, {y1}) to ({x2}, {y2})",
        ]
        
        # Curve prompt templates
        self.curve_basic_templates = [
            "Draw a curve from ({x1}, {y1}) to ({x2}, {y2})",
            "Sketch a curved line from ({x1}, {y1}) to ({x2}, {y2})",
        ]
        
        self.curve_directional_templates = {
            "left": [
                "Draw a curve from ({x1}, {y1}) to ({x2}, {y2}) that bends to the left",
                "Sketch a left-curving line from ({x1}, {y1}) to ({x2}, {y2})",
            ],
            "right": [
                "Draw a curve from ({x1}, {y1}) to ({x2}, {y2}) that bends to the right",
                "Sketch a right-curving line from ({x1}, {y1}) to ({x2}, {y2})",
            ],
            "up": [
                "Draw a curve from ({x1}, {y1}) to ({x2}, {y2}) that bulges upward",
                "Sketch a curve arcing upward from ({x1}, {y1}) to ({x2}, {y2})",
            ],
            "down": [
                "Draw a curve from ({x1}, {y1}) to ({x2}, {y2}) that sags downward",
                "Sketch a curve arcing downward from ({x1}, {y1}) to ({x2}, {y2})",
            ],
        }
        
        self.curve_multipoint_templates = [
            "Draw a curve that passes through ({x1}, {y1}), ({x2}, {y2}), ({x3}, {y3})",
            "Sketch a curved path through ({x1}, {y1}), ({x2}, {y2}), ({x3}, {y3})",
        ]
        
        # Specific curve shape templates
        self.arc_shape_templates = [
            "Draw an arc from ({x1}, {y1}) to ({x2}, {y2})",
            "Sketch an arc connecting ({x1}, {y1}) to ({x2}, {y2})",
        ]
        
        self.curve_templates = [
            "Draw a curve from ({x1}, {y1}) to ({x2}, {y2})",
            "Sketch a curved stroke from ({x1}, {y1}) to ({x2}, {y2})",
        ]
        
        # Shape template mappings
        self.shape_templates = {
            "arc_shape": self.arc_shape_templates,
            "curve": self.curve_templates,
        }
        
        # T-value pattern specific prompts
        self.pattern_specific_templates = {
            "even_spacing": [
                "Draw a curve from ({x1}, {y1}) to ({x2}, {y2}) with even spacing",
                "Sketch a uniform curve from ({x1}, {y1}) to ({x2}, {y2})",
            ],
            "faster_start": [
                "Draw a curve from ({x1}, {y1}) to ({x2}, {y2}) starting fast then slowing down",
                "Sketch a fast-starting curve from ({x1}, {y1}) to ({x2}, {y2})",
            ],
            "faster_end": [
                "Draw a curve from ({x1}, {y1}) to ({x2}, {y2}) starting slow then accelerating",
                "Sketch an accelerating curve from ({x1}, {y1}) to ({x2}, {y2})",
            ],
            "smooth": [
                "Draw a smooth curve from ({x1}, {y1}) to ({x2}, {y2}) with natural flow",
                "Sketch a flowing curve from ({x1}, {y1}) to ({x2}, {y2})",
            ],
        }
        
        # Geometric shape templates
        self.geometric_basic_templates = [
            "Draw a {shape_name} with corners at ({x1}, {y1}), ({x2}, {y2}), ({x3}, {y3}), ({x4}, {y4})",
            "Sketch a {shape_name} from ({x1}, {y1}) through ({x2}, {y2}), ({x3}, {y3}) to ({x4}, {y4})",
        ]
        
        # Corner-style specific templates
        self.rounded_corner_templates = [
            "Draw a {shape_name} with rounded corners at ({x1}, {y1}), ({x2}, {y2}), ({x3}, {y3}), ({x4}, {y4})",
            "Sketch a {shape_name} with soft corners at ({x1}, {y1}), ({x2}, {y2}), ({x3}, {y3}), ({x4}, {y4})",
        ]
        
        self.sharp_inward_templates = [
            "Draw a {shape_name} with sharp corners at ({x1}, {y1}), ({x2}, {y2}), ({x3}, {y3}), ({x4}, {y4})",
            "Sketch a {shape_name} with sharp corners and inward-bending edges going through ({x1}, {y1}), ({x2}, {y2}), ({x3}, {y3}), ({x4}, {y4})",
        ]
        
        self.square_templates = [
            "Draw a square with corners at ({x1}, {y1}), ({x2}, {y2}), ({x3}, {y3}), ({x4}, {y4})",
            "Sketch a square shape from ({x1}, {y1}) to ({x2}, {y2}) to ({x3}, {y3}) to ({x4}, {y4})",
        ]
        
        self.rectangle_templates = [
            "Draw a rectangle with corners at ({x1}, {y1}), ({x2}, {y2}), ({x3}, {y3}), ({x4}, {y4})",
            "Sketch a rectangle from ({x1}, {y1}) through ({x2}, {y2}), ({x3}, {y3}) to ({x4}, {y4})",
        ]
        
        self.rhombus_templates = [
            "Draw a rhombus with corners at ({x1}, {y1}), ({x2}, {y2}), ({x3}, {y3}), ({x4}, {y4})",
            "Sketch a diamond shape connecting ({x1}, {y1}), ({x2}, {y2}), ({x3}, {y3}), ({x4}, {y4})",
        ]
        
        self.geometric_size_templates = [
            "Draw a {size} {shape_name} with corners at ({x1}, {y1}), ({x2}, {y2}), ({x3}, {y3}), ({x4}, {y4})",
            "Sketch a {size} {shape_name} from ({x1}, {y1}) to ({x4}, {y4})",
        ]
        
        self.geometric_position_templates = [
            "Draw a {shape_name} positioned with corners at ({x1}, {y1}), ({x2}, {y2}), ({x3}, {y3}), ({x4}, {y4})",
            "Sketch a {shape_name} at coordinates ({x1}, {y1}), ({x2}, {y2}), ({x3}, {y3}), ({x4}, {y4})",
        ]
        
        self.geometric_general_templates = [
            "Draw a {shape_name} with corners at ({x1}, {y1}), ({x2}, {y2}), ({x3}, {y3}), ({x4}, {y4})",
            "Sketch a {shape_name} connecting the corner points",
        ]
        
        # Geometric shape template mappings
        self.geometric_templates = {
            "square": self.square_templates,
            "rectangle": self.rectangle_templates,
            "rhombus": self.rhombus_templates,
        }
        
        # Triangular shape templates
        self.triangular_basic_templates = [
            "Draw a {shape_name} with corners at ({x1}, {y1}), ({x2}, {y2}), ({x3}, {y3})",
            "Sketch a {shape_name} triangle connecting ({x1}, {y1}), ({x2}, {y2}), ({x3}, {y3})",
        ]
        
        # Triangle-specific templates
        self.equilateral_templates = [
            "Draw an equilateral triangle with equal sides connecting ({x1}, {y1}), ({x2}, {y2}), ({x3}, {y3})",
            "Sketch an equal-sided triangle with vertices at ({x1}, {y1}), ({x2}, {y2}), ({x3}, {y3})",
        ]
        
        self.isosceles_templates = [
            "Draw an isosceles triangle with two equal sides connecting ({x1}, {y1}), ({x2}, {y2}), ({x3}, {y3})",
            "Sketch a triangle with two equal sides at ({x1}, {y1}), ({x2}, {y2}), ({x3}, {y3})",
        ]
        
        self.right_templates = [
            "Draw a right triangle with a 90-degree angle connecting ({x1}, {y1}), ({x2}, {y2}), ({x3}, {y3})",
            "Sketch a right-angled triangle with vertices at ({x1}, {y1}), ({x2}, {y2}), ({x3}, {y3})",
        ]
        
        self.scalene_templates = [
            "Draw a scalene triangle with all different sides connecting ({x1}, {y1}), ({x2}, {y2}), ({x3}, {y3})",
            "Sketch an irregular triangle at ({x1}, {y1}), ({x2}, {y2}), ({x3}, {y3})",
        ]
        
        # Corner-style specific templates for triangles
        self.triangular_rounded_corner_templates = [
            "Draw a {shape_name} with rounded corners at ({x1}, {y1}), ({x2}, {y2}), ({x3}, {y3})",
            "Sketch a {shape_name} with soft corner curves at ({x1}, {y1}), ({x2}, {y2}), ({x3}, {y3})",
        ]
        
        self.triangular_sharp_inward_templates = [
            "Draw a {shape_name} with sharp corners at ({x1}, {y1}), ({x2}, {y2}), ({x3}, {y3})",
            "Sketch a {shape_name} with pointed corners at ({x1}, {y1}), ({x2}, {y2}), ({x3}, {y3})",
        ]
        
        self.triangular_size_templates = [
            "Draw a {size} {shape_name} with corners at ({x1}, {y1}), ({x2}, {y2}), ({x3}, {y3})",
            "Sketch a {size} {shape_name} from ({x1}, {y1}) to ({x3}, {y3})",
        ]
        
        self.triangular_position_templates = [
            "Draw a {shape_name} positioned with corners at ({x1}, {y1}), ({x2}, {y2}), ({x3}, {y3})",
            "Sketch a {shape_name} at coordinates ({x1}, {y1}), ({x2}, {y2}), ({x3}, {y3})",
        ]
        
        self.triangular_general_templates = [
            "Draw a {shape_name} with corners at ({x1}, {y1}), ({x2}, {y2}), ({x3}, {y3})",
            "Sketch a {shape_name} connecting the three points",
        ]
        
        # Triangular shape template mappings
        self.triangular_templates = {
            "equilateral": self.equilateral_templates,
            "isosceles": self.isosceles_templates,
            "right": self.right_templates,
            "scalene": self.scalene_templates,
        }
        
        # Circular shape templates
        self.circular_basic_templates = [
            "Draw a {shape_name} with center at ({center_x}, {center_y})",
            "Sketch a {shape_name} starting and ending at ({start_x}, {start_y})",
        ]
        
        # Circle-specific templates
        self.circle_templates = [
            "Draw a perfect circle with center at ({center_x}, {center_y}) and radius {radius}",
            "Sketch a circular shape starting at ({start_x}, {start_y})",
        ]
        
        # Ellipse-specific templates
        self.ellipse_templates = [
            "Draw an ellipse with center at ({center_x}, {center_y})",
            "Sketch an oval shape starting at ({start_x}, {start_y})",
        ]
        
        # Size-based circular templates
        self.circular_size_templates = [
            "Draw a {size} {shape_name} with center near ({center_x}, {center_y})",
            "Sketch a {size} circular shape starting at ({start_x}, {start_y})",
        ]
        
        # Position-based circular templates
        self.circular_position_templates = [
            "Draw a {shape_name} positioned with center at ({center_x}, {center_y})",
            "Sketch a {shape_name} centered around ({center_x}, {center_y})",
        ]
        
        # General circular templates
        self.circular_general_templates = [
            "Draw a circular {shape_name} that forms a closed loop",
            "Sketch a round shape with continuous curves",
        ]
        
        # Circular shape template mappings
        self.circular_templates = {
            "circle": self.circle_templates,
            "ellipse": self.ellipse_templates,
        }
        
        # Angular shape templates
        self.angular_basic_templates = [
            "Draw a {shape_name} with sharp corners connecting ({x1}, {y1}) to ({x2}, {y2})",
            "Sketch a {shape_name} with angular edges from ({x1}, {y1}) to ({x2}, {y2})",
        ]
        
        # V-shape specific templates
        self.v_shape_templates = [
            "Draw a V-shape from ({x1}, {y1}) to ({x2}, {y2}) with a sharp corner",
            "Sketch a pointed V-shape from ({x1}, {y1}) to ({x2}, {y2})",
        ]
        
        # L-shape specific templates
        self.l_shape_templates = [
            "Draw an L-shape from ({x1}, {y1}) to ({x2}, {y2}) with a sharp corner",
            "Sketch a sharp L-shape from ({x1}, {y1}) to ({x2}, {y2})",
        ]
        
        # Zigzag specific templates
        self.zigzag_templates = [
            "Draw a zigzag pattern from ({x1}, {y1}) to ({x2}, {y2}) with sharp peaks",
            "Sketch a jagged zigzag from ({x1}, {y1}) to ({x2}, {y2})",
        ]
        
        # Step pattern templates
        self.step_templates = [
            "Draw a step pattern from ({x1}, {y1}) to ({x2}, {y2}) like stairs",
            "Sketch a staircase pattern from ({x1}, {y1}) to ({x2}, {y2})",
        ]
        
        # Angular shape template mappings
        self.angular_templates = {
            "v_shape": self.v_shape_templates,
            "l_shape": self.l_shape_templates,
            "zigzag": self.zigzag_templates,
            "step": self.step_templates,
        }
        
        # Size-based angular templates
        self.angular_size_templates = [
            "Draw a {size} {shape_name} from ({x1}, {y1}) to ({x2}, {y2})",
            "Sketch a {size} angular shape with sharp corners",
        ]
        
        # General angular templates
        self.angular_general_templates = [
            "Draw an angular {shape_name} from ({x1}, {y1}) to ({x2}, {y2})",
            "Sketch a {shape_name} with sharp direction changes",
        ]
        
        # Spatial relations prompt templates
        self.spatial_relations_templates = {
            "connected_stroke": [
                "Add a {stroke_description} {direction} from it",
                "Draw a {stroke_description} extending {direction} from the shape",
            ]
        }
        
        # Direction descriptions for natural language
        self.direction_descriptions = {
            "top": "to the top",
            "top_right": "to the top-right",
            "top_left": "to the top-left",
            "bottom": "to the bottom",
            "bottom_right": "to the bottom-right",
            "bottom_left": "to the bottom-left",
            "right": "to the right",
            "left": "to the left",
        }
        
        # Stroke type descriptions for spatial relations
        self.stroke_descriptions = {
            "line": "line",
            "curve": "curved stroke",
            "rectangle": "rectangle",
            "triangle": "triangle",
            "circle": "circle",
            "arc": "arc",
            "ellipse": "ellipse",
            "rhombus": "rhombus",
            "square": "square",
        }
    
    def generate_pattern_prompt(
        self, sample_data: Dict[str, Any], pattern_type: str
    ) -> str:
        """Generate a prompt based on the T-value pattern used."""
        raw_points = sample_data["raw_points"]
        if len(raw_points) < 2:
            raise ValueError("Shape must have at least 2 points")
        
        x1, y1 = raw_points[0]
        x2, y2 = raw_points[-1]
        
        if pattern_type in self.pattern_specific_templates:
            template = random.choice(self.pattern_specific_templates[pattern_type])
            return template.format(x1=x1, y1=y1, x2=x2, y2=y2)
        else:
            # Fallback to basic curve templates
            template = random.choice(self.curve_templates)
            return template.format(x1=x1, y1=y1, x2=x2, y2=y2)
    
    def generate_line_prompt(
        self, sample_data: Dict[str, Any], prompt_type: str = "specific"
    ) -> str:
        """Generate a text prompt for a line drawing task."""
        
        raw_points = sample_data["raw_points"]
        if len(raw_points) != 2:
            raise ValueError("Line must have exactly 2 points")
        
        x1, y1 = raw_points[0]
        x2, y2 = raw_points[1]
        
        if prompt_type == "specific":
            # Use specific coordinates
            template = random.choice(self.line_templates)
            return template.format(x1=x1, y1=y1, x2=x2, y2=y2)
        
        elif prompt_type == "directional":
            # Determine line direction using grid-relative tolerance
            tolerance = self.directional_tolerance
            
            if abs(y2 - y1) <= tolerance:  # Approximately horizontal
                template = random.choice(self.horizontal_line_templates)
                return template.format(x1=x1, y1=y1, x2=x2, y2=y2)
            elif abs(x2 - x1) <= tolerance:  # Approximately vertical  
                template = random.choice(self.vertical_line_templates)
                return template.format(x1=x1, y1=y1, x2=x2, y2=y2)
            else:  # Diagonal
                template = random.choice(self.diagonal_line_templates)
                return template.format(x1=x1, y1=y1, x2=x2, y2=y2)
        
        elif prompt_type == "general":
            template = random.choice(self.general_line_templates)
            return template.format(x1=x1, y1=y1, x2=x2, y2=y2)
        
        else:
            # Mix of all types
            prompt_types = ["specific", "directional", "general"]
            chosen_type = random.choice(prompt_types)
            return self.generate_line_prompt(sample_data, chosen_type)
    
    def generate_specific_shape_prompt(
        self, sample_data: Dict[str, Any], shape_type: str
    ) -> str:
        """Generate a prompt for a specific shape type."""
        
        raw_points = sample_data["raw_points"]
        if len(raw_points) < 2:
            raise ValueError("Shape must have at least 2 points")
        
        x1, y1 = raw_points[0]
        x2, y2 = raw_points[-1]
        
        # Check if it's a shape template
        if shape_type in self.shape_templates:
            template = random.choice(self.shape_templates[shape_type])
            return template.format(x1=x1, y1=y1, x2=x2, y2=y2)
        elif shape_type in self.geometric_templates:
            template = random.choice(self.geometric_templates[shape_type])
            return template.format(x1=x1, y1=y1, x2=x2, y2=y2)
        else:
            raise ValueError(f"Unknown shape type: {shape_type}")
    
    def generate_curve_prompt(
        self,
        sample_data: Dict[str, Any],
        prompt_type: str = "mixed",
        shape_type: str = None,
    ) -> str:
        """Generate a text prompt for a curve drawing task."""
        
        raw_points = sample_data["raw_points"]
        
        x1, y1 = raw_points[0]
        x2, y2 = raw_points[-1]
        
        # If specific shape type is provided, use it
        if shape_type:
            return self.generate_specific_shape_prompt(sample_data, shape_type)
        
        if prompt_type == "basic":
            # Basic curve with start/end coordinates
            template = random.choice(self.curve_templates)
            return template.format(x1=x1, y1=y1, x2=x2, y2=y2)
        
        elif prompt_type == "directional":
            # Generate curve with specific direction, then create matching prompt
            direction = random.choice(["left", "right", "up", "down"])
            
            # Import curves module to generate directional curve
            from curves import Curve

            curve_gen = Curve(self.grid_size)
            
            # Generate t_values with a good pattern for directional curves
            pattern_name = random.choice(["even_spacing", "smooth"])
            t_values = curve_gen.T_VALUE_PATTERNS[pattern_name](len(raw_points))
            
            # Generate directional curve points (this replaces the original points)
            directional_points = curve_gen._generate_directional_curve_points(
                t_values, direction
            )
            
            # Update sample_data with new directional points
            sample_data["raw_points"] = directional_points
            sample_data["points"] = [f"x{x}y{y}" for x, y in directional_points]
            
            # Use first and last points for prompt
            x1, y1 = directional_points[0]
            x2, y2 = directional_points[-1]
            
            template = random.choice(self.curve_directional_templates[direction])
            return template.format(x1=x1, y1=y1, x2=x2, y2=y2)
        
        elif prompt_type == "multipoint":
            # Multi-point curve prompts
            if len(raw_points) >= 3:
                # Choose 3 representative points
                points_to_use = [
                    raw_points[0],
                    raw_points[len(raw_points) // 2],
                    raw_points[-1],
                ]
                x1, y1 = points_to_use[0]
                x2, y2 = points_to_use[1] 
                x3, y3 = points_to_use[2]
                template = random.choice(self.curve_multipoint_templates)
                return template.format(x1=x1, y1=y1, x2=x2, y2=y2, x3=x3, y3=y3)
            else:
                # Fall back to basic curve
                return self.generate_curve_prompt(sample_data, "basic")
        
        elif prompt_type == "shape":
            # Shape-based prompts - randomly choose from all shape types
            all_shape_types = list(self.shape_templates.keys())
            chosen_shape = random.choice(all_shape_types)
            return self.generate_specific_shape_prompt(sample_data, chosen_shape)
        
        elif prompt_type == "pattern":
            # Pattern-based prompts - detect T-value pattern if available
            pattern_type = sample_data.get(
                "pattern_type", "even_spacing"
            )  # Default fallback
            return self.generate_pattern_prompt(sample_data, pattern_type)
        
        elif prompt_type == "general":
            template = random.choice(self.curve_templates)
            return template.format(x1=x1, y1=y1, x2=x2, y2=y2)
        
        else:  # mixed
            # Randomly choose from all curve prompt types
            curve_types = [
                "basic",
                "directional",
                "intensity",
                "multipoint",
                "shape",
                "pattern",
                "general",
            ]
            weights = [
                15,
                15,
                15,
                10,
                15,
                20,
                10,
            ]  # Favor pattern and coordinate-specific prompts
            chosen_type = random.choices(curve_types, weights=weights)[0]
            return self.generate_curve_prompt(sample_data, chosen_type)
    
    def generate_geometric_prompt(
        self,
        sample_data: Dict[str, Any],
        prompt_type: str = "mixed",
        shape_type: str = None,
    ) -> str:
        """Generate a text prompt for geometric shape drawing tasks."""
        
        raw_points = sample_data["raw_points"]
        if (
            len(raw_points) < 8
        ):  # Should have 8 points for geometric shapes (corners duplicated)
            raise ValueError(
                "Geometric shape must have at least 8 points (with corner duplication)"
            )
        
        # Extract the unique corner coordinates (skip duplicates)
        # Following the pattern: start, corner1, corner1, corner2, corner2, corner3, corner3, end
        x1, y1 = raw_points[0]  # Start/bottom-left
        x2, y2 = raw_points[1]  # First corner (bottom-right)
        x3, y3 = raw_points[3]  # Second corner (top-right)
        x4, y4 = raw_points[5]  # Third corner (top-left)
        
        # Get shape name from sample data
        if "shape_type" in sample_data:
            shape_name = sample_data["shape_type"]
        elif shape_type:
            shape_name = shape_type
        else:
            shape_name = "shape"
        
        # Check for corner style information from the shape generator
        corner_style = sample_data.get("shape_info", {}).get("corner_style", None)
        pattern_name = sample_data.get("shape_info", {}).get("pattern", None)
        
        # Determine corner style if not provided
        if not corner_style and pattern_name:
            corner_style = (
                "rounded" if pattern_name == "even_corners" else "sharp_inward"
            )
        
        if prompt_type == "basic":
            # Use corner-style specific templates if available, otherwise basic
            if corner_style == "rounded":
                template = random.choice(self.rounded_corner_templates)
            elif corner_style == "sharp_inward":
                template = random.choice(self.sharp_inward_templates)
            else:
                template = random.choice(self.geometric_basic_templates)
            return template.format(
                shape_name=shape_name,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                x3=x3,
                y3=y3,
                x4=x4,
                y4=y4,
            )
            
        elif prompt_type == "shape_type":
            # Mix of specific shape templates and corner-style templates
            if shape_name in self.geometric_templates:
                # 50% chance to use corner-style description
                if corner_style and random.choice([True, False]):
                    if corner_style == "rounded":
                        template = random.choice(self.rounded_corner_templates)
                    else:
                        template = random.choice(self.sharp_inward_templates)
                    return template.format(
                        shape_name=shape_name,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        x3=x3,
                        y3=y3,
                        x4=x4,
                        y4=y4,
                    )
                else:
                    template = random.choice(self.geometric_templates[shape_name])
                    return template.format(
                        x1=x1, y1=y1, x2=x2, y2=y2, x3=x3, y3=y3, x4=x4, y4=y4
                    )
            else:
                template = random.choice(self.geometric_basic_templates)
                return template.format(
                    shape_name=shape_name,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    x3=x3,
                    y3=y3,
                    x4=x4,
                    y4=y4,
                )
                
        elif prompt_type == "size":
            # Determine size based on area or dimensions
            size = self._determine_geometric_size(raw_points, shape_name)
            template = random.choice(self.geometric_size_templates)
            return template.format(
                size=size,
                shape_name=shape_name,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                x3=x3,
                y3=y3,
                x4=x4,
                y4=y4,
            )
            
        elif prompt_type == "position":
            template = random.choice(self.geometric_position_templates)
            return template.format(
                shape_name=shape_name,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                x3=x3,
                y3=y3,
                x4=x4,
                y4=y4,
            )
            
        elif prompt_type == "general":
            template = random.choice(self.geometric_general_templates)
            return template.format(
                shape_name=shape_name,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                x3=x3,
                y3=y3,
                x4=x4,
                y4=y4,
            )
            
        else:  # mixed or unknown
            # Randomly choose from all template types, with higher weight for corner-style templates
            template_choices = []
            weights = []
            
            # Add basic templates
            template_choices.extend(self.geometric_basic_templates)
            weights.extend([10] * len(self.geometric_basic_templates))
            
            # Add corner-style templates with higher weight
            if corner_style == "rounded":
                template_choices.extend(self.rounded_corner_templates)
                weights.extend([20] * len(self.rounded_corner_templates))
            elif corner_style == "sharp_inward":
                template_choices.extend(self.sharp_inward_templates)
                weights.extend([20] * len(self.sharp_inward_templates))
            
            # Add shape-specific templates
            if shape_name in self.geometric_templates:
                template_choices.extend(self.geometric_templates[shape_name])
                weights.extend([15] * len(self.geometric_templates[shape_name]))
            
            # Add position and general templates
            template_choices.extend(self.geometric_position_templates)
            weights.extend([5] * len(self.geometric_position_templates))
            template_choices.extend(self.geometric_general_templates)
            weights.extend([5] * len(self.geometric_general_templates))
            
            template = random.choices(template_choices, weights=weights)[0]
            
            # Handle template formatting
            if "{shape_name}" in template:
                return template.format(
                    shape_name=shape_name,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    x3=x3,
                    y3=y3,
                    x4=x4,
                    y4=y4,
                )
            else:
                return template.format(
                    x1=x1, y1=y1, x2=x2, y2=y2, x3=x3, y3=y3, x4=x4, y4=y4
                )
    
    def _determine_geometric_size(
        self, raw_points: List[Tuple[int, int]], shape_name: str
    ) -> str:
        """Determine the relative size description for a geometric shape."""
        
        # Calculate approximate dimensions
        x_coords = [point[0] for point in raw_points]
        y_coords = [point[1] for point in raw_points]
        
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        
        # Determine size relative to grid
        grid_ratio = max(width, height) / self.grid_size
        
        if grid_ratio < 0.2:
            return "small"
        elif grid_ratio < 0.4:
            return "medium"
        elif grid_ratio < 0.6:
            return "large"
        else:
            return "very large"
    
    def format_sketch_output(
        self, sample_data: Dict[str, Any], concept: str = None
    ) -> str:
        """Format the expected output in the sketch format."""
        
        if concept is None:
            concept = sample_data["shape_id"].replace("_", " ").title()
        
        sid = sample_data.get('shape_type', sample_data.get('shape_id', 'unknown'))
        
        # Check if this is a multi-stroke shape
        if "strokes" in sample_data and sample_data.get("stroke_count", 0) > 1:
            # Multi-stroke format
            strokes_xml = []
            for i, stroke_data in enumerate(sample_data["strokes"], 1):
                points_str = "'" + "', '".join(stroke_data["formatted_points"]) + "'"
                t_values_str = ",".join([f"{t:.2f}" for t in stroke_data["t_values"]])
                stroke_id = stroke_data.get("stroke_id", f"s{i}")
                
                stroke_xml = f"""    <{stroke_id}>
        <points>{points_str}</points>
        <t_values>{t_values_str}</t_values>
        <id>{sid}</id>
    </{stroke_id}>"""
                strokes_xml.append(stroke_xml)
            
            output = f"""<concept>{concept}</concept>
<strokes>
{chr(10).join(strokes_xml)}
</strokes>"""
        else:
            # Single-stroke format (legacy) - expect combined data
            if "points" not in sample_data or "t_values" not in sample_data:
                raise ValueError(
                    "Single-stroke shapes must have 'points' and 't_values' fields"
                )
                
            points_str = "'" + "', '".join(sample_data["points"]) + "'"
            t_values_str = ",".join([f"{t:.2f}" for t in sample_data["t_values"]])
            
            output = f"""<concept>{concept}</concept>
<strokes>
    <s1>
        <points>{points_str}</points>
        <t_values>{t_values_str}</t_values>
        <id>{sid}</id>
    </s1>
</strokes>"""
        
        return output
    
    def get_available_shape_types(self) -> Dict[str, List[str]]:
        """Get all available shape types organized by category."""
        return {
            "shapes": list(self.shape_templates.keys()),
            "patterns": list(
                self.pattern_specific_templates.keys()
            ),  # Add pattern types
            "geometric": list(self.geometric_templates.keys()),
            "angular": list(self.angular_templates.keys()),  # Add angular types
        }
    
    def get_all_shape_types(self) -> List[str]:
        """Get a flat list of all available shape types."""
        all_types = []
        all_types.extend(self.shape_templates.keys())
        all_types.extend(self.geometric_templates.keys())
        all_types.extend(self.angular_templates.keys())  # Add angular types
        return all_types
    
    def create_training_sample(
        self,
        sample_data: Dict[str, Any],
        prompt_type: str = "mixed",
        shape_type: str = None,
    ) -> Dict[str, str]:
        """Create a complete training sample with prompt and formatted output."""
        
        shape_id = sample_data.get("shape_id", "unknown")
        
        if shape_id == "line":
            prompt = self.generate_line_prompt(sample_data, prompt_type)
        elif shape_id == "curve":
            prompt = self.generate_curve_prompt(sample_data, prompt_type, shape_type)
        elif shape_id == "rectangular":
            prompt = self.generate_geometric_prompt(
                sample_data, prompt_type, shape_type
            )
        elif shape_id == "triangular":
            prompt = self.generate_triangular_prompt(
                sample_data, prompt_type, shape_type
            )
        elif shape_id == "circular":
            prompt = self.generate_circular_prompt(sample_data, prompt_type, shape_type)
        elif shape_id == "angular":
            prompt = self.generate_angular_prompt(sample_data, prompt_type, shape_type)
        elif shape_id == "spatial_relation":
            prompt = self.generate_spatial_relations_prompt(
                sample_data, prompt_type, shape_type
            )
        else:
            raise ValueError(f"Unknown shape_id: {shape_id}")
        
        # Format the sketch output
        sketch_output = self.format_sketch_output(sample_data)
        
        return {"prompt": prompt, "sketch": sketch_output}
    
    def generate_triangular_prompt(
        self,
        sample_data: Dict[str, Any],
        prompt_type: str = "mixed",
        shape_type: str = None,
    ) -> str:
        """Generate a text prompt for triangular shape drawing tasks."""
        
        # Extract corner coordinates from multi-stroke data
        if "strokes" in sample_data and sample_data.get("stroke_count", 0) > 1:
            # Multi-stroke format - extract unique vertices
            vertices = set()
            for stroke_data in sample_data["strokes"]:
                for point in stroke_data["points"]:
                    vertices.add(point)
            vertices = list(vertices)
            
            # Should have exactly 3 unique vertices for a triangle
            if len(vertices) != 3:
                raise ValueError(
                    f"Expected 3 unique vertices for triangle, got {len(vertices)}"
                )
            
            # Use the unique vertices
            (x1, y1), (x2, y2), (x3, y3) = vertices
        else:
            raise ValueError("Triangular shapes must use multi-stroke format")
        
        # Get shape name from sample data
        if "shape_type" in sample_data:
            shape_name = sample_data["shape_type"]
        elif shape_type:
            shape_name = shape_type
        else:
            shape_name = "triangle"
        
        # Check for corner style information from the shape generator
        corner_style = sample_data.get("shape_info", {}).get("corner_style", None)
        pattern_name = sample_data.get("shape_info", {}).get("pattern", None)
        
        # Determine corner style if not provided
        if not corner_style and pattern_name:
            corner_style = (
                "rounded" if pattern_name == "even_corners" else "sharp_inward"
            )
        
        if prompt_type == "basic":
            # Use corner-style specific templates if available, otherwise basic
            if corner_style == "rounded":
                template = random.choice(self.triangular_rounded_corner_templates)
            elif corner_style == "sharp_inward":
                template = random.choice(self.triangular_sharp_inward_templates)
            else:
                template = random.choice(self.triangular_basic_templates)
            return template.format(
                shape_name=shape_name, x1=x1, y1=y1, x2=x2, y2=y2, x3=x3, y3=y3
            )
            
        elif prompt_type == "shape_type":
            # Mix of specific shape templates and corner-style templates
            if shape_name in self.triangular_templates:
                # 50% chance to use corner-style description
                if corner_style and random.choice([True, False]):
                    if corner_style == "rounded":
                        template = random.choice(
                            self.triangular_rounded_corner_templates
                        )
                    else:
                        template = random.choice(self.triangular_sharp_inward_templates)
                    return template.format(
                        shape_name=shape_name, x1=x1, y1=y1, x2=x2, y2=y2, x3=x3, y3=y3
                    )
                else:
                    template = random.choice(self.triangular_templates[shape_name])
                    return template.format(x1=x1, y1=y1, x2=x2, y2=y2, x3=x3, y3=y3)
            else:
                template = random.choice(self.triangular_basic_templates)
                return template.format(
                    shape_name=shape_name, x1=x1, y1=y1, x2=x2, y2=y2, x3=x3, y3=y3
                )
                
        elif prompt_type == "size":
            # Determine size based on area or dimensions
            size = self._determine_triangular_size(
                [(x1, y1), (x2, y2), (x3, y3)], shape_name
            )
            template = random.choice(self.triangular_size_templates)
            return template.format(
                size=size,
                shape_name=shape_name,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                x3=x3,
                y3=y3,
            )
            
        elif prompt_type == "position":
            template = random.choice(self.triangular_position_templates)
            return template.format(
                shape_name=shape_name, x1=x1, y1=y1, x2=x2, y2=y2, x3=x3, y3=y3
            )
            
        elif prompt_type == "general":
            template = random.choice(self.triangular_general_templates)
            return template.format(
                shape_name=shape_name, x1=x1, y1=y1, x2=x2, y2=y2, x3=x3, y3=y3
            )
            
        else:  # mixed or unknown
            # Randomly choose from all template types, with higher weight for corner-style templates
            template_choices = []
            weights = []
            
            # Add basic templates
            template_choices.extend(self.triangular_basic_templates)
            weights.extend([10] * len(self.triangular_basic_templates))
            
            # Add corner-style templates with higher weight
            if corner_style == "rounded":
                template_choices.extend(self.triangular_rounded_corner_templates)
                weights.extend([20] * len(self.triangular_rounded_corner_templates))
            elif corner_style == "sharp_inward":
                template_choices.extend(self.triangular_sharp_inward_templates)
                weights.extend([20] * len(self.triangular_sharp_inward_templates))
            
            # Add shape-specific templates
            if shape_name in self.triangular_templates:
                template_choices.extend(self.triangular_templates[shape_name])
                weights.extend([15] * len(self.triangular_templates[shape_name]))
            
            # Add position and general templates
            template_choices.extend(self.triangular_position_templates)
            weights.extend([5] * len(self.triangular_position_templates))
            template_choices.extend(self.triangular_general_templates)
            weights.extend([5] * len(self.triangular_general_templates))
            
            template = random.choices(template_choices, weights=weights)[0]
            
            # Handle template formatting
            if "{shape_name}" in template:
                return template.format(
                    shape_name=shape_name, x1=x1, y1=y1, x2=x2, y2=y2, x3=x3, y3=y3
                )
            else:
                return template.format(x1=x1, y1=y1, x2=x2, y2=y2, x3=x3, y3=y3)
    
    def _determine_triangular_size(
        self, raw_points: List[Tuple[int, int]], shape_name: str
    ) -> str:
        """Determine the relative size description for a triangular shape."""
        
        # Calculate approximate dimensions
        x_coords = [point[0] for point in raw_points]
        y_coords = [point[1] for point in raw_points]
        
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        
        # Determine size relative to grid
        grid_ratio = max(width, height) / self.grid_size
        
        if grid_ratio < 0.2:
            return "small"
        elif grid_ratio < 0.4:
            return "medium"
        elif grid_ratio < 0.6:
            return "large"
        else:
            return "very large"
    
    def generate_circular_prompt(
        self,
        sample_data: Dict[str, Any],
        prompt_type: str = "mixed",
        shape_type: str = None,
    ) -> str:
        """Generate a text prompt for circular shape drawing tasks."""
        
        raw_points = sample_data["raw_points"]
        if len(raw_points) < 3:
            raise ValueError("Circular shape must have at least 3 points")
        
        # Extract coordinates for prompt formatting
        start_x, start_y = raw_points[0]  # Starting point
        end_x, end_y = raw_points[-1]  # Should be same as start for closed loop
        
        # Get shape information
        shape_info = sample_data.get("shape_info", {})
        center = shape_info.get("center", (start_x, start_y))
        center_x, center_y = center
        
        # Get shape name from sample data
        if "shape_type" in sample_data:
            shape_name = sample_data["shape_type"]
        elif shape_type:
            shape_name = shape_type
        else:
            shape_name = "circle"
        
        if prompt_type == "basic":
            template = random.choice(self.circular_basic_templates)
            return template.format(
                shape_name=shape_name, 
                center_x=center_x, 
                center_y=center_y,
                start_x=start_x, 
                start_y=start_y,
            )
            
        elif prompt_type == "shape_type":
            if shape_name in self.circular_templates:
                template = random.choice(self.circular_templates[shape_name])
                
                # Handle special formatting for circle templates with radius
                if "radius" in shape_info and "{radius}" in template:
                    return template.format(
                        center_x=center_x, 
                        center_y=center_y,
                        start_x=start_x, 
                        start_y=start_y,
                        radius=shape_info["radius"],
                    )
                else:
                    return template.format(
                        center_x=center_x, 
                        center_y=center_y,
                        start_x=start_x, 
                        start_y=start_y,
                    )
            else:
                template = random.choice(self.circular_basic_templates)
                return template.format(
                    shape_name=shape_name,
                    center_x=center_x, 
                    center_y=center_y,
                    start_x=start_x, 
                    start_y=start_y,
                )
                
        elif prompt_type == "size":
            # Determine size based on radius
            size = self._determine_circular_size(shape_info, shape_name)
            template = random.choice(self.circular_size_templates)
            return template.format(
                size=size, shape_name=shape_name, center_x=center_x, center_y=center_y
            )
            
        elif prompt_type == "position":
            template = random.choice(self.circular_position_templates)
            return template.format(
                shape_name=shape_name, center_x=center_x, center_y=center_y
            )
            
        elif prompt_type == "general":
            template = random.choice(self.circular_general_templates)
            return template.format(shape_name=shape_name)
            
        else:  # mixed or unknown
            # Randomly choose from all template types
            template_choices = []
            weights = []
            
            # Add basic templates
            template_choices.extend(self.circular_basic_templates)
            weights.extend([15] * len(self.circular_basic_templates))
            
            # Add shape-specific templates
            if shape_name in self.circular_templates:
                template_choices.extend(self.circular_templates[shape_name])
                weights.extend([20] * len(self.circular_templates[shape_name]))
            
            # Add other template types
            template_choices.extend(self.circular_size_templates)
            weights.extend([10] * len(self.circular_size_templates))
            template_choices.extend(self.circular_position_templates)
            weights.extend([10] * len(self.circular_position_templates))
            template_choices.extend(self.circular_general_templates)
            weights.extend([5] * len(self.circular_general_templates))
            
            template = random.choices(template_choices, weights=weights)[0]
            
            # Handle template formatting based on template content
            if "{radius}" in template and "radius" in shape_info:
                return template.format(
                    shape_name=shape_name,
                    center_x=center_x, 
                    center_y=center_y,
                    start_x=start_x, 
                    start_y=start_y,
                    radius=shape_info["radius"],
                )
            elif "{radius}" in template:
                # Fallback: use a template without radius if radius is missing
                fallback_templates = [
                    t
                    for t in self.circular_templates.get(
                        shape_name, self.circular_basic_templates
                    )
                    if "{radius}" not in t
                ]
                if fallback_templates:
                    template = random.choice(fallback_templates)
                    return template.format(
                        shape_name=shape_name,
                        center_x=center_x, 
                        center_y=center_y,
                        start_x=start_x, 
                        start_y=start_y,
                    )
                else:
                    # Ultimate fallback
                    return f"Draw a {shape_name} starting at ({start_x}, {start_y})"
            elif "{size}" in template:
                size = self._determine_circular_size(shape_info, shape_name)
                return template.format(
                    size=size,
                    shape_name=shape_name,
                    center_x=center_x, 
                    center_y=center_y,
                )
            elif "{center_x}" in template or "{center_y}" in template:
                return template.format(
                    shape_name=shape_name,
                    center_x=center_x, 
                    center_y=center_y,
                    start_x=start_x, 
                    start_y=start_y,
                )
            else:
                return template.format(shape_name=shape_name)
    
    def _determine_circular_size(
        self, shape_info: Dict[str, Any], shape_name: str
    ) -> str:
        """Determine the relative size description for a circular shape."""
        
        # Get radius or average radius for ellipse
        if "radius" in shape_info:
            radius = shape_info["radius"]
        elif "radius_x" in shape_info and "radius_y" in shape_info:
            radius = (shape_info["radius_x"] + shape_info["radius_y"]) / 2
        else:
            return "medium"
        
        # Determine size relative to grid
        grid_ratio = radius / self.grid_size
        
        if grid_ratio < 0.1:
            return "small"
        elif grid_ratio < 0.2:
            return "medium"
        elif grid_ratio < 0.3:
            return "large"
        else:
            return "very large"
    
    def generate_angular_prompt(
        self,
        sample_data: Dict[str, Any],
        prompt_type: str = "mixed",
        shape_type: str = None,
    ) -> str:
        """Generate a text prompt for angular shape drawing tasks."""
        
        raw_points = sample_data["raw_points"]
        if len(raw_points) < 4:
            raise ValueError(
                "Angular shape must have at least 4 points (including duplicated corners)"
            )
        
        # Extract start and end points (skip duplicated corners)
        x1, y1 = raw_points[0]  # Start point
        x2, y2 = raw_points[-1]  # End point
        
        # Get shape name from sample data
        if "shape_type" in sample_data:
            shape_name = sample_data["shape_type"]
        elif shape_type:
            shape_name = shape_type
        else:
            shape_name = "angular_shape"
        
        # Get shape information
        shape_info = sample_data.get("shape_info", {})
        
        if prompt_type == "basic":
            template = random.choice(self.angular_basic_templates)
            return template.format(shape_name=shape_name, x1=x1, y1=y1, x2=x2, y2=y2)
            
        elif prompt_type == "shape_type":
            if shape_name in self.angular_templates:
                template = random.choice(self.angular_templates[shape_name])
                return template.format(x1=x1, y1=y1, x2=x2, y2=y2)
            else:
                template = random.choice(self.angular_basic_templates)
                return template.format(
                    shape_name=shape_name, x1=x1, y1=y1, x2=x2, y2=y2
                )
                
        elif prompt_type == "size":
            # Determine size based on overall dimensions
            size = self._determine_angular_size(raw_points, shape_name)
            template = random.choice(self.angular_size_templates)
            return template.format(
                size=size, shape_name=shape_name, x1=x1, y1=y1, x2=x2, y2=y2
            )
            
        elif prompt_type == "general":
            template = random.choice(self.angular_general_templates)
            return template.format(shape_name=shape_name, x1=x1, y1=y1, x2=x2, y2=y2)
            
        else:  # mixed or unknown
            # Randomly choose from all template types
            template_choices = []
            weights = []
            
            # Add basic templates
            template_choices.extend(self.angular_basic_templates)
            weights.extend([15] * len(self.angular_basic_templates))
            
            # Add shape-specific templates
            if shape_name in self.angular_templates:
                template_choices.extend(self.angular_templates[shape_name])
                weights.extend([25] * len(self.angular_templates[shape_name]))
            
            # Add other template types
            template_choices.extend(self.angular_size_templates)
            weights.extend([10] * len(self.angular_size_templates))
            template_choices.extend(self.angular_general_templates)
            weights.extend([10] * len(self.angular_general_templates))
            
            template = random.choices(template_choices, weights=weights)[0]
            
            # Handle template formatting based on template content
            if "{size}" in template:
                size = self._determine_angular_size(raw_points, shape_name)
                return template.format(
                    size=size, shape_name=shape_name, x1=x1, y1=y1, x2=x2, y2=y2
                )
            elif "{shape_name}" in template:
                return template.format(
                    shape_name=shape_name, x1=x1, y1=y1, x2=x2, y2=y2
                )
            else:
                return template.format(x1=x1, y1=y1, x2=x2, y2=y2)
    
    def _determine_angular_size(
        self, raw_points: List[Tuple[int, int]], shape_name: str
    ) -> str:
        """Determine the relative size description for an angular shape."""
        
        # Calculate approximate dimensions
        x_coords = [point[0] for point in raw_points]
        y_coords = [point[1] for point in raw_points]
        
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        
        # Determine size relative to grid
        grid_ratio = max(width, height) / self.grid_size
        
        if grid_ratio < 0.2:
            return "small"
        elif grid_ratio < 0.4:
            return "medium"
        elif grid_ratio < 0.6:
            return "large"
        else:
            return "very large"
    
    def generate_spatial_relations_prompt(
        self,
        sample_data: Dict[str, Any],
        prompt_type: str = "mixed",
        shape_type: str = None,
    ) -> str:
        """Generate a text prompt for spatial relations tasks."""
        
        # Get spatial information from sample data
        spatial_info = sample_data.get("spatial_info", {})
        base_shape_type = spatial_info.get("base_shape_type", "shape")
        direction = spatial_info.get("direction", "top")

        # Build the incomplete sketch showing only the base shape
        if "strokes" in sample_data and len(sample_data["strokes"]) > 1:
            # Show only the base shape strokes (all except the last one which is the added stroke)
            base_strokes = sample_data["strokes"][:-1]

            # Get the stroke type from spatial_info (reliable source)
            stroke_type = spatial_info.get("connected_stroke_type", "line")

            # Format the base strokes as XML
            base_strokes_xml = []
            for i, stroke_data in enumerate(base_strokes, 1):
                points_str = "'" + "', '".join(stroke_data["formatted_points"]) + "'"
                t_values_str = ",".join([f"{t:.2f}" for t in stroke_data["t_values"]])

                stroke_xml = f"""    <s{i}>
        <points>{points_str}</points>
        <t_values>{t_values_str}</t_values>
        <id>{base_shape_type} base</id>
    </s{i}>"""
                base_strokes_xml.append(stroke_xml)

            # Get descriptions
            direction_desc = self.direction_descriptions.get(
                direction, direction.replace("_", " ")
            )
            stroke_desc = self.stroke_descriptions.get(stroke_type, "stroke")

            # Build the complete prompt
            if direction == "inside":
                incomplete_sketch = f"""Given an incomplete sketch:
<strokes>
{chr(10).join(base_strokes_xml)}
</strokes>
Add a {stroke_desc} inside the {base_shape_type}."""
            else:
                incomplete_sketch = f"""Given an incomplete sketch:
<strokes>
{chr(10).join(base_strokes_xml)}
</strokes>
Add a {stroke_desc} {direction_desc} of the {base_shape_type} base."""

            return incomplete_sketch

        # Fallback for single stroke base shapes (shouldn't happen in spatial relations)
        direction_desc = self.direction_descriptions.get(
            direction, direction.replace("_", " ")
        )
        return f"Add a stroke {direction_desc} from the shape."
    
    def get_spatial_relations_available_shape_types(self) -> Dict[str, List[str]]:
        """Get all available shape types for spatial relations tasks."""
        return {
            "stroke_types": list(self.stroke_descriptions.keys()),
            "directions": list(self.direction_descriptions.keys()),
            "base_shapes": ["rectangle", "triangle", "circle"],
        }
    
    def get_spatial_relations_all_shape_types(self) -> List[str]:
        """Get a flat list of all available shape types for spatial relations tasks."""
        all_types = []
        all_types.extend(self.stroke_descriptions.keys())
        all_types.extend(self.direction_descriptions.keys())
        return all_types
    
    def create_spatial_relations_training_sample(
        self,
        sample_data: Dict[str, Any],
        prompt_type: str = "mixed",
        shape_type: str = None,
    ) -> Dict[str, str]:
        """Create a complete training sample with prompt and formatted output for spatial relations tasks."""
        
        shape_id = sample_data.get("shape_id", "unknown")
        
        if shape_id == "spatial_relation":
            prompt = self.generate_spatial_relations_prompt(
                sample_data, prompt_type, shape_type
            )
        else:
            raise ValueError(f"Unknown shape_id: {shape_id}")
        
        # Format the sketch output
        sketch_output = self.format_sketch_output(sample_data)
        
        return {"prompt": prompt, "sketch": sketch_output}
