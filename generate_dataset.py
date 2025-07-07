#!/usr/bin/env python3
"""
Dataset Generation Script for LoRA Training Data

Generates N=5 samples for each shape type available in the prompt visualizer,
saves them in JSON format with 'prompt' and 'answer' fields, and creates
visualizations in a separate directory.
"""

import json
import random
import time
from pathlib import Path
from tqdm import tqdm
from itertools import product

from curves import Curve
from sample_generation.rectangular_shapes import RectangularShape
from triangular_shapes import TriangularShape
from circular_shapes import CircularShape
from angular_shapes import AngularShape
from spatial_relations import SpatialRelationsGenerator
from prompt_generator import PromptGenerator

# Import utils for SVG rendering
import SketchAgent.utils as utils
import ast

# Configuration
N_SAMPLES = 200
SPATIAL_RELATIONS_REPEATS = 10
GRID_SIZE = 100
CELL_SIZE = 6
OUTPUT_DIR = Path("dataset_output")
VISUALIZATION_DIR = OUTPUT_DIR / "visualizations"


def setup_directories():
    """Create output directories."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    VISUALIZATION_DIR.mkdir(exist_ok=True)


def create_sketch_xml(sample_data):
    """Convert sample data to XML format for utils processing."""

    if "strokes" in sample_data and sample_data.get("stroke_count", 0) > 1:
        # Multi-stroke format
        strokes_xml = []
        for i, stroke_data in enumerate(sample_data["strokes"], 1):
            points_str = "'" + "', '".join(stroke_data["formatted_points"]) + "'"
            t_values_str = ",".join([f"{t:.2f}" for t in stroke_data["t_values"]])
            stroke_id = stroke_data.get("stroke_id", f"s{i}")
            if "spatial_info" in sample_data:
                sid = (
                    sample_data["spatial_info"]["connected_stroke_type"]
                    if i > sample_data["spatial_info"]["base_stroke_count"]
                    else f'{sample_data["spatial_info"]["base_shape_type"]} base'
                )
            else:
                sid = sample_data.get(
                    "shape_type", sample_data.get("shape_id", "unknown")
                )
            stroke_xml = f"""    <{stroke_id}>
        <points>{points_str}</points>
        <t_values>{t_values_str}</t_values>
        <id>{sid}</id>
    </{stroke_id}>"""
            strokes_xml.append(stroke_xml)

        xml_content = f"""<strokes>
{chr(10).join(strokes_xml)}
</strokes>"""
    else:
        # Single-stroke format
        points_str = "'" + "', '".join(sample_data["points"]) + "'"
        t_values_str = ",".join([f"{t:.2f}" for t in sample_data["t_values"]])
        shape_id = sample_data.get("shape_type", sample_data.get("shape_id", "unknown"))

        xml_content = f"""<strokes>
    <s1>
        <points>{points_str}</points>
        <t_values>{t_values_str}</t_values>
        <id>{shape_id}</id>
    </s1>
</strokes>"""

    return xml_content


def format_svg_with_colors(all_control_points, dim, stroke_width, sample_data=None):
    """Enhanced SVG formatting with color support for spatial relations."""
    svg_width, svg_height = dim
    sketch_text_svg = f"""<svg width="{svg_width}" height="{svg_height}" xmlns="http://www.w3.org/2000/svg">\n"""

    # Check if this is a spatial relations sample
    is_spatial_relations = (
        sample_data
        and sample_data.get("shape_type", "").endswith("_stroke")
        and "spatial_info" in sample_data
    )

    for i, path in enumerate(all_control_points):
        # Determine stroke color
        if is_spatial_relations:
            # For spatial relations: first stroke(s) are base shape (black), last stroke is added stroke (red)
            spatial_info = sample_data.get("spatial_info", {})
            base_shape_type = spatial_info.get("base_shape_type", "")

            # Determine number of base shape strokes
            base_stroke_count = 1  # Default for most shapes
            if base_shape_type == "triangle":
                base_stroke_count = 2  # Triangles have 2 strokes

            if i < base_stroke_count:
                stroke_color = "black"  # Base shape
            else:
                stroke_color = "red"  # Added stroke
        else:
            stroke_color = "black"  # Default color for non-spatial relations

        group_text = f"""<g id="s{i + 1}" stroke="{stroke_color}" stroke-width="{stroke_width}" fill="none" stroke-linecap="round">\n"""
        for sub_path_cp in path:
            path_data = utils.create_svg_path_data(sub_path_cp)
            group_text += f"""<path d="{path_data}"/>\n"""
        group_text += "</g>\n"
        sketch_text_svg += group_text

    sketch_text_svg += "</svg>"
    return sketch_text_svg


def render_to_svg(xml_content, filename):
    """Render sample to SVG file."""
    try:
        # Create XML from sample data
        # xml_content = create_sketch_xml(sample_data)

        # Parse with utils
        strokes_list_str, t_values_str = utils.parse_xml_string(xml_content, GRID_SIZE)
        strokes_list, t_values = ast.literal_eval(strokes_list_str), ast.literal_eval(
            t_values_str
        )

        # Create cells to pixels mapping
        grid_size = (GRID_SIZE + 1) * CELL_SIZE
        cells_to_pixels_map = utils.cells_to_pixels(
            GRID_SIZE, CELL_SIZE, header_size=CELL_SIZE
        )

        # Get control points
        all_control_points = utils.get_control_points(
            strokes_list, t_values, cells_to_pixels_map
        )

        # Generate SVG with colors
        svg_content = format_svg_with_colors(
            all_control_points, (grid_size, grid_size), 3.0
        )

        # Save SVG
        svg_path = VISUALIZATION_DIR / f"{filename}.svg"
        with open(svg_path, "w") as f:
            f.write(svg_content)

    except Exception as e:
        print(f"Error rendering {filename}: {e}")


def generate_spatial_relations(generator, prompt_gen, n_samples):
    """Generate spatial relations samples."""
    samples = []
    configs = list(
        product(
            generator.generators.keys(),
            generator.DIRECTIONS.keys(),
            generator.stroke_types,
        )
    )
    
    for j in tqdm(range(SPATIAL_RELATIONS_REPEATS), desc="Generating spatial relations"):
        for i, (base_shape_type, direction, stroke_type) in tqdm(enumerate(configs), total=len(configs), desc="Spatial relations step"):            
            # Generate sample
            sample_data = generator.generate_sample(
                base_shape_type=base_shape_type,
                direction=direction,
                stroke_type=stroke_type,
            )

            # Generate prompt
            training_sample = prompt_gen.create_training_sample(sample_data, "mixed")

            # Create output format
            answer_xml = create_sketch_xml(sample_data)

            sample = {"prompt": training_sample["prompt"], "answer": answer_xml}

            samples.append(sample)
        
    return samples


def generate_specific_shapes(shape_type, generator, prompt_gen, n_samples):
    """Generate samples for specific geometric shapes."""
    samples = []

    for i in range(n_samples):
        if shape_type in ["square", "rectangle", "rhombus"]:
            # Rectangular shapes
            points, t_values = generator.generate_shape_type(shape_type)
            if random.choices([True, False], weights=[0.5, 0.5])[0]:
                points, t_values, transformations = (
                    generator.apply_random_transformations(points, t_values)
                )
            else:
                transformations = {
                    "scale": 1.0,
                    "shift_x": 0,
                    "shift_y": 0,
                    "rotation_deg": 0.0,
                }
            sample_data = {
                "points": generator.format_points_for_prompt(points),
                "t_values": t_values,
                "shape_id": "rectangular",
                "shape_type": shape_type,
                "transformations": transformations,
                "raw_points": points,
            }
            training_sample = prompt_gen.create_training_sample(
                sample_data, "shape_type", shape_type
            )

        elif shape_type in ["equilateral", "isosceles", "right", "scalene"]:
            # Triangular shapes
            generator.clear_strokes()
            generator._generate_triangular_shape(shape_type)
            sample_data = {
                "shape_id": "triangular",
                "shape_type": shape_type,
                "transformations": {
                    "scale": 1.0,
                    "shift_x": 0,
                    "shift_y": 0,
                    "rotation_deg": 0.0,
                },
                "shape_info": {},
                "strokes": [
                    {
                        "points": stroke.points,
                        "t_values": stroke.t_values,
                        "formatted_points": stroke.formatted_points,
                        "stroke_id": stroke.stroke_id,
                    }
                    for stroke in generator.strokes
                ],
                "stroke_count": generator.get_stroke_count(),
            }
            training_sample = prompt_gen.create_training_sample(
                sample_data, "shape_type", shape_type
            )

        elif shape_type in ["circle", "ellipse"]:
            # Circular shapes
            points, t_values = generator.generate_shape_type(shape_type)
            shape_info = generator.get_last_shape_info()
            sample_data = {
                "points": generator.format_points_for_prompt(points),
                "t_values": t_values,
                "shape_id": "circular",
                "shape_type": shape_type,
                "transformations": {
                    "scale": 1.0,
                    "shift_x": 0,
                    "shift_y": 0,
                    "rotation_deg": 0.0,
                },
                "raw_points": points,
                "shape_info": shape_info,
            }
            training_sample = prompt_gen.create_training_sample(
                sample_data, "shape_type", shape_type
            )

        elif shape_type in ["v_shape", "l_shape", "zigzag", "step"]:
            # Angular shapes
            points, t_values = generator.generate_shape_type(shape_type)
            sample_data = {
                "points": generator.format_points_for_prompt(points),
                "t_values": t_values,
                "shape_id": "angular",
                "shape_type": shape_type,
                "transformations": {
                    "scale": 1.0,
                    "shift_x": 0,
                    "shift_y": 0,
                    "rotation_deg": 0.0,
                },
                "raw_points": points,
                "shape_info": {"type": shape_type},
            }
            training_sample = prompt_gen.create_training_sample(
                sample_data, "shape_type", shape_type
            )

        elif shape_type in ["arc_shape", "curve"]:
            # Curve shapes
            points, t_values = generator.generate_shape_curve(shape_type)
            sample_data = {
                "points": generator.format_points_for_prompt(points),
                "t_values": t_values,
                "shape_id": "curve",
                "transformations": {
                    "scale": 1.0,
                    "shift_x": 0,
                    "shift_y": 0,
                    "rotation_deg": 0.0,
                },
                "raw_points": points,
            }
            training_sample = prompt_gen.create_training_sample(
                sample_data, "shape", shape_type
            )

        # Create output format
        answer_xml = create_sketch_xml(sample_data)

        sample = {"prompt": training_sample["prompt"], "answer": answer_xml}

        samples.append(sample)

    return samples


def main():
    """Main generation function."""
    print("🎨 Starting Dataset Generation...")

    # Setup
    setup_directories()

    # Initialize generators
    rectangular_gen = RectangularShape(grid_size=GRID_SIZE)
    triangular_gen = TriangularShape(grid_size=GRID_SIZE)
    circular_gen = CircularShape(grid_size=GRID_SIZE)
    angular_gen = AngularShape(grid_size=GRID_SIZE)
    curve_gen = Curve(grid_size=GRID_SIZE)
    spatial_gen = SpatialRelationsGenerator(grid_size=GRID_SIZE)
    prompt_gen = PromptGenerator(grid_size=GRID_SIZE)

    # Define all shape types from the HTML template
    shape_configs = [
        # Spatial Relations
        ("spatial_relations", spatial_gen, None),
        # Curve shapes
        ("arc_shape", curve_gen, "curve"),
        ("curve", curve_gen, "curve"),
        # Rectangular shapes
        ("square", rectangular_gen, "rectangular"),
        ("rectangle", rectangular_gen, "rectangular"),
        ("rhombus", rectangular_gen, "rectangular"),
        # Triangular shapes
        ("equilateral", triangular_gen, "triangular"),
        ("isosceles", triangular_gen, "triangular"),
        ("right", triangular_gen, "triangular"),
        ("scalene", triangular_gen, "triangular"),
        # Circular shapes
        ("circle", circular_gen, "circular"),
        ("ellipse", circular_gen, "circular"),
        # Angular shapes
        ("v_shape", angular_gen, "angular"),
        ("l_shape", angular_gen, "angular"),
        ("zigzag", angular_gen, "angular"),
        ("step", angular_gen, "angular"),
    ]

    all_samples = []

    # Generate samples for each shape type
    for shape_type, generator, category in tqdm(
        shape_configs, desc="Generating shape types"
    ):
        print(f"\nGenerating {N_SAMPLES} samples for {shape_type}...")

        # Set random seed for reproducibility within each shape type
        random.seed(int(time.time() * 1000000) % 2147483647 + hash(shape_type))

        if shape_type == "spatial_relations":
            samples = generate_spatial_relations(generator, prompt_gen, N_SAMPLES)
        else:
            samples = generate_specific_shapes(
                shape_type, generator, prompt_gen, N_SAMPLES
            )

        # Add metadata to samples
        for sample in samples:
            sample["shape_type"] = shape_type
            sample["category"] = category

        all_samples.extend(samples)
        print(f"✓ Generated {len(samples)} {shape_type} samples")

    for i, sample in enumerate(all_samples):
        render_to_svg(sample['answer'], f"{i:03d}-{sample['shape_type']}")

    # Save dataset
    output_file = OUTPUT_DIR / "lora_training_dataset.json"
    with open(output_file, "w") as f:
        json.dump(all_samples, f, indent=2)

    print(f"\n🎉 Dataset generation complete!")
    print(f"📊 Total samples: {len(all_samples)}")
    print(f"💾 Saved to: {output_file}")
    print(f"🖼️  Visualizations saved to: {VISUALIZATION_DIR}")
    print(f"📁 Shape types: {len(shape_configs)}")


if __name__ == "__main__":
    main()
#``
