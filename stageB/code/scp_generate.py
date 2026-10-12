"""
scp_generate.py

Generate the Synthetic Counting Probe (SCP): test images placing a known
number, position, and type of shapes on a clutter background. Each shape
has an independent random size and rotation. Outputs PNG images plus
per-image ground-truth JSON files.

Usage:
    python3 -I stageB/code/scp_generate.py
    python3 -I stageB/code/scp_generate.py --max_shapes 60 --clutter_lines 80 --clutter_grid
    python3 -I stageB/code/scp_generate.py --clutter_lines 0     # no clutter background
SCP-dense (the 50 images of the paper):
    python3 -I stageB/code/scp_generate.py --num_images 50 --min_shapes 120 --max_shapes 200 --min_radius 9 --max_radius 16 --seed 2026
"""

import argparse
import json
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw

# Available shape types and outline-fill colors
SHAPE_TYPES = ["circle", "triangle", "square", "star", "pentagon"]
COLORS = [
    "#E74C3C", "#2ECC71", "#3498DB", "#F39C12",
    "#9B59B6", "#1ABC9C", "#E67E22", "#2C3E50",
]

# Clutter background colors (mid-gray, close to shape outline value)
CLUTTER_COLORS = ["#606060", "#707070", "#808080", "#686868"]

OUT_DIR = Path("test_images")
GT_DIR = Path("ground_truth")


# ─── Shape rendering (with rotation) ──────────────────────────────

def _rotated_polygon(cx, cy, r, n_sides, rotation_deg, start_angle_deg=-90):
    """Return vertex list for a regular polygon with optional rotation."""
    pts = []
    rotation_rad = math.radians(rotation_deg)
    for i in range(n_sides):
        angle = math.radians(i * (360 / n_sides) + start_angle_deg) + rotation_rad
        pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    return pts


def draw_circle(draw, cx, cy, r, color, rotation=0):
    # Rotation is a no-op for circles
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill="white", outline="black", width=3)


def draw_square(draw, cx, cy, r, color, rotation=0):
    # r is the circumscribed-circle radius, so side length = r * sqrt(2)
    pts = _rotated_polygon(cx, cy, r, 4, rotation, start_angle_deg=-45)
    draw.polygon(pts, fill="white", outline="black", width=3)


def draw_triangle(draw, cx, cy, r, color, rotation=0):
    pts = _rotated_polygon(cx, cy, r, 3, rotation)
    draw.polygon(pts, fill="white", outline="black", width=3)


def draw_pentagon(draw, cx, cy, r, color, rotation=0):
    pts = _rotated_polygon(cx, cy, r, 5, rotation)
    draw.polygon(pts, fill="white", outline="black", width=3)


def draw_star(draw, cx, cy, r, color, rotation=0):
    """Five-pointed star with alternating outer/inner vertices."""
    pts = []
    rotation_rad = math.radians(rotation)
    for i in range(10):
        angle = math.radians(i * 36 - 90) + rotation_rad
        rad = r if i % 2 == 0 else r * 0.45
        pts.append((cx + rad * math.cos(angle), cy + rad * math.sin(angle)))
    draw.polygon(pts, fill="white", outline="black", width=3)


DRAW_FN = {
    "circle": draw_circle,
    "triangle": draw_triangle,
    "square": draw_square,
    "star": draw_star,
    "pentagon": draw_pentagon,
}


# ─── Clutter background ──────────────────────────────────────────

def add_clutter_background(draw, img_size, n_lines=40, n_vh_lines=20,
                            draw_grid=False, line_width_range=(1, 2)):
    """
    Render light-gray clutter to simulate engineering-drawing line work:
      - n_lines: random-angle straight lines
      - n_vh_lines: strict vertical/horizontal lines (reference / axis grid lines)
      - draw_grid: optional regular building-style grid
    All clutter elements use mid-gray colors so they do not blend with shape outlines.
    """
    # 1. Random-angle lines (length 150-800 px)
    for _ in range(n_lines):
        color = random.choice(CLUTTER_COLORS)
        width = random.randint(*line_width_range)
        x0 = random.randint(0, img_size)
        y0 = random.randint(0, img_size)
        length = random.randint(150, 800)
        angle = random.uniform(0, 2 * math.pi)
        x1 = int(x0 + length * math.cos(angle))
        y1 = int(y0 + length * math.sin(angle))
        draw.line([(x0, y0), (x1, y1)], fill=color, width=width)

    # 2. Strict vertical/horizontal lines (axis / wall / reference lines)
    for _ in range(n_vh_lines):
        color = random.choice(CLUTTER_COLORS)
        width = random.randint(*line_width_range)
        is_vertical = random.random() < 0.5
        if is_vertical:
            x = random.randint(0, img_size)
            y0 = random.randint(0, img_size)
            length = random.randint(200, 800)
            y1 = max(0, min(img_size, y0 + length * random.choice([-1, 1])))
            draw.line([(x, y0), (x, y1)], fill=color, width=width)
        else:
            y = random.randint(0, img_size)
            x0 = random.randint(0, img_size)
            length = random.randint(200, 800)
            x1 = max(0, min(img_size, x0 + length * random.choice([-1, 1])))
            draw.line([(x0, y), (x1, y)], fill=color, width=width)

    # 3. Optional regular grid (common in architectural drawings)
    if draw_grid:
        grid_color = random.choice(CLUTTER_COLORS)
        grid_spacing = random.randint(80, 150)
        offset_x = random.randint(0, grid_spacing)
        offset_y = random.randint(0, grid_spacing)
        for x in range(offset_x, img_size, grid_spacing):
            draw.line([(x, 0), (x, img_size)], fill=grid_color, width=1)
        for y in range(offset_y, img_size, grid_spacing):
            draw.line([(0, y), (img_size, y)], fill=grid_color, width=1)


# ─── Overlap check ───────────────────────────────────────────────

def shapes_overlap(new_shape, existing_shapes, min_gap=5):
    """Bounding-circle overlap test with a minimum-gap pixel margin."""
    nx, ny, nr = new_shape["cx"], new_shape["cy"], new_shape["radius"]
    for s in existing_shapes:
        dist = ((nx - s["cx"]) ** 2 + (ny - s["cy"]) ** 2) ** 0.5
        if dist < nr + s["radius"] + min_gap:
            return True
    return False


# ─── Image generation ────────────────────────────────────────────

def generate_one_image(img_id, img_size, min_shapes, max_shapes,
                       min_radius, max_radius,
                       n_clutter_lines, n_vh_lines, draw_grid,
                       shape_types=None):
    """
    Generate one test image and return its ground-truth dict.
    Each shape instance has an independent size and rotation.
    """
    if shape_types is None:
        shape_types = SHAPE_TYPES

    num_shapes = random.randint(min_shapes, max_shapes)
    img = Image.new("RGB", (img_size, img_size), "white")
    draw = ImageDraw.Draw(img)

    # Step 1: place target shapes first
    margin = 40

    shapes = []
    attempts = 0
    max_attempts = num_shapes * 80

    while len(shapes) < num_shapes and attempts < max_attempts:
        attempts += 1
        stype = random.choice(shape_types)
        # Per-instance independent radius
        r = random.randint(min_radius, max_radius)
        cx = random.randint(margin + r, img_size - margin - r)
        cy = random.randint(margin + r, img_size - margin - r)
        color = random.choice(COLORS)
        # Per-instance independent rotation (stored even for circles, where it is unused)
        rotation = random.uniform(0, 360)

        candidate = {
            "cx": cx, "cy": cy, "radius": r,
            "type": stype, "color": color, "rotation": rotation,
        }
        if not shapes_overlap(candidate, shapes):
            shapes.append(candidate)
            DRAW_FN[stype](draw, cx, cy, r, color, rotation=rotation)

    # Step 2: render clutter background on top (cuts through the white shape fills)
    add_clutter_background(draw, img_size,
                           n_lines=n_clutter_lines,
                           n_vh_lines=n_vh_lines,
                           draw_grid=draw_grid)

    img_path = OUT_DIR / f"test_{img_id:03d}.png"
    img.save(str(img_path), "PNG")

    gt = {
        "image_id": img_id,
        "image_file": str(img_path),
        "image_size": img_size,
        "total_shapes": len(shapes),
        "n_clutter_lines": n_clutter_lines,
        "n_vh_lines": n_vh_lines,
        "has_grid_clutter": draw_grid,
        "counts_by_type": {},
        "shapes": [],
    }
    for s in shapes:
        gt["shapes"].append({
            "type": s["type"],
            "cx": s["cx"],
            "cy": s["cy"],
            "radius": s["radius"],
            "color": s["color"],
            "rotation": round(s["rotation"], 1),
        })
        gt["counts_by_type"][s["type"]] = gt["counts_by_type"].get(s["type"], 0) + 1

    gt_path = GT_DIR / f"gt_{img_id:03d}.json"
    with open(gt_path, "w") as f:
        json.dump(gt, f, indent=2)

    return gt


def main():
    parser = argparse.ArgumentParser(
        description="Generate test images with random shapes on a clutter background.")
    parser.add_argument("--num_images", type=int, default=100, help="number of images to generate")
    parser.add_argument("--min_shapes", type=int, default=15, help="minimum shapes per image")
    parser.add_argument("--max_shapes", type=int, default=80, help="maximum shapes per image")
    parser.add_argument("--img_size", type=int, default=1024, help="image edge length (px)")
    parser.add_argument("--min_radius", type=int, default=15, help="minimum shape radius (px)")
    parser.add_argument("--max_radius", type=int, default=30, help="maximum shape radius (px)")

    parser.add_argument("--clutter_lines", type=int, default=60,
                        help="number of random-angle clutter lines (0 = none)")
    parser.add_argument("--vh_lines", type=int, default=40,
                        help="number of strict vertical/horizontal clutter lines (0 = none)")
    parser.add_argument("--clutter_grid", action="store_true",
                        help="draw architectural-style grid in background")
    parser.add_argument("--clutter_grid_prob", type=float, default=0.6,
                        help="per-image probability of drawing the grid (only if --clutter_grid set)")

    parser.add_argument(
        "--shape_types", nargs="+", default=None, choices=SHAPE_TYPES,
        help="restrict to specific shape types (default: all)",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    OUT_DIR.mkdir(exist_ok=True)
    GT_DIR.mkdir(exist_ok=True)

    summary = []
    for i in range(args.num_images):
        # Per-image random decision for grid (only when --clutter_grid is enabled)
        has_grid = args.clutter_grid and random.random() < args.clutter_grid_prob

        gt = generate_one_image(
            img_id=i,
            img_size=args.img_size,
            min_shapes=args.min_shapes,
            max_shapes=args.max_shapes,
            min_radius=args.min_radius,
            max_radius=args.max_radius,
            n_clutter_lines=args.clutter_lines,
            n_vh_lines=args.vh_lines,
            draw_grid=has_grid,
            shape_types=args.shape_types,
        )
        summary.append({
            "image_id": i,
            "total_shapes": gt["total_shapes"],
            "counts_by_type": gt["counts_by_type"],
            "n_clutter_lines": gt["n_clutter_lines"],
            "n_vh_lines": gt["n_vh_lines"],
            "has_grid_clutter": gt["has_grid_clutter"],
        })
        grid_marker = " [+grid]" if has_grid else ""
        print(f"  [done] test_{i:03d}.png - {gt['total_shapes']} shapes, "
              f"{args.clutter_lines} angled + {args.vh_lines} v/h lines{grid_marker}")

    with open(GT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nGenerated {args.num_images} images in {OUT_DIR}/")
    print(f"Ground truth in {GT_DIR}/")


if __name__ == "__main__":
    main()
