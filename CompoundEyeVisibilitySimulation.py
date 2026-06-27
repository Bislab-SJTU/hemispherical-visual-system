"""
This script implements the core model used in the variable-FOV compound-eye
experiments. 

Two experiments are supported:
1. fixed-angle: one incidence angle, multiple source-center distances.
Example:
    python CompoundEyeVisibilitySimulation.py fixed-angle \
    --incidence-angle 40 \
    --distance-range 20 120 2 \
    --output-dir output_fixed_angle_40

2. fixed-distance: one source-center distance, multiple incidence angles.
Example:
    python CompoundEyeVisibilitySimulation.py fixed-distance \
    --distance 60 \
    --angle-range 0 90 10 \
    --output-dir output_fixed_distance_60

Incidence angle convention:
- 0 deg: the source is above the compound eye along +Z.
- 90 deg: the source is lateral to the compound eye along +X.
- The source is constrained to the XZ plane and always points to the origin.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SimulationConfig:
    """Numerical and optical parameters for the simulation."""

    shape: tuple[int, int] = (601, 601)
    sphere_radius_mm: float = 6.4
    image_sensor_radius_mm: float = 0.01

    center_fov_deg: float = 22.0
    edge_fov_deg: float = 40.0
    fov_blend: float = 1.0
    fov_weight_gamma: float = 0.273

    source_intensity: float = 20000.0
    source_diameter_mm: float = 0.2
    source_divergence_deg: float = 180.0

    pixel_gap_to_diameter_ratio: float = 1.0
    pixel_area_sample_resolution: int = 21
    chunk_size: int = 50000

    output_dir: Path = Path("compound_eye_visibility_output")


@dataclass(frozen=True)
class EyeLayout:
    """Array representation of all valid ommatidia on the hemispherical eye."""

    rows: np.ndarray
    cols: np.ndarray
    surface_positions: np.ndarray
    directions: np.ndarray
    apex_positions: np.ndarray
    tan_half_fov: np.ndarray
    pixel_size_coeff: np.ndarray
    reference_distance_mm: float
    image_shape: tuple[int, int]

    @property
    def eye_count(self) -> int:
        return int(self.rows.shape[0])


@dataclass(frozen=True)
class SourceState:
    """Position, orientation, and size of the spherical light source."""

    position: np.ndarray
    axis: np.ndarray
    radius_mm: float
    cos_half_divergence: float


@dataclass(frozen=True)
class FrameStats:
    """Visibility statistics for one rendered frame."""

    fov_overlap_count: int
    geometric_visible_count: int
    responsive_pixel_count: int


def inclusive_float_range(start: float, end: float, step: float) -> list[float]:
    """Return an inclusive floating-point sequence."""

    if step == 0.0:
        raise ValueError("step must be non-zero.")
    if start < end and step < 0.0:
        raise ValueError("step must be positive for an increasing range.")
    if start > end and step > 0.0:
        raise ValueError("step must be negative for a decreasing range.")

    values: list[float] = []
    current = float(start)
    tolerance = abs(step) * 1e-6 + 1e-9
    if step > 0.0:
        while current <= end + tolerance:
            values.append(round(current, 6))
            current += step
    else:
        while current >= end - tolerance:
            values.append(round(current, 6))
            current += step
    return values


def compute_variable_fov(
    z_values: np.ndarray,
    center_fov_deg: float,
    edge_fov_deg: float,
    blend: float,
    weight_gamma: float,
    z_floor_mm: float = 0.5,
) -> np.ndarray:
    """
    Assign a field of view to each ommatidium using a smoothed inverse-z law.

    Larger z values are closer to the top of the hemisphere and receive a
    smaller FOV. Smaller z values are closer to the rim and receive a larger
    FOV. The blend parameter interpolates between uniform and variable FOV.
    """

    blend = float(np.clip(blend, 0.0, 1.0))
    weight_gamma = max(float(weight_gamma), 1e-6)

    base = np.full_like(z_values, fill_value=center_fov_deg, dtype=np.float64)
    if blend == 0.0:
        return base

    z_eff = np.maximum(z_values.astype(np.float64), z_floor_mm)
    inv_z = 1.0 / z_eff
    inv_min = float(np.min(inv_z))
    inv_max = float(np.max(inv_z))

    if math.isclose(inv_min, inv_max):
        variable = base
    else:
        weight = (inv_z - inv_min) / (inv_max - inv_min)
        weight = np.power(np.clip(weight, 0.0, 1.0), weight_gamma)
        variable = center_fov_deg + (edge_fov_deg - center_fov_deg) * weight

    return base * (1.0 - blend) + variable * blend


def compute_cylinder_cut_pixel_size_coeff(
    surface_positions: np.ndarray,
    sphere_radius_mm: float,
    image_shape: tuple[int, int],
    gap_to_diameter_ratio: float,
    sample_resolution: int,
) -> np.ndarray:
    """
    Estimate relative pixel size using a cylinder-cut geometry.

    Each pixel is modeled as a vertical cylinder on a square lattice. The
    hemispherical shell cuts the cylinder, and the resulting surface patch area
    is estimated by numerical integration. The coefficient is normalized by the
    central pixel area.
    """

    gap_to_diameter_ratio = max(float(gap_to_diameter_ratio), 0.0)
    sample_resolution = max(int(sample_resolution), 5)

    height, width = image_shape
    half_h = (height - 1) / 2.0
    half_w = (width - 1) / 2.0
    pitch_y = sphere_radius_mm / half_h
    pitch_x = sphere_radius_mm / half_w
    pitch = 0.5 * (pitch_x + pitch_y)

    diameter = pitch / (1.0 + gap_to_diameter_ratio)
    cylinder_radius = diameter / 2.0
    base_disk_area = math.pi * cylinder_radius * cylinder_radius

    sample_axis = np.linspace(-1.0, 1.0, sample_resolution, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(sample_axis, sample_axis, indexing="xy")
    disk_mask = (grid_x * grid_x + grid_y * grid_y) <= 1.0
    offsets = np.stack([grid_x[disk_mask], grid_y[disk_mask]], axis=1)
    offsets = offsets.astype(np.float64) * cylinder_radius
    area_per_sample = base_disk_area / float(offsets.shape[0])

    xy_centers = surface_positions[:, :2].astype(np.float64)
    sample_x = xy_centers[:, None, 0] + offsets[None, :, 0]
    sample_y = xy_centers[:, None, 1] + offsets[None, :, 1]
    radial_sq = sample_x * sample_x + sample_y * sample_y

    radius_sq = sphere_radius_mm * sphere_radius_mm
    inside = radial_sq < radius_sq
    safe_gap = np.maximum(radius_sq - radial_sq, 1e-12)
    z_values = np.sqrt(safe_gap)

    area_density = np.where(inside, sphere_radius_mm / z_values, 0.0)
    patch_areas = area_per_sample * np.sum(area_density, axis=1)

    center_index = int(np.argmax(surface_positions[:, 2]))
    center_area = max(float(patch_areas[center_index]), 1e-12)
    return patch_areas / center_area


def reference_source_position(config: SimulationConfig) -> np.ndarray:
    """
    Return the reference source position used for intensity normalization.

    This preserves the original simulation convention: the reference distance
    is measured from the central ommatidium apex to P=(40, 0, R+20).
    """

    return np.array([40.0, 0.0, config.sphere_radius_mm + 20.0], dtype=np.float64)


def build_eye_layout(config: SimulationConfig) -> EyeLayout:
    """Build the hemispherical ommatidial lattice and precompute optical terms."""

    height, width = config.shape
    half_h = (height - 1) / 2.0
    half_w = (width - 1) / 2.0
    gap_h = config.sphere_radius_mm / half_h
    gap_w = config.sphere_radius_mm / half_w

    row_grid, col_grid = np.indices((height, width), dtype=np.float64)
    x_grid = (col_grid - half_w) * gap_w
    y_grid = -(row_grid - half_h) * gap_h

    radial_sq = x_grid * x_grid + y_grid * y_grid
    valid_mask = radial_sq < (config.sphere_radius_mm * config.sphere_radius_mm)

    z_grid = np.zeros_like(x_grid)
    z_grid[valid_mask] = np.sqrt(config.sphere_radius_mm * config.sphere_radius_mm - radial_sq[valid_mask])

    rows, cols = np.nonzero(valid_mask)
    rows = rows.astype(np.int32)
    cols = cols.astype(np.int32)
    surface_positions = np.stack(
        [x_grid[valid_mask], y_grid[valid_mask], z_grid[valid_mask]],
        axis=1,
    ).astype(np.float64)

    directions = surface_positions / np.linalg.norm(surface_positions, axis=1, keepdims=True)
    fov_deg = compute_variable_fov(
        z_values=surface_positions[:, 2],
        center_fov_deg=config.center_fov_deg,
        edge_fov_deg=config.edge_fov_deg,
        blend=config.fov_blend,
        weight_gamma=config.fov_weight_gamma,
    )
    tan_half_fov = np.tan(np.radians(fov_deg / 2.0))

    apex_offset = config.image_sensor_radius_mm / tan_half_fov
    apex_positions = surface_positions - directions * apex_offset[:, None]

    pixel_size_coeff = compute_cylinder_cut_pixel_size_coeff(
        surface_positions=surface_positions,
        sphere_radius_mm=config.sphere_radius_mm,
        image_shape=config.shape,
        gap_to_diameter_ratio=config.pixel_gap_to_diameter_ratio,
        sample_resolution=config.pixel_area_sample_resolution,
    )

    center_eye_index = int(np.argmax(surface_positions[:, 2]))
    reference_distance = float(
        np.linalg.norm(reference_source_position(config) - apex_positions[center_eye_index])
    )

    return EyeLayout(
        rows=rows,
        cols=cols,
        surface_positions=surface_positions,
        directions=directions,
        apex_positions=apex_positions,
        tan_half_fov=tan_half_fov.astype(np.float64),
        pixel_size_coeff=pixel_size_coeff.astype(np.float64),
        reference_distance_mm=reference_distance,
        image_shape=config.shape,
    )


def build_source_state(distance_mm: float, incidence_angle_deg: float, config: SimulationConfig) -> SourceState:
    """
    Construct the source state under the 0-to-90 degree incidence convention.

    The source center is located at distance d from the origin:
    x=d*sin(theta), y=0, z=d*cos(theta). Its optical axis points to the origin.
    """

    if distance_mm <= 0.0:
        raise ValueError("distance_mm must be positive.")
    if not (0.0 <= incidence_angle_deg <= 90.0):
        raise ValueError("incidence_angle_deg must be in [0, 90].")

    theta = math.radians(float(incidence_angle_deg))
    position = np.array(
        [
            float(distance_mm) * math.sin(theta),
            0.0,
            float(distance_mm) * math.cos(theta),
        ],
        dtype=np.float64,
    )
    axis = -position / max(float(np.linalg.norm(position)), 1e-12)
    half_angle = config.source_divergence_deg / 2.0

    return SourceState(
        position=position,
        axis=axis.astype(np.float64),
        radius_mm=config.source_diameter_mm / 2.0,
        cos_half_divergence=math.cos(math.radians(half_angle)),
    )


def intersection_area_ratio(
    circle_distance: np.ndarray,
    view_radius: np.ndarray,
    light_radius: float,
) -> np.ndarray:
    """Compute the overlap area divided by the ommatidial view-circle area."""

    distance = circle_distance.astype(np.float64)
    rv = view_radius.astype(np.float64)
    rl = float(light_radius)
    ratio = np.zeros_like(distance, dtype=np.float64)

    no_overlap = distance >= (rv + rl)
    light_inside_view = (distance <= np.abs(rv - rl)) & (rv >= rl)
    view_inside_light = (distance <= np.abs(rv - rl)) & (rv < rl)
    partial = ~(no_overlap | light_inside_view | view_inside_light)

    if np.any(light_inside_view):
        ratio[light_inside_view] = (rl * rl) / (rv[light_inside_view] * rv[light_inside_view])
    if np.any(view_inside_light):
        ratio[view_inside_light] = 1.0
    if np.any(partial):
        d_p = distance[partial]
        rv_p = rv[partial]
        proj = (rv_p * rv_p - rl * rl + d_p * d_p) / (2.0 * d_p)
        alpha = np.arccos(np.clip(proj / rv_p, -1.0, 1.0))
        beta = np.arccos(np.clip((d_p - proj) / rl, -1.0, 1.0))
        area_a = rv_p * rv_p * alpha - rv_p * np.sin(alpha) * proj
        area_b = rl * rl * beta - rl * np.sin(beta) * (d_p - proj)
        ratio[partial] = (area_a + area_b) / (math.pi * rv_p * rv_p)

    return np.clip(ratio, 0.0, 1.0)


def render_frame(source: SourceState, layout: EyeLayout, config: SimulationConfig) -> tuple[np.ndarray, FrameStats]:
    """
    Render one grayscale compound-eye response and return visibility statistics.

    The geometric visibility count is independent of final grayscale
    quantization. The responsive count is the number of non-zero pixels in the
    rendered uint8 image.
    """

    flat_image = np.zeros(layout.eye_count, dtype=np.float64)
    fov_overlap_count = 0
    geometric_visible_count = 0

    for start in range(0, layout.eye_count, config.chunk_size):
        end = min(start + config.chunk_size, layout.eye_count)

        apex = layout.apex_positions[start:end]
        directions = layout.directions[start:end]
        tan_half_fov = layout.tan_half_fov[start:end]
        pixel_size = layout.pixel_size_coeff[start:end]

        eye_to_source = source.position - apex
        axial = np.sum(eye_to_source * directions, axis=1)
        front_mask = axial > 0.0
        if not np.any(front_mask):
            continue

        front_indices = np.nonzero(front_mask)[0]
        vec_front = eye_to_source[front_mask]
        axial_front = axial[front_mask]
        view_radius = axial_front * tan_half_fov[front_mask]

        distance_sq = np.sum(vec_front * vec_front, axis=1)
        radial_sq = np.maximum(distance_sq - axial_front * axial_front, 0.0)
        radial_distance = np.sqrt(radial_sq)

        overlap_mask = radial_distance < (view_radius + source.radius_mm)
        fov_overlap_count += int(np.count_nonzero(overlap_mask))
        if not np.any(overlap_mask):
            continue

        overlap_indices = front_indices[overlap_mask]
        vec_overlap = vec_front[overlap_mask]
        view_radius_overlap = view_radius[overlap_mask]
        radial_overlap = radial_distance[overlap_mask]
        distance = np.sqrt(distance_sq[overlap_mask])

        source_to_eye = -vec_overlap
        cos_theta = (source_to_eye @ source.axis) / np.maximum(distance, 1e-12)
        emit_mask = cos_theta > source.cos_half_divergence
        geometric_visible_count += int(np.count_nonzero(emit_mask))
        if not np.any(emit_mask):
            continue

        emit_indices = overlap_indices[emit_mask]
        distance_emit = distance[emit_mask]
        overlap_ratio = intersection_area_ratio(
            circle_distance=radial_overlap[emit_mask],
            view_radius=view_radius_overlap[emit_mask],
            light_radius=source.radius_mm,
        )
        angular_weight = np.clip(
            (cos_theta[emit_mask] - source.cos_half_divergence) / (1.0 - source.cos_half_divergence),
            0.0,
            1.0,
        )
        distance_weight = (layout.reference_distance_mm / np.maximum(distance_emit, 1e-12)) ** 2
        pixel_size_emit = pixel_size[overlap_indices][emit_mask]

        intensity = (
            config.source_intensity
            * overlap_ratio
            * angular_weight
            * distance_weight
            * pixel_size_emit
        )
        flat_image[start + emit_indices] = np.clip(intensity, 0.0, 255.0)

    image = np.zeros(layout.image_shape, dtype=np.uint8)
    image[layout.rows, layout.cols] = np.clip(flat_image, 0.0, 255.0).astype(np.uint8)

    stats = FrameStats(
        fov_overlap_count=fov_overlap_count,
        geometric_visible_count=geometric_visible_count,
        responsive_pixel_count=int(np.count_nonzero(image)),
    )
    return image, stats


def save_gray_image(image: np.ndarray, path: Path) -> None:
    """Save a uint8 grayscale image."""

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="L").save(path)


def save_single_curve(
    x_values: np.ndarray,
    y_values: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
) -> None:
    """Save a publication-style line plot."""

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=180)
    ax.plot(x_values, y_values, linewidth=2.2, marker="o", markersize=3.0, markeredgewidth=0.0)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    ax.set_xlim(float(np.min(x_values)), float(np.max(x_values)))
    ax.set_ylim(bottom=0.0)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run_fixed_angle_experiment(
    config: SimulationConfig,
    incidence_angle_deg: float,
    distance_range: tuple[float, float, float],
) -> None:
    """Run one incidence angle over multiple source-center distances."""

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    layout = build_eye_layout(config)
    distances = inclusive_float_range(*distance_range)
    id_width = max(3, len(str(max(len(distances) - 1, 0))))

    rows: list[list[float]] = []
    for index, distance_mm in enumerate(distances):
        source = build_source_state(distance_mm, incidence_angle_deg, config)
        image, stats = render_frame(source, layout, config)
        save_gray_image(image, output_dir / f"{index:0{id_width}d}_distance_{distance_mm:06.2f}mm.png")
        rows.append(
            [
                distance_mm,
                stats.responsive_pixel_count,
                stats.geometric_visible_count,
                stats.fov_overlap_count,
            ]
        )

    data = np.asarray(rows, dtype=np.float64)
    np.savetxt(
        output_dir / f"fixed_angle_{incidence_angle_deg:05.1f}_distance_sweep.csv",
        data,
        delimiter=",",
        header="distance_mm,responsive_pixel_count,geometric_visible_count,fov_overlap_count",
        comments="",
        fmt=["%.6f", "%.0f", "%.0f", "%.0f"],
    )
    save_single_curve(
        x_values=data[:, 0],
        y_values=data[:, 2],
        title=f"Geometric Visibility vs Distance (Incidence {incidence_angle_deg:.1f} deg)",
        xlabel="Distance to sphere center (mm)",
        ylabel="Geometrically visible ommatidia",
        path=output_dir / f"geometric_visibility_vs_distance_angle_{incidence_angle_deg:05.1f}.png",
    )

    print(f"Fixed-angle experiment completed: {output_dir}")
    print(f"Frames: {len(distances)}")


def run_fixed_distance_experiment(
    config: SimulationConfig,
    distance_mm: float,
    angle_range: tuple[float, float, float],
) -> None:
    """Run one source-center distance over multiple incidence angles."""

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    layout = build_eye_layout(config)
    angles = inclusive_float_range(*angle_range)
    id_width = max(3, len(str(max(len(angles) - 1, 0))))

    rows: list[list[float]] = []
    for index, angle_deg in enumerate(angles):
        source = build_source_state(distance_mm, angle_deg, config)
        image, stats = render_frame(source, layout, config)
        save_gray_image(image, output_dir / f"{index:0{id_width}d}_angle_{angle_deg:06.2f}deg.png")
        rows.append(
            [
                angle_deg,
                stats.responsive_pixel_count,
                stats.geometric_visible_count,
                stats.fov_overlap_count,
            ]
        )

    data = np.asarray(rows, dtype=np.float64)
    np.savetxt(
        output_dir / f"fixed_distance_{distance_mm:06.2f}mm_angle_sweep.csv",
        data,
        delimiter=",",
        header="incidence_angle_deg,responsive_pixel_count,geometric_visible_count,fov_overlap_count",
        comments="",
        fmt=["%.6f", "%.0f", "%.0f", "%.0f"],
    )
    save_single_curve(
        x_values=data[:, 0],
        y_values=data[:, 2],
        title=f"Geometric Visibility vs Incidence Angle (Distance {distance_mm:.1f} mm)",
        xlabel="Incidence angle (deg)",
        ylabel="Geometrically visible ommatidia",
        path=output_dir / f"geometric_visibility_vs_angle_distance_{distance_mm:06.2f}mm.png",
    )

    print(f"Fixed-distance experiment completed: {output_dir}")
    print(f"Frames: {len(angles)}")


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add parameters shared by both experiments."""

    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shape", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"), default=(601, 601))
    parser.add_argument("--source-intensity", type=float, default=20000.0)
    parser.add_argument("--source-diameter", type=float, default=0.2)
    parser.add_argument("--source-divergence", type=float, default=180.0)
    parser.add_argument("--center-fov", type=float, default=22.0)
    parser.add_argument("--edge-fov", type=float, default=40.0)
    parser.add_argument("--fov-blend", type=float, default=1.0)
    parser.add_argument("--fov-weight-gamma", type=float, default=0.273)
    parser.add_argument("--chunk-size", type=int, default=50000)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Minimal compound-eye visibility simulator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fixed_angle = subparsers.add_parser(
        "fixed-angle",
        help="Use one incidence angle and scan source-center distance.",
    )
    add_common_arguments(fixed_angle)
    fixed_angle.add_argument("--incidence-angle", type=float, required=True)
    fixed_angle.add_argument(
        "--distance-range",
        nargs=3,
        type=float,
        metavar=("START_MM", "END_MM", "STEP_MM"),
        required=True,
    )

    fixed_distance = subparsers.add_parser(
        "fixed-distance",
        help="Use one source-center distance and scan incidence angle.",
    )
    add_common_arguments(fixed_distance)
    fixed_distance.add_argument("--distance", type=float, required=True)
    fixed_distance.add_argument(
        "--angle-range",
        nargs=3,
        type=float,
        metavar=("START_DEG", "END_DEG", "STEP_DEG"),
        required=True,
    )

    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> SimulationConfig:
    """Create a simulation config from CLI options."""

    return SimulationConfig(
        shape=(int(args.shape[0]), int(args.shape[1])),
        source_intensity=float(args.source_intensity),
        source_diameter_mm=float(args.source_diameter),
        source_divergence_deg=float(args.source_divergence),
        center_fov_deg=float(args.center_fov),
        edge_fov_deg=float(args.edge_fov),
        fov_blend=float(args.fov_blend),
        fov_weight_gamma=float(args.fov_weight_gamma),
        chunk_size=max(1, int(args.chunk_size)),
        output_dir=args.output_dir,
    )


def main() -> None:
    """Run the selected experiment."""

    args = parse_args()
    config = build_config_from_args(args)

    if args.command == "fixed-angle":
        run_fixed_angle_experiment(
            config=config,
            incidence_angle_deg=float(args.incidence_angle),
            distance_range=tuple(float(v) for v in args.distance_range),
        )
    elif args.command == "fixed-distance":
        run_fixed_distance_experiment(
            config=config,
            distance_mm=float(args.distance),
            angle_range=tuple(float(v) for v in args.angle_range),
        )
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
