#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script performs dual-eye tracking and gaze estimation from sequential
images.

It monitors an image folder, extracts foreground responses in mirrored left and
right eye regions of interest, estimates centroid-based hemispherical angles,
maps them to calibrated gaze values, and optionally sends synchronized servo
commands. The script also records diagnostic visualizations and summary outputs
for offline inspection.
"""

import time
from pathlib import Path

import cv2
import imageio.v2 as imageio
import matplotlib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import serial
except ImportError:
    serial = None


# =========================================================
# 1. User configuration
# =========================================================

# Project-local paths are used to keep the script portable and anonymous.
PROJECT_ROOT = Path(__file__).resolve().parent

# Folder where the camera or imaging software writes full-frame images.
IMAGE_DIR = PROJECT_ROOT / "data" / "images"

# Accepted image formats.
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# If True, the newest existing image is used as the background image.
# If False, the oldest existing image is used as the background image.
USE_NEWEST_EXISTING_IMAGE_AS_BACKGROUND = False


# Servo-control CSV. By default, the first two numeric columns are raw servo_x
# and servo_y values.
CONTROL_CSV = PROJECT_ROOT / "data" / "control" / "eye_move_data.csv"
CONTROL_IS_RAW_SERVO = True
AUTO_CREATE_DUMMY_CONTROL_CSV = False


# Calibration CSV files.
# Required convention:
#     x = true gaze
#     y = centroid hemispherical angle
# Column names can be x,y or true_gaze,centroid_angle.
LEFT_FIT_CSV = PROJECT_ROOT / "data" / "calibration" / "left_fit.csv"
RIGHT_FIT_CSV = PROJECT_ROOT / "data" / "calibration" / "right_fit.csv"


# Right ROI format: x_min, x_max, y_min, y_max.
# The left ROI is mirrored automatically by the full image width.
RIGHT_ROI = (110, 220, 105, 160)


# Image-processing parameters.
DIFF_THRESHOLD = 40
DBSCAN_EPS = 10
DBSCAN_MIN_SAMPLES = 10


# Hemisphere geometry.
# If HEMISPHERE_CENTER is None, the full image center is used.
# If HEMISPHERE_RADIUS_PIXELS is None, min(width, height) / 2 is used.
HEMISPHERE_CENTER = None
HEMISPHERE_RADIUS_PIXELS = None

# If True, the centroid angle is computed from full 2D radial distance.
# If False, only the horizontal distance from the apex is used.
USE_RADIAL_HEMISPHERE_ANGLE = True

# The side used for branch selection is based on centroid_x - apex_x.
RIGHT_SIDE_SIGN = 1.0
LEFT_SIDE_SIGN = -1.0


# Servo settings.
ENABLE_SERVO = True
SERIAL_PORT = "COM6"
BAUD_RATE = 115200
ALLOW_RUN_WITHOUT_SERIAL = True

SWAP_XY = True
INVERT_X = True
INVERT_Y = True

HARD_LIMIT_X_MIN, HARD_LIMIT_X_MAX = 165, 835
HARD_LIMIT_Y_MIN, HARD_LIMIT_Y_MAX = 330, 615

SERVO_SETTLE_TIME = 0.15


# Output settings.
SAVE_VIDEO_PATH = PROJECT_ROOT / "outputs" / "dual_eye_gaze_video.mp4"
OUTPUT_FPS = 10
MAX_WAIT_SECONDS = 120
FILE_STABLE_WAIT = 0.08


# Gaze plotting range.
GAZE_X_LIM = (-50, 50)
GAZE_Y_LIM = (-5, 5)


# Raw servo to true-gaze conversion.
SERVO_X_DEG_RANGE = 73.0
SERVO_Y_DEG_RANGE = 90.0
SERVO_CENTER = 500.0
SERVO_HALF_RANGE = 500.0

# The calibration file predicts one gaze axis from one centroid angle.
# Set this to "x" for horizontal gaze or "y" for vertical gaze.
PREDICTED_GAZE_AXIS = "x"


# =========================================================
# 2. CSV helpers
# =========================================================

def create_dummy_control_csv(path: Path):
    """Create a small servo-control CSV for dry-run testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    demo = pd.DataFrame({
        "servo_x": [500, 540, 580, 620, 660, 620, 580, 540, 500],
        "servo_y": [500, 500, 500, 500, 500, 530, 560, 530, 500],
    })
    demo.to_csv(path, index=False)
    print(f"Created dummy control CSV: {path}")


def read_numeric_columns(csv_path: Path):
    """Read a CSV and return numeric columns only."""
    df = pd.read_csv(csv_path)
    numeric = df.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.loc[:, numeric.notna().any()]
    numeric = numeric.dropna(how="all")

    if numeric.shape[1] == 0:
        df = pd.read_csv(csv_path, header=None)
        numeric = df.apply(pd.to_numeric, errors="coerce")
        numeric = numeric.loc[:, numeric.notna().any()]
        numeric = numeric.dropna(how="all")

    return numeric


def load_control_csv(control_csv: Path):
    """Load servo-control values and compute the corresponding true gaze."""
    if not control_csv.exists():
        if AUTO_CREATE_DUMMY_CONTROL_CSV:
            create_dummy_control_csv(control_csv)
        else:
            raise FileNotFoundError(f"Control CSV not found: {control_csv}")

    numeric = read_numeric_columns(control_csv)
    if numeric.shape[1] < 2:
        raise ValueError("The control CSV must contain at least two numeric columns.")

    arr = numeric.iloc[:, :2].dropna().to_numpy(dtype=float)
    if arr.size == 0:
        raise ValueError("The control CSV contains no valid numeric rows.")

    if CONTROL_IS_RAW_SERVO:
        servo_raw = arr.copy()
        servo_x = servo_raw[:, 0]
        servo_y = servo_raw[:, 1]

        true_gaze_x = (servo_x - SERVO_CENTER) / SERVO_HALF_RANGE * SERVO_X_DEG_RANGE
        true_gaze_y = (servo_y - SERVO_CENTER) / SERVO_HALF_RANGE * SERVO_Y_DEG_RANGE
        true_gaze = np.column_stack([true_gaze_x, true_gaze_y])
    else:
        servo_raw = None
        true_gaze = arr.copy()

    return servo_raw, true_gaze


# =========================================================
# 3. Image-file helpers
# =========================================================

def list_image_files(image_dir: Path):
    """Return image files sorted by modification time."""
    if not image_dir.exists():
        return []

    files = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(files, key=lambda p: p.stat().st_mtime)


def wait_file_stable(path: Path, stable_wait=0.08, max_try=10):
    """Wait until the image file size becomes stable."""
    last_size = -1
    for _ in range(max_try):
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            time.sleep(stable_wait)
            continue

        if size == last_size and size > 0:
            return True

        last_size = size
        time.sleep(stable_wait)

    return True


def wait_for_first_background_image(image_dir: Path):
    """Wait for and return the background image path."""
    print("Waiting for background image...")
    start = time.time()

    while True:
        files = list_image_files(image_dir)
        if files:
            bg_path = files[-1] if USE_NEWEST_EXISTING_IMAGE_AS_BACKGROUND else files[0]
            wait_file_stable(bg_path, FILE_STABLE_WAIT)
            return bg_path

        if time.time() - start > MAX_WAIT_SECONDS:
            raise RuntimeError("Timed out while waiting for the background image.")

        time.sleep(0.1)


def wait_for_next_image(image_dir: Path, last_mtime: float, last_path: Path):
    """Wait until a new image appears after the previous image."""
    start = time.time()

    while True:
        candidates = []
        for p in list_image_files(image_dir):
            try:
                mtime = p.stat().st_mtime
            except FileNotFoundError:
                continue

            if mtime > last_mtime and p != last_path:
                candidates.append((mtime, p))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            mtime, new_path = candidates[0]
            wait_file_stable(new_path, FILE_STABLE_WAIT)
            return new_path, mtime

        if time.time() - start > MAX_WAIT_SECONDS:
            return None, last_mtime

        time.sleep(0.05)


def mirror_roi_by_image_width(right_roi, image_width):
    """Mirror the right ROI horizontally to obtain the left ROI."""
    x_min, x_max, y_min, y_max = right_roi
    return (image_width - x_max, image_width - x_min, y_min, y_max)


def crop_roi(gray_img, roi):
    """Crop a rectangular ROI from a grayscale image."""
    x_min, x_max, y_min, y_max = roi
    return gray_img[y_min:y_max, x_min:x_max]


def check_roi_valid(roi, image_shape, name="ROI"):
    """Validate that an ROI is inside the image."""
    h, w = image_shape[:2]
    x_min, x_max, y_min, y_max = roi

    if x_min < 0 or y_min < 0 or x_max > w or y_max > h:
        raise ValueError(f"{name} is outside the image: {roi}, image width={w}, height={h}")

    if x_max <= x_min or y_max <= y_min:
        raise ValueError(f"{name} is invalid: {roi}")


# =========================================================
# 4. Servo control
# =========================================================

class ServoController:
    """Send raw servo positions through a serial port."""

    def __init__(self):
        self.ser = None

        if not ENABLE_SERVO:
            print("Servo output is disabled.")
            return

        if serial is None:
            msg = "pyserial is not installed. Install it with: pip install pyserial"
            if ALLOW_RUN_WITHOUT_SERIAL:
                print(f"{msg}. Continuing without servo output.")
                return
            raise ImportError(msg)

        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
            time.sleep(1.0)
            print(f"Serial connected: {SERIAL_PORT}, baud={BAUD_RATE}")
        except Exception as exc:
            if ALLOW_RUN_WITHOUT_SERIAL:
                print(f"Serial connection failed. Continuing without servo output: {exc}")
                self.ser = None
            else:
                raise RuntimeError(f"Serial connection failed: {exc}") from exc

    def send(self, raw_x, raw_y):
        """Send one raw servo command."""
        if self.ser is None:
            return

        rx = max(HARD_LIMIT_X_MIN, min(HARD_LIMIT_X_MAX, int(round(raw_x))))
        ry = max(HARD_LIMIT_Y_MIN, min(HARD_LIMIT_Y_MAX, int(round(raw_y))))

        vx = 1000 - rx if INVERT_X else rx
        vy = 1000 - ry if INVERT_Y else ry

        if SWAP_XY:
            cmd = f"X{vy}Y{vx}\n"
        else:
            cmd = f"X{vx}Y{vy}\n"

        self.ser.write(cmd.encode("utf-8"))
        print(f"Servo command: {cmd.strip()}")

    def close(self):
        """Close the serial port."""
        if self.ser is not None:
            self.ser.close()
            print("Serial port closed.")


# =========================================================
# 5. Hemisphere geometry and calibration lookup
# =========================================================

def get_hemisphere_geometry(image_shape):
    """Return the projected apex center and radius of the hemisphere."""
    h, w = image_shape[:2]

    if HEMISPHERE_CENTER is None:
        center_x = w / 2.0
        center_y = h / 2.0
    else:
        center_x, center_y = HEMISPHERE_CENTER

    if HEMISPHERE_RADIUS_PIXELS is None:
        radius = min(w, h) / 2.0
    else:
        radius = float(HEMISPHERE_RADIUS_PIXELS)

    if radius <= 0:
        raise ValueError("HEMISPHERE_RADIUS_PIXELS must be positive.")

    return center_x, center_y, radius


def full_centroid_to_hemisphere_angle(full_center, image_shape):
    """
    Convert a full-image centroid to a hemispherical angle.

    The rim is 0 degrees and the apex is 90 degrees.
    """
    if full_center is None:
        return np.nan, np.nan

    center_x, center_y, radius = get_hemisphere_geometry(image_shape)
    x, y = full_center

    dx = float(x) - center_x
    dy = float(y) - center_y

    if USE_RADIAL_HEMISPHERE_ANGLE:
        rho = np.sqrt(dx * dx + dy * dy)
    else:
        rho = abs(dx)

    normalized_rho = np.clip(rho / radius, 0.0, 1.0)
    angle_deg = float(np.degrees(np.arccos(normalized_rho)))

    side_sign = RIGHT_SIDE_SIGN if dx >= 0 else LEFT_SIDE_SIGN
    return angle_deg, side_sign


def roi_center_to_full_center(roi_center, roi):
    """Convert an ROI-local centroid to full-image coordinates."""
    if roi_center is None:
        return None

    x_min, _, y_min, _ = roi
    cx, cy = roi_center
    return float(cx + x_min), float(cy + y_min)


class HemisphericalAngleFit:
    """
    Predict true gaze from centroid hemispherical angle.

    The calibration CSV uses:
        x = true gaze
        y = centroid hemispherical angle
    """

    def __init__(self, fit_csv: Path):
        self.fit_csv = fit_csv
        self.all_branch = None
        self.positive_branch = None
        self.negative_branch = None
        self._load()

    @staticmethod
    def _compress_duplicate_angles(angle, gaze):
        """Average duplicate angle values so that interpolation is well defined."""
        df = pd.DataFrame({"angle": angle, "gaze": gaze}).dropna()
        df = df.groupby("angle", as_index=False)["gaze"].mean()
        df = df.sort_values("angle")
        return df["angle"].to_numpy(dtype=float), df["gaze"].to_numpy(dtype=float)

    @staticmethod
    def _make_branch(gaze, angle):
        """Create one interpolation branch sorted by centroid angle."""
        if len(gaze) == 0:
            return None

        angle_arr, gaze_arr = HemisphericalAngleFit._compress_duplicate_angles(angle, gaze)
        if len(angle_arr) == 0:
            return None

        return angle_arr, gaze_arr

    @staticmethod
    def _interp_branch(branch, centroid_angle):
        """Interpolate within one branch, with endpoint clipping."""
        if branch is None or not np.isfinite(centroid_angle):
            return np.nan

        angle_arr, gaze_arr = branch

        if len(angle_arr) == 1:
            return float(gaze_arr[0])

        clipped = np.clip(float(centroid_angle), angle_arr[0], angle_arr[-1])
        return float(np.interp(clipped, angle_arr, gaze_arr))

    def _load(self):
        if not self.fit_csv.exists():
            raise FileNotFoundError(f"Calibration CSV not found: {self.fit_csv}")

        df = pd.read_csv(self.fit_csv)
        lower_cols = {str(c).strip().lower(): c for c in df.columns}

        true_gaze_keys = ["x", "true_gaze", "gaze", "gaze_x", "real_gaze"]
        centroid_angle_keys = ["y", "centroid_angle", "hemisphere_angle", "solid_angle", "angle"]

        true_col = next((lower_cols[k] for k in true_gaze_keys if k in lower_cols), None)
        angle_col = next((lower_cols[k] for k in centroid_angle_keys if k in lower_cols), None)

        if true_col is not None and angle_col is not None:
            data = pd.DataFrame({
                "true_gaze": pd.to_numeric(df[true_col], errors="coerce"),
                "centroid_angle": pd.to_numeric(df[angle_col], errors="coerce"),
            }).dropna()
        else:
            numeric = read_numeric_columns(self.fit_csv)
            if numeric.shape[1] < 2:
                raise ValueError(
                    "The calibration CSV must contain at least two numeric columns: "
                    "x=true_gaze and y=centroid_angle."
                )

            data = numeric.iloc[:, :2].dropna().copy()
            data.columns = ["true_gaze", "centroid_angle"]

        if len(data) < 2:
            raise ValueError(f"The calibration CSV has too few valid points: {self.fit_csv}")

        gaze = data["true_gaze"].to_numpy(dtype=float)
        angle = data["centroid_angle"].to_numpy(dtype=float)

        self.all_branch = self._make_branch(gaze, angle)
        self.positive_branch = self._make_branch(gaze[gaze >= 0], angle[gaze >= 0])
        self.negative_branch = self._make_branch(gaze[gaze <= 0], angle[gaze <= 0])

        print(f"Loaded calibration CSV: {self.fit_csv}")
        print("Calibration convention: x=true_gaze, y=centroid_hemisphere_angle")

    def predict(self, centroid_angle, side_sign):
        """Predict true gaze by selecting the side-specific calibration branch."""
        if not np.isfinite(centroid_angle):
            return np.nan

        if side_sign >= 0 and self.positive_branch is not None:
            return self._interp_branch(self.positive_branch, centroid_angle)

        if side_sign < 0 and self.negative_branch is not None:
            return self._interp_branch(self.negative_branch, centroid_angle)

        return self._interp_branch(self.all_branch, centroid_angle)


def scalar_to_gaze_pair(pred_scalar):
    """Place the scalar prediction on the configured gaze axis."""
    if PREDICTED_GAZE_AXIS.lower() == "y":
        return (0.0, pred_scalar)
    return (pred_scalar, 0.0)


# =========================================================
# 6. ROI response processing
# =========================================================

def process_roi(curr_roi_img, bg_roi_img):
    """
    Extract the largest moving/changed cluster and return its centroid.

    Returns:
        center: ROI-local centroid as (cx, cy)
        bin_img: thresholded difference image
        final_pts: points in the largest valid cluster as [[y, x], ...]
    """
    if curr_roi_img.shape != bg_roi_img.shape:
        raise ValueError(f"ROI shape mismatch: {curr_roi_img.shape} vs {bg_roi_img.shape}")

    diff = cv2.absdiff(curr_roi_img, bg_roi_img)
    _, bin_img = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    pts_y, pts_x = np.where(bin_img > 0)
    if len(pts_y) < DBSCAN_MIN_SAMPLES:
        return None, bin_img, np.empty((0, 2), dtype=int)

    coords = np.column_stack((pts_y, pts_x))

    try:
        db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(coords)
    except Exception:
        return None, bin_img, np.empty((0, 2), dtype=int)

    labels = db.labels_
    valid_mask = labels != -1
    if not np.any(valid_mask):
        return None, bin_img, np.empty((0, 2), dtype=int)

    valid_coords = coords[valid_mask]
    valid_labels = labels[valid_mask]

    unique_labels, counts = np.unique(valid_labels, return_counts=True)
    max_label = unique_labels[np.argmax(counts)]
    final_pts = valid_coords[valid_labels == max_label]

    cy, cx = np.mean(final_pts, axis=0)
    return (float(cx), float(cy)), bin_img, final_pts


def roi_points_to_full_coords(roi_pts, roi):
    """Convert ROI-local points [[y, x], ...] to full-image points [[x, y], ...]."""
    if len(roi_pts) == 0:
        return np.empty((0, 2), dtype=float)

    x_min, _, y_min, _ = roi
    full_x = roi_pts[:, 1] + x_min
    full_y = roi_pts[:, 0] + y_min
    return np.column_stack([full_x, full_y])


# =========================================================
# 7. Visualization helpers
# =========================================================

def gaze_to_vector(gaze_x_deg, gaze_y_deg):
    """Convert horizontal and vertical gaze angles to a 3D unit vector."""
    if not np.isfinite(gaze_x_deg) or not np.isfinite(gaze_y_deg):
        return None

    hx = np.radians(gaze_x_deg)
    vy = np.radians(gaze_y_deg)

    x = np.sin(hx) * np.cos(vy)
    y = np.sin(vy)
    z = np.cos(hx) * np.cos(vy)

    vec = np.array([x, y, z], dtype=float)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return None

    return vec / norm


def make_basis_from_direction(direction):
    """Build an orthonormal basis around a direction vector."""
    direction = direction / np.linalg.norm(direction)

    tmp = np.array([0, 1, 0], dtype=float)
    if abs(np.dot(tmp, direction)) > 0.95:
        tmp = np.array([1, 0, 0], dtype=float)

    basis_u = np.cross(tmp, direction)
    basis_u = basis_u / np.linalg.norm(basis_u)

    basis_v = np.cross(direction, basis_u)
    basis_v = basis_v / np.linalg.norm(basis_v)

    return basis_u, basis_v, direction


def cap_surface(direction, theta_start, theta_end, radius=1.02, n_theta=10, n_phi=50):
    """Generate a spherical cap oriented along a direction vector."""
    basis_u, basis_v, basis_d = make_basis_from_direction(direction)

    theta = np.linspace(theta_start, theta_end, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    theta_grid, phi_grid = np.meshgrid(theta, phi)

    pts = (
        np.cos(theta_grid)[..., None] * basis_d
        + np.sin(theta_grid)[..., None] * np.cos(phi_grid)[..., None] * basis_u
        + np.sin(theta_grid)[..., None] * np.sin(phi_grid)[..., None] * basis_v
    ) * radius

    return pts[..., 0], pts[..., 1], pts[..., 2]


def draw_reconstructed_eye(ax, true_gaze, pred_gaze, title):
    """Draw a simple 3D eye with true and predicted gaze rays."""
    ax.set_title(title, fontsize=10)

    u = np.linspace(0, 2 * np.pi, 45)
    v = np.linspace(0, np.pi, 28)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))

    ax.plot_wireframe(xs, ys, zs, color="lightgray", linewidth=0.4, alpha=0.85)

    true_vec = gaze_to_vector(true_gaze[0], true_gaze[1])
    pred_vec = gaze_to_vector(pred_gaze[0], pred_gaze[1])

    if true_vec is not None:
        ix, iy, iz = cap_surface(true_vec, np.radians(8), np.radians(24), radius=1.03)
        px, py, pz = cap_surface(true_vec, 0, np.radians(8), radius=1.04)

        ax.plot_surface(ix, iy, iz, color="#A1AFB2", linewidth=0, shade=True, alpha=1.0)
        ax.plot_surface(px, py, pz, color="#636360", linewidth=0, shade=True, alpha=1.0)

        gt_end = true_vec * 1.45
        ax.plot([0, gt_end[0]], [0, gt_end[1]], [0, gt_end[2]], color="green", linestyle="--", linewidth=2.0)

    if pred_vec is not None:
        pred_end = pred_vec * 1.45
        ax.plot([0, pred_end[0]], [0, pred_end[1]], [0, pred_end[2]], color="red", linestyle="-", linewidth=2.0)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-0.2, 1.4)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=25, azim=-70)
    ax.axis("off")


def draw_response_map(ax, image_shape, left_full_pts, right_full_pts):
    """Draw the imager response points on a projected hemisphere."""
    ax.set_title("Imager response map", fontsize=10)

    center_x, center_y, radius = get_hemisphere_geometry(image_shape)

    grid = 95
    gx, gy = np.meshgrid(np.linspace(-1, 1, grid), np.linspace(-1, 1, grid))
    r2 = gx ** 2 + gy ** 2
    mask = r2 <= 1.0
    gz = np.sqrt(1.0 - r2[mask])

    ax.scatter(gx[mask], gy[mask], gz, s=4, c="#13205B", alpha=1.0, depthshade=False)

    def map_points(full_pts):
        if len(full_pts) == 0:
            return np.empty((0, 3), dtype=float)

        x = (full_pts[:, 0] - center_x) / radius
        y = (full_pts[:, 1] - center_y) / radius
        rr = x ** 2 + y ** 2
        valid = rr <= 1.0

        x = x[valid]
        y = y[valid]
        z = np.sqrt(1.0 - x ** 2 - y ** 2) + 0.035
        return np.column_stack([x, y, z])

    for pts in (map_points(left_full_pts), map_points(right_full_pts)):
        if len(pts) > 0:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=12, c="white", edgecolors="none", depthshade=False)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_zlim(0.0, 1.2)
    ax.set_box_aspect((1, 1, 0.7))
    ax.view_init(elev=20, azim=-90)
    ax.axis("off")


def valid_history_array(history):
    """Convert history points to a finite NumPy array."""
    if len(history) == 0:
        return np.empty((0, 2), dtype=float)

    arr = np.array(history, dtype=float).reshape(-1, 2)
    mask = np.isfinite(arr).all(axis=1)
    return arr[mask]


def draw_gaze_plot(ax, true_history, pred_history, title):
    """Draw true and predicted gaze history."""
    ax.set_title(title, fontsize=10)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axvline(0, color="black", linewidth=0.6)

    true_arr = valid_history_array(true_history)
    pred_arr = valid_history_array(pred_history)

    if len(true_arr) > 1:
        ax.plot(true_arr[:, 0], true_arr[:, 1], color="green", linewidth=1.0, alpha=0.45)

    if len(pred_arr) > 1:
        ax.plot(pred_arr[:, 0], pred_arr[:, 1], color="red", linewidth=1.0, alpha=0.45)

    if len(true_arr) > 0:
        ax.scatter(true_arr[-1, 0], true_arr[-1, 1], c="green", s=36, label="True Gaze")

    if len(pred_arr) > 0:
        ax.scatter(pred_arr[-1, 0], pred_arr[-1, 1], c="red", s=36, label="Est. Gaze")

    ax.set_xlim(*GAZE_X_LIM)
    ax.set_ylim(*GAZE_Y_LIM)
    ax.set_xlabel("Horizontal gaze (deg)", fontsize=8)
    ax.set_ylabel("Vertical gaze (deg)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.legend(loc="upper left", fontsize=7, frameon=False)


def render_dashboard_frame(
    frame_idx,
    image_shape,
    left_full_pts,
    right_full_pts,
    true_gaze,
    pred_left,
    pred_right,
    true_history,
    pred_left_history,
    pred_right_history,
    left_center,
    right_center,
    left_angle,
    right_angle,
):
    """Render one dashboard frame for the output video."""
    fig = plt.figure(figsize=(12, 6.5), dpi=120, facecolor="white")
    gs = fig.add_gridspec(2, 3)

    ax_info = fig.add_subplot(gs[0, 0])
    ax_info.axis("off")
    ax_info.text(0.5, 0.76, "Dual-eye gaze tracking setup", ha="center", va="center", fontsize=12)
    ax_info.text(0.5, 0.60, f"Frame: {frame_idx}", ha="center", va="center", fontsize=10)
    ax_info.text(0.5, 0.46, f"True gaze: ({true_gaze[0]:+.2f}, {true_gaze[1]:+.2f}) deg", ha="center", va="center", fontsize=10)
    ax_info.text(0.5, 0.32, f"Left center: {left_center}, angle={left_angle:.2f} deg", ha="center", va="center", fontsize=8)
    ax_info.text(0.5, 0.22, f"Right center: {right_center}, angle={right_angle:.2f} deg", ha="center", va="center", fontsize=8)

    ax_response = fig.add_subplot(gs[1, 0], projection="3d")
    draw_response_map(ax_response, image_shape, left_full_pts, right_full_pts)

    ax_right_eye = fig.add_subplot(gs[0, 1], projection="3d")
    draw_reconstructed_eye(ax_right_eye, true_gaze, pred_right, "Reconstructed 3D right eye")

    ax_left_eye = fig.add_subplot(gs[0, 2], projection="3d")
    draw_reconstructed_eye(ax_left_eye, true_gaze, pred_left, "Reconstructed 3D left eye")

    ax_right_gaze = fig.add_subplot(gs[1, 1])
    draw_gaze_plot(ax_right_gaze, true_history, pred_right_history, "Gaze tracking (right eye)")

    ax_left_gaze = fig.add_subplot(gs[1, 2])
    draw_gaze_plot(ax_left_gaze, true_history, pred_left_history, "Gaze tracking (left eye)")

    fig.tight_layout()
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return frame


# =========================================================
# 8. Main workflow
# =========================================================

def main():
    """Run the complete servo-image-gaze-video workflow."""
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Image folder not found: {IMAGE_DIR}")

    SAVE_VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)

    servo_raw_seq, true_gaze_seq = load_control_csv(CONTROL_CSV)
    if CONTROL_IS_RAW_SERVO and servo_raw_seq is None:
        raise RuntimeError("Raw servo values are required but were not loaded.")

    bg_path = wait_for_first_background_image(IMAGE_DIR)
    bg_mtime = bg_path.stat().st_mtime

    bg_img = cv2.imread(str(bg_path), cv2.IMREAD_GRAYSCALE)
    if bg_img is None:
        raise RuntimeError(f"Failed to read background image: {bg_path}")

    image_h, image_w = bg_img.shape[:2]
    left_roi = mirror_roi_by_image_width(RIGHT_ROI, image_w)
    right_roi = RIGHT_ROI

    check_roi_valid(left_roi, bg_img.shape, "LEFT_ROI")
    check_roi_valid(right_roi, bg_img.shape, "RIGHT_ROI")

    center_x, center_y, radius = get_hemisphere_geometry(bg_img.shape)
    print(f"Background image: {bg_path.name}")
    print(f"Image size: width={image_w}, height={image_h}")
    print(f"LEFT_ROI  = {left_roi}")
    print(f"RIGHT_ROI = {right_roi}")
    print(f"Hemisphere center=({center_x:.2f}, {center_y:.2f}), radius={radius:.2f}")

    bg_left = crop_roi(bg_img, left_roi)
    bg_right = crop_roi(bg_img, right_roi)

    left_lookup = HemisphericalAngleFit(LEFT_FIT_CSV)
    right_lookup = HemisphericalAngleFit(RIGHT_FIT_CSV)

    servo = ServoController()
    writer = imageio.get_writer(
        str(SAVE_VIDEO_PATH),
        fps=OUTPUT_FPS,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )

    true_history = []
    pred_left_history = []
    pred_right_history = []

    last_path = bg_path
    last_mtime = bg_mtime

    print("Starting servo control, image processing, and video rendering...")

    try:
        total = len(true_gaze_seq)

        for i in range(total):
            true_gaze = true_gaze_seq[i]

            if CONTROL_IS_RAW_SERVO:
                servo_x, servo_y = servo_raw_seq[i]
                servo.send(servo_x, servo_y)
            else:
                print("CONTROL_IS_RAW_SERVO=False, so no raw servo command is sent.")

            time.sleep(SERVO_SETTLE_TIME)

            curr_path, curr_mtime = wait_for_next_image(
                IMAGE_DIR,
                last_mtime=last_mtime,
                last_path=last_path,
            )

            if curr_path is None:
                print(f"Timed out while waiting for frame {i + 1}. Stopping early.")
                break

            curr_img = cv2.imread(str(curr_path), cv2.IMREAD_GRAYSCALE)
            if curr_img is None:
                print(f"Failed to read image, skipping: {curr_path}")
                continue

            if curr_img.shape != bg_img.shape:
                print(f"Image shape mismatch, skipping: {curr_path.name}, {curr_img.shape} vs {bg_img.shape}")
                continue

            last_path = curr_path
            last_mtime = curr_mtime

            curr_left = crop_roi(curr_img, left_roi)
            curr_right = crop_roi(curr_img, right_roi)

            left_center, _, left_pts = process_roi(curr_left, bg_left)
            right_center, _, right_pts = process_roi(curr_right, bg_right)

            left_full_center = roi_center_to_full_center(left_center, left_roi)
            right_full_center = roi_center_to_full_center(right_center, right_roi)

            left_angle, left_side = full_centroid_to_hemisphere_angle(left_full_center, curr_img.shape)
            right_angle, right_side = full_centroid_to_hemisphere_angle(right_full_center, curr_img.shape)

            pred_left_scalar = left_lookup.predict(left_angle, left_side)
            pred_right_scalar = right_lookup.predict(right_angle, right_side)

            pred_left = scalar_to_gaze_pair(pred_left_scalar)
            pred_right = scalar_to_gaze_pair(pred_right_scalar)

            left_full_pts = roi_points_to_full_coords(left_pts, left_roi)
            right_full_pts = roi_points_to_full_coords(right_pts, right_roi)

            true_history.append(tuple(true_gaze))
            pred_left_history.append(tuple(pred_left))
            pred_right_history.append(tuple(pred_right))

            frame = render_dashboard_frame(
                frame_idx=i + 1,
                image_shape=curr_img.shape,
                left_full_pts=left_full_pts,
                right_full_pts=right_full_pts,
                true_gaze=true_gaze,
                pred_left=pred_left,
                pred_right=pred_right,
                true_history=true_history,
                pred_left_history=pred_left_history,
                pred_right_history=pred_right_history,
                left_center=left_center,
                right_center=right_center,
                left_angle=left_angle,
                right_angle=right_angle,
            )

            writer.append_data(frame)

            print(
                f"[{i + 1}/{total}] "
                f"image={curr_path.name} | "
                f"True=({true_gaze[0]:+.2f}, {true_gaze[1]:+.2f}) | "
                f"L angle={left_angle:.2f}, L pred={pred_left_scalar:+.2f} | "
                f"R angle={right_angle:.2f}, R pred={pred_right_scalar:+.2f}"
            )

    finally:
        writer.close()
        servo.close()

    print(f"Video saved: {SAVE_VIDEO_PATH}")


if __name__ == "__main__":
    main()
