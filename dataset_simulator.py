#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script implements the standard dataset-generation pipeline used in the
project:
1. normalize a binary shape image,
2. sample it with a regular 64x64 compound-eye lattice,
3. add local shape perturbation,
4. apply edge-aware dropout noise.

Example:
    python dataset_simulator.py \
        --input-root ./MPEG7dataset/original \
        --output-root ./MPEG7dataset/sce/all
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


def load_binary_image(
    path: str,
    target_size: Tuple[int, int],
    invert: bool = False,
    pad_value: int = 0,
) -> np.ndarray:
    """
    Load an image, square-pad it, resize it, and binarize it.

    Parameters
    ----------
    path:
        Input image path.
    target_size:
        Output image size in (H, W).
    invert:
        Whether to invert intensity before thresholding.
    pad_value:
        Padding background value, 0 for black and 255 for white.
    """
    img = Image.open(path)

    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg.convert("L")
    else:
        img = img.convert("L")

    orig_w, orig_h = img.size
    square_size = max(orig_w, orig_h)
    if orig_w != orig_h:
        canvas = Image.new("L", (square_size, square_size), color=pad_value)
        left = (square_size - orig_w) // 2
        top = (square_size - orig_h) // 2
        canvas.paste(img, (left, top))
        img = canvas

    img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    if invert:
        arr = 1.0 - arr
    return (arr >= 0.5).astype(np.float32)


def center_on_canvas(
    image: np.ndarray,
    canvas_size: Tuple[int, int] = (256, 256),
) -> np.ndarray:
    """Place a smaller binary image on the center of a larger black canvas."""
    out_h, out_w = canvas_size
    h, w = image.shape
    canvas = np.zeros((out_h, out_w), dtype=np.float32)
    top = (out_h - h) // 2
    left = (out_w - w) // 2
    canvas[top:top + h, left:left + w] = image
    return canvas


def save_image(image: np.ndarray, path: str) -> None:
    """Save a float image in [0, 1] as an 8-bit grayscale PNG."""
    path = str(path)
    Image.fromarray((np.clip(image, 0.0, 1.0) * 255).astype(np.uint8), mode="L").save(path)


def build_shape_templates() -> Dict[str, List[np.ndarray]]:
    """Build local perturbation templates inside one 4x4 ommatidium patch."""
    base = [(1, 1), (1, 2), (2, 1), (2, 2)]

    shapes_full = [np.array(base, dtype=np.int32)]

    shapes_missing = []
    for k in range(len(base)):
        coords = [base[m] for m in range(len(base)) if m != k]
        shapes_missing.append(np.array(coords, dtype=np.int32))

    neighbors = [
        (0, 1), (0, 2),
        (3, 1), (3, 2),
        (1, 0), (2, 0),
        (1, 3), (2, 3),
    ]
    shapes_protrusion = [np.array(base + [nb], dtype=np.int32) for nb in neighbors]

    shapes_cross = [
        np.array([(1, 1), (0, 1), (2, 1), (1, 0), (1, 2)], dtype=np.int32),
        np.array([(1, 2), (0, 2), (2, 2), (1, 1), (1, 3)], dtype=np.int32),
        np.array([(2, 1), (1, 1), (3, 1), (2, 0), (2, 2)], dtype=np.int32),
        np.array([(2, 2), (1, 2), (3, 2), (2, 1), (2, 3)], dtype=np.int32),
    ]

    shapes_single = [np.array([coord], dtype=np.int32) for coord in base]

    shapes_pair = [
        np.array([(1, 1), (1, 2)], dtype=np.int32),
        np.array([(2, 1), (2, 2)], dtype=np.int32),
        np.array([(1, 1), (2, 1)], dtype=np.int32),
        np.array([(1, 2), (2, 2)], dtype=np.int32),
        np.array([(1, 1), (2, 2)], dtype=np.int32),
        np.array([(1, 2), (2, 1)], dtype=np.int32),
    ]

    shapes_triangle = [
        np.array([(0, 1), (2, 0), (2, 2)], dtype=np.int32),
        np.array([(3, 1), (1, 0), (1, 2)], dtype=np.int32),
    ]

    return {
        "full": shapes_full,
        "missing": shapes_missing,
        "protrusion": shapes_protrusion,
        "cross": shapes_cross,
        "single": shapes_single,
        "pair": shapes_pair,
        "triangle": shapes_triangle,
    }


SHAPE_TEMPLATES = build_shape_templates()
SHAPE_TYPE_PROBS = {
    "full": 0.10,
    "missing": 0.25,
    "protrusion": 0.25,
    "cross": 0.05,
    "single": 0.20,
    "pair": 0.10,
    "triangle": 0.05,
}


def sample_shape_local_coords(rng: np.random.Generator) -> np.ndarray:
    """Sample one perturbation template according to fixed class probabilities."""
    p = rng.random()
    acc = 0.0
    chosen_key = "full"
    for key, prob in SHAPE_TYPE_PROBS.items():
        acc += prob
        if p < acc:
            chosen_key = key
            break

    shape_list = SHAPE_TEMPLATES[chosen_key]
    idx = int(rng.integers(0, len(shape_list)))
    return shape_list[idx]


def simulate_compound_eye(
    ideal_img: np.ndarray,
    sensor_size: Tuple[int, int] = (256, 256),
    num_wires: Tuple[int, int] = (64, 64),
    eye_size: int = 2,
    gap: int = 2,
    threshold: float = 0.5,
    add_shape_noise: bool = False,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate a regular compound-eye readout.

    Without shape noise, each active ommatidium produces a 2x2 block.
    With shape noise, the block is replaced by a sampled local template.
    """
    h, w = sensor_size
    ny, nx = num_wires
    assert ideal_img.shape == (h, w)

    needed = ny * eye_size + (ny - 1) * gap
    assert needed <= h

    margin = (h - needed) // 2
    stride = eye_size + gap
    out = np.zeros((h, w), dtype=np.float32)
    rng = np.random.default_rng(seed) if add_shape_noise else None

    for iy in range(ny):
        for ix in range(nx):
            r0 = margin + stride * iy
            c0 = margin + stride * ix
            cy = r0 + (eye_size / 2.0 - 0.5)
            cx = c0 + (eye_size / 2.0 - 0.5)

            y0 = int(np.floor(cy))
            x0 = int(np.floor(cx))
            y1 = min(y0 + 1, h - 1)
            x1 = min(x0 + 1, w - 1)

            wy = cy - y0
            wx = cx - x0
            meas = (
                (1 - wy) * (1 - wx) * ideal_img[y0, x0] +
                (1 - wy) * wx * ideal_img[y0, x1] +
                wy * (1 - wx) * ideal_img[y1, x0] +
                wy * wx * ideal_img[y1, x1]
            )

            if meas < threshold:
                continue

            if not add_shape_noise:
                out[r0:r0 + eye_size, c0:c0 + eye_size] = 1.0
            else:
                assert rng is not None
                coords_local = sample_shape_local_coords(rng)
                r_origin = r0 - 1
                c_origin = c0 - 1
                for lr, lc in coords_local:
                    rr = r_origin + int(lr)
                    cc = c_origin + int(lc)
                    if 0 <= rr < h and 0 <= cc < w:
                        out[rr, cc] = 1.0

    return out


def apply_block_and_pixel_dropout(
    perfect_img: np.ndarray,
    noisy_img: np.ndarray,
    block_drop_prob: float = 0.001,
    pixel_drop_prob: float = 0.1,
    sensor_size: Tuple[int, int] = (256, 256),
    num_wires: Tuple[int, int] = (64, 64),
    eye_size: int = 2,
    gap: int = 2,
    seed: Optional[int] = None,
    edge_factor: float = 100.0,
) -> np.ndarray:
    """
    Apply edge-aware dropout on the noisy compound-eye image.

    Three cell types are identified on the 64x64 ommatidium grid:
    inner, edge, and isolated. Dropout is applied only to non-inner cells.
    """
    h, w = sensor_size
    ny, nx = num_wires
    assert perfect_img.shape == (h, w)
    assert noisy_img.shape == (h, w)

    rng = np.random.default_rng(seed)
    out = noisy_img.copy()

    needed = ny * eye_size + (ny - 1) * gap
    margin = (h - needed) // 2
    stride = eye_size + gap

    cell_on = np.zeros((ny, nx), dtype=bool)
    for iy in range(ny):
        for ix in range(nx):
            r0 = margin + stride * iy
            c0 = margin + stride * ix
            block = perfect_img[r0:r0 + eye_size, c0:c0 + eye_size]
            cell_on[iy, ix] = (block.max() > 0.5)

    if not cell_on.any():
        return out

    m = cell_on.astype(np.uint8)
    padded = np.pad(m, 1, mode="constant", constant_values=0)
    neighbor_sum = (
        padded[0:-2, 0:-2] + padded[0:-2, 1:-1] + padded[0:-2, 2:] +
        padded[1:-1, 0:-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:] +
        padded[2:,   0:-2] + padded[2:,   1:-1] + padded[2:,   2:]
    )

    iso_cells = cell_on & (neighbor_sum <= 1)
    edge_cells = cell_on & (neighbor_sum >= 2) & (neighbor_sum <= 5)
    inner_cells = cell_on & (neighbor_sum >= 6)

    edge_prob = min(1.0, pixel_drop_prob * edge_factor)
    iso_prob = min(1.0, edge_prob * 2.0)
    drop_cells = np.zeros((ny, nx), dtype=bool)

    if edge_prob > 0:
        for iy in range(ny):
            for ix in range(nx):
                if not edge_cells[iy, ix] or drop_cells[iy, ix]:
                    continue
                if rng.random() < edge_prob:
                    drop_cells[iy, ix] = True
                    neighs = []
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny_ = iy + dy
                        nx_ = ix + dx
                        if 0 <= ny_ < ny and 0 <= nx_ < nx:
                            if edge_cells[ny_, nx_] and not drop_cells[ny_, nx_]:
                                neighs.append((ny_, nx_))

                    if neighs:
                        ny1, nx1 = neighs[int(rng.integers(len(neighs)))]
                        drop_cells[ny1, nx1] = True
                        remain = [(a, b) for (a, b) in neighs if (a, b) != (ny1, nx1)]
                        if remain and rng.random() < 0.5:
                            ny2, nx2 = remain[int(rng.integers(len(remain)))]
                            drop_cells[ny2, nx2] = True

    if iso_prob > 0:
        ys, xs = np.where(iso_cells)
        for iy, ix in zip(ys, xs):
            if rng.random() < iso_prob:
                drop_cells[iy, ix] = True

    if block_drop_prob > 0:
        ys, xs = np.where(cell_on & ~inner_cells & ~drop_cells)
        for iy, ix in zip(ys, xs):
            if rng.random() < block_drop_prob:
                drop_cells[iy, ix] = True

    ys, xs = np.where(drop_cells)
    for iy, ix in zip(ys, xs):
        r0 = margin + stride * iy
        c0 = margin + stride * ix
        out[r0:r0 + eye_size, c0:c0 + eye_size] = 0.0

    return out


def stable_seed(key: str) -> int:
    """Convert a string key into a reproducible 32-bit seed."""
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def process_single_image(
    input_path: Path,
    output_dir: Path,
    object_size: Tuple[int, int] = (180, 180),
) -> Tuple[str, str]:
    """Generate ideal, perfect, and noisy outputs for one image."""
    stem = input_path.stem
    ideal_small = load_binary_image(
        str(input_path),
        target_size=object_size,
        invert=False,
        pad_value=0,
    )
    ideal = center_on_canvas(ideal_small, canvas_size=(256, 256))

    perfect = simulate_compound_eye(ideal, add_shape_noise=False)
    seed = stable_seed(str(input_path))
    noisy_shape = simulate_compound_eye(ideal, add_shape_noise=True, seed=seed)
    noisy = apply_block_and_pixel_dropout(
        perfect_img=perfect,
        noisy_img=noisy_shape,
        block_drop_prob=0.001,
        pixel_drop_prob=0.1,
        seed=seed ^ 0x9E3779B1,
        edge_factor=100.0,
    )

    ideal_name = f"{stem}_ideal_256x256.png"
    perfect_name = f"{stem}_device_sim_perfect_256x256.png"
    noisy_name = f"{stem}_device_sim_noisy_256x256.png"

    save_image(ideal, output_dir / ideal_name)
    save_image(perfect, output_dir / perfect_name)
    save_image(noisy, output_dir / noisy_name)

    return noisy_name, ideal_name


def list_image_files(folder: Path) -> List[Path]:
    """List image files with common silhouette extensions."""
    exts = {".png", ".gif", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"}
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts])


def split_pairs(
    pairs: List[Tuple[str, str]],
    rng: np.random.Generator,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Split one class into train/val/test sets with deterministic shuffling."""
    if not pairs:
        return [], [], []

    shuffled = list(pairs)
    rng.shuffle(shuffled)
    n = len(shuffled)

    if n == 1:
        return shuffled, [], []
    if n == 2:
        return shuffled[:1], shuffled[1:], []

    train_count = int(round(n * train_ratio))
    val_count = int(round(n * val_ratio))
    train_count = max(1, train_count)
    val_count = max(1, val_count)

    test_count = n - train_count - val_count
    if test_count < 1:
        deficit = 1 - test_count
        while deficit > 0 and train_count > val_count and train_count > 1:
            train_count -= 1
            deficit -= 1
        while deficit > 0 and val_count > 1:
            val_count -= 1
            deficit -= 1

    train_end = train_count
    val_end = train_count + val_count
    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def write_pair_list(path: Path, pairs: List[Tuple[str, str]]) -> None:
    """Write one noisy/clean pair per line using paths relative to output_root."""
    with path.open("w", encoding="utf-8") as f:
        for noisy_rel, clean_rel in pairs:
            f.write(f"{noisy_rel} {clean_rel}\n")


def run_batch(
    input_root: Path,
    output_root: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    split_seed: int = 42,
) -> None:
    """
    Run batch conversion for a flat or class-organized shape dataset.

    If the input root contains no subdirectories, all files are treated as one
    class named 'all'.
    """
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Require 0 < train_ratio, 0 <= val_ratio, and train_ratio + val_ratio < 1.")

    output_root.mkdir(parents=True, exist_ok=True)

    class_dirs = sorted([p for p in input_root.iterdir() if p.is_dir()])
    if not class_dirs:
        class_dirs = [input_root]
        single_class_mode = True
        print(f"No subfolders found under {input_root}, using one class named 'all'.")
    else:
        single_class_mode = False
        print(f"Found {len(class_dirs)} classes under {input_root}.")

    split_rng = np.random.default_rng(split_seed)
    train_pairs: List[Tuple[str, str]] = []
    val_pairs: List[Tuple[str, str]] = []
    test_pairs: List[Tuple[str, str]] = []

    for class_dir in class_dirs:
        if single_class_mode:
            out_class_name = "all"
            out_dir = output_root / out_class_name
        else:
            out_class_name = class_dir.name
            out_dir = output_root / out_class_name

        out_dir.mkdir(parents=True, exist_ok=True)
        image_files = list_image_files(class_dir)
        print(f"[class] {out_class_name}: {len(image_files)} images")

        class_pairs: List[Tuple[str, str]] = []
        for image_path in image_files:
            noisy_name, ideal_name = process_single_image(image_path, out_dir)
            class_pairs.append(
                (
                    str(Path(out_class_name) / noisy_name),
                    str(Path(out_class_name) / ideal_name),
                )
            )
            print(f"    Processed image: {image_path.name}")

        class_train, class_val, class_test = split_pairs(
            class_pairs,
            split_rng,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
        train_pairs.extend(class_train)
        val_pairs.extend(class_val)
        test_pairs.extend(class_test)

    write_pair_list(output_root / "train_list.txt", train_pairs)
    write_pair_list(output_root / "val_list.txt", val_pairs)
    write_pair_list(output_root / "test_list.txt", test_pairs)
    print(
        "Wrote split files: "
        f"train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}"
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Release version of the compound-eye forward simulator.")
    parser.add_argument(
        "--input-root",
        type=str,
        default="/mnt/drive1/zhuyi/datasets/MPEG7dataset/original",
        help="Input dataset root. Supports both flat and class-organized folders.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="./output/mpeg7_sce_release",
        help="Output root for generated ideal/perfect/noisy images.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of samples assigned to the training split.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of samples assigned to the validation split.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed used for deterministic data splitting.",
    )
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_args()
    run_batch(
        Path(args.input_root),
        Path(args.output_root),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
    )


if __name__ == "__main__":
    main()
