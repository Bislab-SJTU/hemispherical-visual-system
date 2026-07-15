#!/usr/bin/env python3
"""
SphereUFormer training for hemispherical image denoising.

Dataset layout:
    dataset_root/
      train_list.txt
      val_list.txt
      noisy_image.png
      clean_image.png

Each list file contains one noisy/clean pair per line:
    relative/path/to/noisy.png relative/path/to/clean.png

Example:
    python train_sphereuformer_fig3a.py \
      --dataset_root_dir ./MPEG7dataset/sce/all \
      --output_dir runs/denoise \
      --num_epochs 15 \
      --batch_size 8 \
      --learning_rate 5e-5
"""

import argparse
import math
import os
import random
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from network_fig3a.sphere_model import SphereUFormer
from trimesh_utils_fig3a import IcoSphereRef, asSpherical


MEAN = 0.5
STD = 0.225


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SphereUFormer denoising.")
    parser.add_argument("--dataset_root_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="runs/denoise")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_frequency", type=int, default=1)
    parser.add_argument("--limit_train_batches", type=int, default=math.inf)

    parser.add_argument("--img_rank", type=int, default=7)
    parser.add_argument("--grid_width", type=int, default=256)
    parser.add_argument("--mode", type=str, default="vertex", choices=["vertex"])
    parser.add_argument("--num_scales", type=int, default=4)
    parser.add_argument("--win_size_coef", type=int, default=2)
    parser.add_argument("--scale_factor", type=int, default=2)
    parser.add_argument("--scale_depth", type=int, default=2)
    parser.add_argument("--d_head_coef", type=int, default=2)
    parser.add_argument("--enc_num_heads", nargs="+", type=int, default=[2, 4, 8, 16])
    parser.add_argument("--dec_num_heads", nargs="+", type=int, default=[16, 16, 8, 4])
    parser.add_argument("--downsample", type=str, default="center")
    parser.add_argument("--upsample", type=str, default="interpolate")
    parser.add_argument("--abs_pos_enc_in", type=int, default=1)
    parser.add_argument("--abs_pos_enc", type=int, default=1)
    parser.add_argument("--rel_pos_bias", type=int, default=1)
    parser.add_argument("--rel_pos_bias_size", type=int, default=7)
    parser.add_argument("--rel_pos_init_variance", type=float, default=1.0)
    parser.add_argument("--use_checkpoint", type=int, default=1)
    parser.add_argument("--no_gpu", dest="use_gpu", action="store_false")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_pairs(list_path: str) -> List[Tuple[str, str]]:
    pairs = []
    with open(list_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))
    if not pairs:
        raise RuntimeError(f"No image pairs found in {list_path}")
    return pairs


def read_gray(path: str, size: int) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    if image.shape[0] != size or image.shape[1] != size:
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    return image.astype(np.float32) / 255.0


class SCEDenoise(Dataset):
    """Maps planar grayscale pairs to rank-N sphere vertices."""

    def __init__(self, root_dir: str, list_file: str, sphere_rank: int, grid_width: int):
        self.root_dir = root_dir
        self.pairs = read_pairs(list_file)
        self.grid_width = grid_width

        ref = IcoSphereRef("vertex")
        normals = ref.get_normals(rank=sphere_rank)
        spherical = asSpherical(normals)
        theta = (spherical[:, 2] + 360) % 360
        phi = spherical[:, 1]

        hemi_mask = (theta >= 90) & (theta <= 270)
        self.hemi_mask = torch.tensor(hemi_mask, dtype=torch.bool)

        u = (theta - 90) / 180
        v = phi / 180
        grid = torch.from_numpy(np.stack((u, v), axis=1).astype(np.float32) * 2 - 1)
        grid = grid.reshape(1, -1, 1, 2)
        grid[:, ~self.hemi_mask, :, :] = 1.1
        self.sphere_grid = grid.float()

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        noisy_rel, clean_rel = self.pairs[idx]
        noisy = read_gray(os.path.join(self.root_dir, noisy_rel), self.grid_width)
        clean = read_gray(os.path.join(self.root_dir, clean_rel), self.grid_width)

        noisy_sphere = self._project_to_sphere(noisy)
        clean_sphere = self._project_to_sphere(clean)

        return {
            "noisy": (noisy_sphere - MEAN) / STD,
            "clean": (clean_sphere - MEAN) / STD,
            "mask": self.hemi_mask,
        }

    def _project_to_sphere(self, image: np.ndarray) -> torch.Tensor:
        image_t = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
        sphere = F.grid_sample(
            image_t,
            self.sphere_grid,
            padding_mode="border",
            align_corners=False,
        ).squeeze(0).squeeze(2).transpose(0, 1)
        return sphere * self.hemi_mask.unsqueeze(-1)


def build_model(args: argparse.Namespace) -> SphereUFormer:
    return SphereUFormer(
        img_rank=args.img_rank,
        node_type=args.mode,
        in_channels=1,
        out_channels=1,
        in_scale_factor=args.scale_factor,
        num_scales=args.num_scales,
        win_size_coef=args.win_size_coef,
        enc_depths=args.scale_depth,
        dec_depths=args.scale_depth,
        bottleneck_depth=args.scale_depth,
        d_head_coef=args.d_head_coef,
        enc_num_heads=args.enc_num_heads,
        dec_num_heads=args.dec_num_heads,
        abs_pos_enc_in=bool(args.abs_pos_enc_in),
        abs_pos_enc=bool(args.abs_pos_enc),
        rel_pos_bias=bool(args.rel_pos_bias),
        rel_pos_bias_size=args.rel_pos_bias_size,
        rel_pos_init_variance=args.rel_pos_init_variance,
        downsample=args.downsample,
        upsample=args.upsample,
        use_checkpoint=bool(args.use_checkpoint),
    )


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    reconstruction loss over valid spherical vertices.

    L = sum_{b,v} M_{b,v} |y_hat_{b,v} - y_{b,v}| / sum_{b,v} M_{b,v}
    """
    mask = mask.unsqueeze(-1).float()
    return ((pred - target).abs() * mask).sum() / mask.sum().clamp(min=1e-6)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    loss_sum = 0.0
    psnr_sum = 0.0
    count = 0

    for batch in loader:
        noisy = batch["noisy"].to(device)
        clean = batch["clean"].to(device)
        mask = batch["mask"].to(device)

        pred = model(noisy)
        loss = masked_l1(pred, clean, mask)

        pred01 = (pred * STD + MEAN).clamp(0, 1)
        clean01 = (clean * STD + MEAN).clamp(0, 1)
        mask_exp = mask.unsqueeze(-1).float()
        mse = ((pred01 - clean01).pow(2) * mask_exp).sum(dim=(1, 2))
        mse = mse / mask_exp.sum(dim=(1, 2)).clamp(min=1e-6)
        psnr = 20 * torch.log10(torch.tensor(1.0, device=device) / torch.sqrt(mse.clamp(min=1e-8)))

        batch_size = noisy.shape[0]
        loss_sum += loss.item() * batch_size
        psnr_sum += psnr.mean().item() * batch_size
        count += batch_size

    return {"loss": loss_sum / count, "psnr": psnr_sum / count}


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    limit_train_batches: int,
) -> float:
    model.train()
    loss_sum = 0.0
    count = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc="train"), start=1):
        noisy = batch["noisy"].to(device)
        clean = batch["clean"].to(device)
        mask = batch["mask"].to(device)

        pred = model(noisy)
        loss = masked_l1(pred, clean, mask)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_size = noisy.shape[0]
        loss_sum += loss.item() * batch_size
        count += batch_size

        if batch_idx >= limit_train_batches:
            break

    return loss_sum / count


def save_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    train_set = SCEDenoise(
        args.dataset_root_dir,
        os.path.join(args.dataset_root_dir, "train_list.txt"),
        args.img_rank,
        args.grid_width,
    )
    val_set = SCEDenoise(
        args.dataset_root_dir,
        os.path.join(args.dataset_root_dir, "val_list.txt"),
        args.img_rank,
        args.grid_width,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=False,
    )

    model = build_model(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_val = float("inf")
    for epoch in range(1, args.num_epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.grad_clip,
            args.limit_train_batches,
        )
        val_metrics = evaluate(model, val_loader, device)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_psnr={val_metrics['psnr']:.3f}"
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_checkpoint(os.path.join(args.output_dir, "best.pt"), model, optimizer, epoch)
        if args.save_frequency > 0 and epoch % args.save_frequency == 0:
            save_checkpoint(os.path.join(args.output_dir, "last.pt"), model, optimizer, epoch)


if __name__ == "__main__":
    main()
