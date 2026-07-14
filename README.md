# SphereUFormer Compound-Eye Denoising

This repository provides the core code for compound-eye dataset simulation and SphereUFormer denoising. Supporting scripts for visibility simulation and eye tracking are also included.

## Contents

- `dataset_simulator.py`: generates `ideal`, `perfect`, and `noisy` images and creates the train, validation, and test lists.
- `train_sphereuformer.py`: trains SphereUFormer using paired noisy and ideal images.
- `compound_eye_visibility_simulation.py`: simulates compound-eye visibility under controlled source geometry.
- `eye_tracking.py` and `eye_motion.ino`: perform gaze estimation and optional servo-based eye motion.

## 1. System Requirements

The simulation and training workflow was tested with Ubuntu 20.04.6 LTS, Python 3.8.16, PyTorch 2.0.1, Torchvision 0.15.2, and CUDA 11.7. All remaining Python dependencies and tested version numbers are listed in `requirements.txt`.

An NVIDIA CUDA-capable GPU is recommended for full training. Dataset simulation and the compact demonstration can run on a CPU. No non-standard hardware is required for simulation or training. The optional eye-motion experiment requires a camera, an Arduino-compatible controller, a PCA9685 servo driver, and two-axis eye servos.

The required SphereUFormer modules are included in `network/` and `trimesh_utils.py`. Their third-party license is provided in `SPHERE_UFORMER_LICENSE`.

## 2. Installation

Run all commands from this directory.

```bash
conda create -n sphere_network python=3.8.16 pip -y
conda activate sphere_network
python -m pip install torch==2.0.1 torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cu117
python -m pip install -r requirements.txt
```

For CPU-only execution, replace the PyTorch command with:

```bash
python -m pip install torch==2.0.1+cpu torchvision==0.15.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu
```

Typical installation time is 5-15 minutes on a normal desktop computer.

## 3. Demo

The MPEG-7 CE Shape-1 Part B dataset contains 1,400 binary silhouette images from 70 object categories, with 20 images per category. Place these images in `./MPEG7dataset/original`. If the downloaded archive also contains `shapedata.gif` and `confusions.gif`, exclude these two overview images.

Generate the complete simulated dataset:

```bash
python dataset_simulator.py \
    --input-root ./MPEG7dataset/original \
    --output-root ./MPEG7dataset/sce/all
```

The expected output contains 1,400 `ideal`, 1,400 `perfect`, and 1,400 `noisy` PNG images. The default deterministic 80/10/10 split produces 1,120 training, 140 validation, and 140 test pairs.

Run the compact CPU demonstration:

```bash
python train_sphereuformer.py \
    --dataset_root_dir ./MPEG7dataset/sce/all \
    --output_dir runs/demo \
    --num_epochs 1 \
    --batch_size 2 \
    --num_workers 0 \
    --limit_train_batches 0 \
    --img_rank 3 \
    --num_scales 2 \
    --scale_depth 1 \
    --d_head_coef 1 \
    --enc_num_heads 2 4 \
    --dec_num_heads 8 4 \
    --use_checkpoint 0 \
    --no_gpu
```

Expected outputs are epoch-wise loss and PSNR values, `runs/demo/best.pt`, and `runs/demo/last.pt`. Full dataset simulation typically takes approximately 1-3 minutes on a desktop CPU; the compact training demonstration typically completes within 2 minutes.

## 4. Instructions for Use

Train SphereUFormer with the manuscript configuration:

```bash
python train_sphereuformer.py \
    --dataset_root_dir ./MPEG7dataset/sce/all \
    --output_dir runs/denoise \
    --num_epochs 15 \
    --batch_size 8 \
    --learning_rate 5e-5
```

The dataset root has the following structure:

```text
dataset_root/
  train_list.txt
  val_list.txt
  test_list.txt
  class_or_subset/
    xxx_device_sim_noisy_256x256.png
    xxx_ideal_256x256.png
```

Each split file contains one noisy/clean pair per line. To reproduce training, retain the generated split files and the default split seed (`42`).

## License

This project is released under the MIT License in `LICENSE`. The included SphereUFormer components retain their original MIT license in `SPHERE_UFORMER_LICENSE`.
