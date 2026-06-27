## Contents

- `DatasetsSimulator.py`  
  Generates binary-shape datasets under a regular compound-eye forward model. The script produces `ideal`, `perfect`, and `noisy` images, and also writes `train_list.txt`, `val_list.txt`, and `test_list.txt`.

- `CompoundEyeVisibilitySimulation.py`  
  Simulates variable-field-of-view visibility on a hemispherical compound eye under controlled source geometry.

- `Train_sphereuformer.py`  
  Trains a SphereUFormer-based denoiser from paired noisy/clean grayscale images stored in the dataset format produced by `DatasetsSimulator.py`.

- `Eye_tracking.py`  
  Tracks binocular foreground responses from sequential images, estimates centroid-based hemispherical angles, maps them to calibrated gaze values, and optionally sends servo commands.

- `Eye_motion.ino`  
  Arduino firmware for the servo controller used by `Eye_tracking.py`. It receives serial `X...Y...` commands and converts them into calibrated dual-axis eye motion.

## Minimal Workflow

1. Generate a synthetic dataset:

```bash
python DatasetsSimulator.py \
    --input-root ./MPEG7dataset/original \
    --output-root ./MPEG7dataset/sce/all
```

2. Train the denoising model:

```bash
python Train_sphereuformer.py \
    --dataset_root_dir ./MPEG7dataset/sce/all \
    --output_dir runs/denoise \
    --num_epochs 15 \
    --batch_size 8 \
    --learning_rate 5e-5
```



## Dataset Format

The denoising dataset root is expected to contain:

```text
dataset_root/
  train_list.txt
  val_list.txt
  test_list.txt
  class_or_subset/
    xxx_device_sim_noisy_256x256.png
    xxx_ideal_256x256.png
```

Each split file contains one paired sample per line:

```text
relative/path/to/noisy.png relative/path/to/clean.png
```
