# Segmentation Evaluation

This folder contains the scripts used to train and evaluate the segmentation models for segmentation-based assessment in this project.

## Folder contents

```text
segmentation_evalustions/
├── train_segmentation.py
├── train_segmentation_t2.py
├── test_segmentation.py
├── test_segmentation_t2.py
└── README.md
```

## Models used in this folder

### 1. Four-channel segmentation model
Input order:

```text
[t1n, t1c, t2w, t2f]
```

This model can be evaluated in two modes:

- `real`: uses the real test modalities directly
- `generated`: generates `t1n`, `t1c`, and `t2f` from `t2w` and real `t2w`

### 2. T2-only segmentation model
Input:

```text
[t2w]
```


## Before running

Check the paths in `project_config.py`, especially:

```python
DATA["gan_test"]
TEST_UNET["model_path"]
TEST_GAN["checkpoint_path"]
```

## Training

### 1. Train the four-channel segmentation model

```bash
python train_segmentation.py
```

This writes:

```text
weight/unet3d_brats_binary.pth
```

### 2. Train the T2-only segmentation model

```bash
python train_segmentation_t2.py
```

This writes:

```text
weight/unet3d_brats_binary_t2_only.pth
```

## Evaluation

### 3. Evaluate the four-channel model on real test data

```bash
python test_segmentation.py --mode real
```

### 4. Evaluate the four-channel model on generated test data

```bash
python test_segmentation.py --mode generated
```

In generated mode, the segmentation input is built as:

```text
[generated t1n, generated t1c, real t2w, generated t2f]
```

### 5. Evaluate the T2-only segmentation model

```bash
python test_segmentation_t2.py
```

## Recommended workflow

```bash
python train_segmentation.py
python train_segmentation_t2.py
python test_segmentation.py --mode real
python test_segmentation.py --mode generated
python test_segmentation_t2.py
```
