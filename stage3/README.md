# ECS 170 — Stage 3 (CNN: MNIST, ORL, CIFAR)

## Data

Unzip **`stage_3_data.zip`** so this folder exists:

`stage_3_data/MNIST`, `stage_3_data/ORL`, `stage_3_data/CIFAR` (pickle files, no extension).

Example (from your Downloads):

```bash
cd stage3   # inside your cloned ECS_170_Project repo
unzip -q /path/to/stage_3_data.zip
```

That creates `stage_3_data/` next to this README (with `MNIST`, `ORL`, `CIFAR` pickle files inside).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

**MNIST** (raise `--epochs` for your final report, e.g. 15–25):

```bash
python train_cnn.py --dataset mnist --data_root stage_3_data --epochs 12 --out_json results_mnist_default.json
```

**ORL** (40 people, 1 channel from 112×92):

```bash
python train_cnn.py --dataset orl --data_root stage_3_data --epochs 30 --batch_size 32 --out_json results_orl_default.json
```

**CIFAR-10**:

```bash
python train_cnn.py --dataset cifar --data_root stage_3_data --epochs 15 --out_json results_cifar_default.json
```

## Ablations (course asks to vary architecture / hyperparameters)

`--variant wide` or `--variant heavy_dropout` changes the model in `models.py`.

Example:

```bash
python train_cnn.py --dataset cifar --variant wide --epochs 15 --out_json results_cifar_wide.json
```

## Plots

```bash
python plot_history.py results_mnist_default.json --out plot_mnist.png
```

## Files

| File | Role |
|------|------|
| `datasets.py` | Load pickles → `TensorDataset` (NCHW, labels 0-based) |
| `models.py` | `CNNMNIST`, `CNNORL`, `CNNCIFAR` |
| `train_cnn.py` | Training loop, JSON metrics + history |

Same code is also committed under the course repo: **https://github.com/SKalakuntla1/ECS_170_Project/tree/main/stage3**
