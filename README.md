# ECS 170 — Stage 2 (digit MLP)

This repo is for **ECS 170 Stage 2**. We train a small neural network (MLP) on the course digit data.

**Data:** Put the instructor files **`train.csv`** and **`test.csv`** inside a folder named **`stage_2_data/`**. Those files have **no header**. Column 0 is the label. Columns 1–784 are pixels.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_stage2.txt
```

## Run training

Baseline:

```bash
python stage2_train.py --data_dir stage_2_data --hidden 128,64 --optimizer adam --lr 1e-3 --epochs 50 --out_json results_baseline.json
```

Bigger hidden layers:

```bash
python stage2_train.py --data_dir stage_2_data --hidden 256,128,64 --optimizer adam --lr 1e-3 --epochs 50 --out_json results_deeper.json
```

Same shape as baseline, but SGD:

```bash
python stage2_train.py --data_dir stage_2_data --hidden 128,64 --optimizer sgd --lr 1e-3 --momentum 0.9 --epochs 50 --out_json results_sgd.json
```

The script prints **JSON**. It has loss and accuracy **each epoch**. It also prints final **accuracy** and **macro** precision, recall, and F1.

## Plots for the report

```bash
python plot_history.py results_baseline.json --out convergence_baseline.png
```

Repo: **https://github.com/SKalakuntla1/ESC_170_Project**
