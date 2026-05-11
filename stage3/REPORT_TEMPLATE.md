# ECS 170 Stage 3 — Report text for official template (Sections 1–3.7)

Use the same **Stage_2–5_Report_Template** as Stage 2: **10 pt**, **single space**, **≤ 5 pages**. Replace placeholders with your numbers, plots, and GitHub link.

**Team block (edit if needed)**  
Team name: …  
Student 1: Saahith Kalakuntla | ID: 920895078 | Email: skalakuntla@ucdavis.edu  

---

## Section 1: Task description

Stage 3 is **image classification** with **convolutional neural networks (CNNs)**. We use three course datasets: **MNIST** (handwritten digits 0–9, 28×28 gray), **ORL** (40 people, face images 112×92, stored as RGB with identical channels—we use **one** channel), and **CIFAR-10** (10 object classes, 32×32 color). For each dataset we train on the provided **training** split and report metrics on the **fixed test** split.

---

## Section 2: Model description

For each dataset we use a **CNN** implemented in PyTorch (`models.py` + `train_cnn.py` in our repo). Blocks use **convolution → ReLU** (and **BatchNorm** on CIFAR), **max pooling** where noted, then **global average pooling** and a **linear** classifier. ORL uses **40** outputs (one per person). MNIST and CIFAR use **10** outputs.

*(Insert three small architecture figures or one combined diagram: MNIST / ORL / CIFAR.)*

---

## Section 3: Experiment settings

### 3.1 Dataset description

Pickles **`MNIST`**, **`ORL`**, **`CIFAR`** under `stage_3_data/` match the course layout: each file has `train` and `test` lists of `{image, label}`. We convert images to **NCHW** tensors, scale pixels to **\[0, 1\]**, and map ORL labels **1…40** to **0…39** for training.

### 3.2 Detailed experimental setups

**Optimizer:** Adam. **Loss:** cross-entropy. **Batch sizes:** 128 for MNIST/CIFAR, **32** for ORL (smaller dataset). **Epochs:** (fill in—e.g. MNIST 12–20, ORL 60–120, CIFAR 15–25; increase if accuracy is still climbing.) **Learning rate:** (fill in—default `1e-3` in script; we used a lower LR for ORL in one run—note here if you change it.) **Device:** CUDA / MPS / CPU (auto in script).

**Ablations (course requirement):** run **default** vs **`--variant wide`** or **`--variant heavy_dropout`** on at least one dataset and compare.

### 3.3 Evaluation metrics

We report **accuracy** and **macro** precision, recall, and F1 (scikit-learn, `average='macro'`, all class indices included). Macro scores treat each class equally.

### 3.4 Source code

**https://github.com/SKalakuntla1/ECS_170_Project/tree/main/stage3**  

Stage 3 code is in the **`stage3/`** folder: `datasets.py`, `models.py`, `train_cnn.py`, `plot_history.py`, `README.md`.

### 3.5 Training convergence plot

For **each** dataset, plot **epoch** (x-axis) vs **train/test loss** and/or **test accuracy** (y-axis). Generate JSON with `train_cnn.py --out_json …`, then `python plot_history.py …`.

### 3.6 Model performance

Fill a table like:

| Dataset | Config | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---------|--------|----------|-------------------|----------------|------------|
| MNIST | default | … | … | … | … |
| ORL | default | … | … | … | … |
| CIFAR | default | … | … | … | … |
| (pick one) | ablation … | … | … | … | … |

### 3.7 Ablation studies

Describe what you changed (**depth / width / dropout / batch size / epochs / variant flag**). Tie numbers back to the table in §3.6. Explain **why** you think accuracy went up or down (overfitting, too few epochs, etc.).

---

## Checklist

- [ ] Unzipped **`stage_3_data`** next to `train_cnn.py`.  
- [ ] Trained **three** CNNs (MNIST, ORL, CIFAR) + at least **one** ablation.  
- [ ] Report uses the **official template** layout and **≤ 5 pages**.  
- [ ] §3.4 public GitHub link works.
