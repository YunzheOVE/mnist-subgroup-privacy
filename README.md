# Subgroup Privacy & Utility Research on MNIST

This repository investigates classification performance, utility disparity, and privacy leakage on rare subgroups. Specifically, we study an imbalanced MNIST classification benchmark where digit **8** is made artificially rare.

---

## Project Structure

```text
research/
├── .gitignore            # Excludes datasets (data/), weights, caches
├── README.md             # Project overview, research notes & documentation
├── experiment_log.md     # Chronological log of experimental runs and metrics
├── phase1_mnist.py       # Phase 1: Non-private baseline CNN training script
├── phase1_results.json   # Phase 1 output metrics
├── phase2_dpsgd.py       # Phase 2: DP-SGD training script (Opacus)
└── phase2_results.json   # Phase 2 output metrics & privacy cost comparison
```

---

## Phase 1: Non-Private Baseline

### Overview
Before introducing Differential Privacy (e.g., DP-SGD / Opacus), Phase 1 measures how a standard deep learning model performs on majority digits (0–7, 9) versus the minority digit (8).

* **Dataset:** MNIST (60,000 train, 10,000 test).
* **Downsampling:** Digit 8 is subsampled to a specified fraction (default: `keep_eight = 0.09` ~9% kept), while digits 0–7 and 9 remain at 100%.
* **Model Architecture:**
  * 2× Convolutional Blocks: `Conv2d(1->16, 3x3)` + `ReLU` + `MaxPool2d(2)`, `Conv2d(16->32, 3x3)` + `ReLU` + `MaxPool2d(2)`
  * Classifier: `Linear(32 * 7 * 7 -> 10)` (~20.5k parameters)
* **Optimization:** SGD (`lr = 0.1`, `momentum = 0.9`, batch size 128, 8 epochs)
* **Metrics Tracked:**
  * Overall test accuracy
  * Per-digit test accuracy (especially for rare digit 8)
  * Per-digit test loss

---

## How to Run

### 1. Requirements
Ensure PyTorch and TorchVision are installed:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```
*(Or standard `pip install torch torchvision` if running with CUDA on a dedicated GPU).*

### 2. Running Phase 1
Run with default settings (`keep-eight=0.09`, `epochs=8`, `seed=42`):
```powershell
python phase1_mnist.py
```

Custom arguments:
```powershell
python phase1_mnist.py --keep-eight 0.05 --epochs 10 --seed 123
```

Results are printed to the console and automatically saved to `phase1_results.json`.

---

## Phase 2: Differential Privacy (DP-SGD) with Opacus

### Overview
Phase 2 introduces DP-SGD using Opacus to investigate how per-sample gradient clipping ($C$) and calibrated Gaussian noise ($\sigma$) affect the rare digit 8 compared to the control digit 2 and majority classes.

* **Privacy Engine:** Opacus `PrivacyEngine` with PRV/RDP privacy accounting.
* **Target Parameters:** $\varepsilon \approx 5.90$, $\delta = 10^{-6}$ (paper baseline).
* **Metrics Tracked:**
  * Achieved $\varepsilon$, $\delta$, and calibrated noise multiplier $\sigma$.
  * Per-digit test accuracy and test loss under DP.
  * **Privacy Cost:** $\text{Acc}_{\text{non-private}} - \text{Acc}_{\text{DP}}$.

### Running Phase 2
Run with default target $\varepsilon = 5.90$, $\delta = 10^{-6}$, 8 epochs:
```powershell
python phase2_dpsgd.py
```

Sweep with custom privacy targets:
```powershell
# Looser privacy (e.g. eps = 8.0):
python phase2_dpsgd.py --target-epsilon 8.0

# Tighter privacy (e.g. eps = 3.0):
python phase2_dpsgd.py --target-epsilon 3.0

# Explicit noise multiplier sigma and clipping norm C:
python phase2_dpsgd.py --noise-multiplier 1.5 --max-grad-norm 1.0
```

Results are printed with a comparative Privacy Cost table and saved to `phase2_results.json`.

---

## Research Notes & Observations

> *Record your hypotheses, findings, and notes for upcoming phases below.*

### Research Questions
1. **Utility Disparity:** At what threshold of `--keep-eight` does the classification accuracy of digit 8 collapse compared to the overall model accuracy?
2. **Impact of Differential Privacy (Future Phase):** When DP-SGD is added in subsequent phases, does noise addition and gradient clipping harm the rare digit 8 disproportionately compared to digits with full representation?

### Notes Log
* **Baseline Setup:** Phase 1 uses standard cross-entropy loss without class weighting to observe natural minority class decay.
* **Run 001 Baseline (9% keep-eight, 8 epochs, seed 42):**
  * Overall Test Accuracy: **98.2%**
  * `Test accuracy by digit:` `['99.7%', '99.6%', '99.2%', '99.1%', '99.3%', '99.3%', '98.3%', '98.8%', '91.0%', '97.8%']`
  * `Test loss by digit:` `['0.014', '0.013', '0.016', '0.017', '0.012', '0.021', '0.057', '0.037', '0.372', '0.083']`
  * Minority digit 8 (504 training samples) scored 91.0% accuracy with an elevated loss of 0.372 (vs ~0.016 for digit 2). Full metrics logged in `experiment_log.md` and `phase1_results.json`.
* **Runs 002–004 (Phase 2 DP-SGD Privacy Sweep):**
  * Evaluated across $\varepsilon \in \{8.0, 5.9, 3.0\}$ with $\delta = 10^{-6}, C = 1.0$.
  * Digit 8 suffered severe accuracy degradation from **91.0%** down to **70.0%** ($\varepsilon \approx 8.0$), **68.8%** ($\varepsilon \approx 5.90$), and **62.7%** ($\varepsilon \approx 3.0$).
  * The Privacy Cost on rare digit 8 (+20.9% to +28.2%) was **~4× to 4.5× larger** than on control digit 2 (+4.7% to +6.3%), demonstrating clear disparate utility impact under vanilla DP-SGD.


