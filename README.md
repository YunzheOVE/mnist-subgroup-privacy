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
└── phase1_results.json   # Output metrics from the latest run (generated on run)
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

## Research Notes & Observations

> *Record your hypotheses, findings, and notes for upcoming phases below.*

### Research Questions
1. **Utility Disparity:** At what threshold of `--keep-eight` does the classification accuracy of digit 8 collapse compared to the overall model accuracy?
2. **Impact of Differential Privacy (Future Phase):** When DP-SGD is added in subsequent phases, does noise addition and gradient clipping harm the rare digit 8 disproportionately compared to digits with full representation?

### Notes Log
* **Baseline Setup:** Phase 1 uses standard cross-entropy loss without class weighting to observe natural minority class decay.
* *[Add additional observations here]*
