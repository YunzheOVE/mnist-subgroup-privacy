# Experiment Log

Record results from each experiment run here to track how configuration changes impact performance (especially on the rare digit 8).

---

## Run Summary Table

| Run # | Date | Keep-8 Ratio | Epochs | Seed | Overall Acc | Digit 8 Acc | Digit 8 Loss | Notes / Comments |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `001` | 2026-09-24 | 0.09 (9%) | 8 | 42 | 98.2% | 91.0% | 0.372 | Phase 1 baseline (CUDA, RTX 4070 SUPER) |
| | | | | | | | | |

---

## Detailed Run Notes

### Run 001 (Baseline)
* **Command:** `python phase1_mnist.py --keep-eight 0.09 --epochs 8 --seed 42`
* **Hardware:** CUDA (NVIDIA GeForce RTX 4070 SUPER)
* **Purpose:** Initial non-private baseline with 9% digit 8 retention (504 samples vs ~5,900 for others).
* **Summary Metrics:**
  * **Overall Accuracy:** 98.2%
  * **Test accuracy by digit:** `['99.7%', '99.6%', '99.2%', '99.1%', '99.3%', '99.3%', '98.3%', '98.8%', '91.0%', '97.8%']`
  * **Test loss by digit:** `['0.014', '0.013', '0.016', '0.017', '0.012', '0.021', '0.057', '0.037', '0.372', '0.083']`
  * **Per-Digit Breakdown Table:**

| Digit | Training Count | Test Count | Test Accuracy | Test Loss | Note |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | 5,923 | 980 | 99.7% | 0.014 | Majority |
| **1** | 6,742 | 1,135 | 99.6% | 0.013 | Majority |
| **2** | 5,958 | 1,032 | 99.2% | 0.016 | Majority / Control |
| **3** | 6,131 | 1,010 | 99.1% | 0.017 | Majority |
| **4** | 5,842 | 982 | 99.3% | 0.012 | Majority |
| **5** | 5,421 | 892 | 99.3% | 0.021 | Majority |
| **6** | 5,918 | 958 | 98.3% | 0.057 | Majority |
| **7** | 6,265 | 1,028 | 98.8% | 0.037 | Majority |
| **8** | **504** | 974 | **91.0%** | **0.372** | **Rare Subgroup (9%)** |
| **9** | 5,949 | 1,009 | 97.8% | 0.083 | Majority |

* **Observations:**
  * Despite extreme imbalance (digit 8 having under 10% of standard data), the non-private CNN achieves 91.0% accuracy on digit 8.
  * The disparity is clearly visible in the loss: digit 8 loss (0.372) is ~23x higher than digit 2 loss (0.016) and >4x higher than any other class.
  * This establishes the unclipped, non-private baseline against which DP-SGD degradation in Phase 2 will be measured.

---

*(Append future runs below)*
