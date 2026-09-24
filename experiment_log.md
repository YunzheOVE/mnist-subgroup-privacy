# Experiment Log

Record results from each experiment run here to track how configuration changes impact performance (especially on the rare digit 8).

---

## Run Summary Table

| Run # | Date | Keep-8 Ratio | Epochs | Seed | Overall Acc | Digit 8 Acc | Digit 8 Loss | Notes / Comments |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `001` | 2026-09-24 | 0.09 (9%) | 8 | 42 | 98.2% | 91.0% | 0.372 | Phase 1 baseline (Non-private) |
| `002` | 2026-09-24 | 0.09 (9%) | 8 | 42 | 92.5% | 68.8% | 2.646 | Phase 2 DP-SGD benchmark ($\varepsilon \approx 5.90, \delta = 10^{-6}, \sigma = 0.5484, C = 1.0$) |
| `003` | 2026-09-24 | 0.09 (9%) | 8 | 42 | 91.3% | 62.7% | 3.471 | Phase 2 DP-SGD tighter privacy ($\varepsilon \approx 3.0, \delta = 10^{-6}, \sigma = 0.6635, C = 1.0$) |
| `004` | 2026-09-24 | 0.09 (9%) | 8 | 42 | 92.9% | 70.0% | 2.415 | Phase 2 DP-SGD looser privacy ($\varepsilon \approx 8.0, \delta = 10^{-6}, \sigma = 0.5038, C = 1.0$) |

---

## Detailed Run Notes

### Run 001 (Baseline - Non-Private)
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

### Run 002 (Phase 2: DP-SGD Benchmark $\varepsilon \approx 5.90$)
* **Command:** `python phase2_dpsgd.py`
* **Privacy Guarantee:** $\varepsilon = 5.8975, \delta = 10^{-6}$ (PRV accountant, $\sigma = 0.5484, C = 1.0$)
* **Summary Metrics:**
  * **Overall Accuracy:** 92.5% (down 5.7% from non-private 98.2%)
  * **Test accuracy by digit:** `['98.1%', '98.4%', '93.5%', '96.1%', '96.3%', '92.4%', '96.2%', '94.5%', '68.8%', '89.4%']`
  * **Test loss by digit:** `['0.155', '0.132', '0.564', '0.267', '0.195', '0.444', '0.219', '0.517', '2.646', '0.771']`
  * **Privacy Cost by Digit ($\text{Acc}_{\text{non-private}} - \text{Acc}_{\text{DP}}$):**

| Digit | Non-Private Acc | DP-SGD Acc | Privacy Cost | Role |
| :---: | :---: | :---: | :---: | :--- |
| **0** | 99.7% | 98.1% | +1.6% | Majority |
| **1** | 99.6% | 98.4% | +1.2% | Majority |
| **2** | 99.2% | 93.5% | **+5.7%** | **Control Digit** |
| **3** | 99.1% | 96.1% | +3.0% | Majority |
| **4** | 99.3% | 96.3% | +3.0% | Majority |
| **5** | 99.3% | 92.4% | +7.0% | Majority |
| **6** | 98.3% | 96.2% | +2.1% | Majority |
| **7** | 98.8% | 94.5% | +4.4% | Majority |
| **8** | 91.0% | 68.8% | **+22.2%** | **Rare Subgroup (9%)** |
| **9** | 97.8% | 89.4% | +8.4% | Majority |

* **Key Finding:**
  * Digit 8 suffered a **+22.2% privacy cost**, which is **~3.9× higher** than Control Digit 2 (+5.7%) and **~7× higher** than the average majority class (~3.1%).
  * Digit 8 test loss exploded to **2.646** (compared to 0.564 for digit 2).

---

### Run 003 (Phase 2: DP-SGD Tighter Privacy $\varepsilon \approx 3.0$)
* **Command:** `python phase2_dpsgd.py --target-epsilon 3.0`
* **Privacy Guarantee:** $\varepsilon = 2.9909, \delta = 10^{-6}$ (PRV accountant, $\sigma = 0.6635, C = 1.0$)
* **Summary Metrics:**
  * **Overall Accuracy:** 91.3% (down 6.9% from non-private 98.2%)
  * **Test accuracy by digit:** `['97.4%', '98.3%', '92.9%', '95.1%', '95.4%', '91.5%', '95.9%', '93.4%', '62.7%', '88.5%']`
  * **Test loss by digit:** `['0.183', '0.138', '0.624', '0.355', '0.256', '0.518', '0.290', '0.650', '3.471', '0.866']`
  * **Privacy Cost by Digit ($\text{Acc}_{\text{non-private}} - \text{Acc}_{\text{DP}}$):**

| Digit | Non-Private Acc | DP-SGD Acc | Privacy Cost | Role |
| :---: | :---: | :---: | :---: | :--- |
| **0** | 99.7% | 97.4% | +2.2% | Majority |
| **1** | 99.6% | 98.3% | +1.3% | Majority |
| **2** | 99.2% | 92.9% | **+6.3%** | **Control Digit** |
| **3** | 99.1% | 95.1% | +4.0% | Majority |
| **4** | 99.3% | 95.4% | +3.9% | Majority |
| **5** | 99.3% | 91.5% | +7.8% | Majority |
| **6** | 98.3% | 95.9% | +2.4% | Majority |
| **7** | 98.8% | 93.4% | +5.4% | Majority |
| **8** | 91.0% | 62.7% | **+28.2%** | **Rare Subgroup (9%)** |
| **9** | 97.8% | 88.5% | +9.3% | Majority |

* **Key Finding:**
  * With tighter privacy (higher noise $\sigma = 0.6635$), Digit 8 accuracy dropped sharply to **62.7%** (**+28.2% privacy cost**).
  * Control Digit 2 remained resilient at 92.9% (+6.3% privacy cost), showing that tighter privacy penalizes the minority class disproportionately (**4.5× disparity**).

---

### Run 004 (Phase 2: DP-SGD Looser Privacy $\varepsilon \approx 8.0$)
* **Command:** `python phase2_dpsgd.py --target-epsilon 8.0`
* **Privacy Guarantee:** $\varepsilon = 7.9946, \delta = 10^{-6}$ (PRV accountant, $\sigma = 0.5038, C = 1.0$)
* **Summary Metrics:**
  * **Overall Accuracy:** 92.9% (down 5.3% from non-private 98.2%)
  * **Test accuracy by digit:** `['98.1%', '98.3%', '94.5%', '96.4%', '96.6%', '92.3%', '97.3%', '94.8%', '70.0%', '89.6%']`
  * **Test loss by digit:** `['0.156', '0.128', '0.486', '0.227', '0.171', '0.465', '0.181', '0.461', '2.415', '0.720']`
  * **Privacy Cost by Digit ($\text{Acc}_{\text{non-private}} - \text{Acc}_{\text{DP}}$):**

| Digit | Non-Private Acc | DP-SGD Acc | Privacy Cost | Role |
| :---: | :---: | :---: | :---: | :--- |
| **0** | 99.7% | 98.1% | +1.6% | Majority |
| **1** | 99.6% | 98.3% | +1.3% | Majority |
| **2** | 99.2% | 94.5% | **+4.7%** | **Control Digit** |
| **3** | 99.1% | 96.4% | +2.7% | Majority |
| **4** | 99.3% | 96.6% | +2.6% | Majority |
| **5** | 99.3% | 92.3% | +7.1% | Majority |
| **6** | 98.3% | 97.3% | +1.0% | Majority |
| **7** | 98.8% | 94.8% | +4.0% | Majority |
| **8** | 91.0% | 70.0% | **+20.9%** | **Rare Subgroup (9%)** |
| **9** | 97.8% | 89.6% | +8.2% | Majority |

---

## Phase 2 Privacy Sweep Comparison Summary

Across all privacy regimes, the rare subgroup (Digit 8) consistently bears the vast majority of the privacy penalty:

| Privacy Level | Target $\varepsilon$ | Achieved $\varepsilon$ | Noise $\sigma$ | Overall Acc | Control (Digit 2) Acc | Rare (Digit 8) Acc | Digit 8 Privacy Cost | Digit 2 Privacy Cost | Disparity Ratio |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Non-Private** | $\infty$ | — | $0.0$ | **98.2%** | **99.2%** | **91.0%** | *Baseline* | *Baseline* | 1.0× |
| **Looser** | $8.0$ | 7.99 | 0.5038 | 92.9% | 94.5% | 70.0% | **+20.9%** | +4.7% | **4.4×** |
| **Published** | $5.9$ | 5.90 | 0.5484 | 92.5% | 93.5% | 68.8% | **+22.2%** | +5.7% | **3.9×** |
| **Tighter** | $3.0$ | 2.99 | 0.6635 | 91.3% | 92.9% | 62.7% | **+28.2%** | +6.3% | **4.5×** |

---

*(Append future runs below)*
