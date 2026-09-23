# Research Plan: Subgroup Privacy & Gradient Alignment

A 4-phase experimental roadmap investigating the impact of Differential Privacy (DP-SGD) and adaptive gradient clipping on rare subgroups using imbalanced MNIST.

---

## Phase 1: The Imbalanced Non-Private Baseline

* **Code:** Load MNIST with `torchvision`. Keep all training digits except retain digit **8** with **9% probability**, matching the paper; keep the test set balanced. *(Treat 5% as an optional stress test).*
* **Train:** Train a non-private CNN with standard SGD and no clipping or noise. Use the same model and data settings for the private comparisons; record differences from the paper.
* **Record:** Save training counts, test accuracy, and loss for digits 0–9 (especially digits **2** and **8**), across several random seeds.
* **Expected Result:** Treat accuracy as a measurement, not a target. In the paper’s 9% setup, digit 8 scored **84.3%** without privacy; a different model or 5% retention may change this.

---

## Phase 2: Measure the Impact of Vanilla DP-SGD

* **Code:** Use the same CNN and imbalance split with an Opacus-compatible model and private data loader; document any training changes.
* **Train & Sweep:** Reproduce the paper’s $\varepsilon \approx 5.90$ MNIST point first, then vary $\varepsilon$ if time permits. Repeat each setting across seeds and record achieved $\varepsilon$, $\delta$, noise multiplier $\sigma$, clipping threshold $C$, and privacy accountant used:
  1. **Published MNIST setting:** $\varepsilon \approx 5.90$, $\delta = 10^{-6}$
  2. **Looser privacy:** Target $\varepsilon \approx 8$ (if time permits)
  3. **Tighter privacy:** Target $\varepsilon \approx 3$ ($\varepsilon \approx 1$ is optional)
* **Record:** Compare digit **8** with digit **2**, showing all-digit and overall accuracy. Compute each digit’s privacy cost relative to the non-private baseline:
  $$\text{Privacy Cost} = \text{Acc}_{\text{non-private}} - \text{Acc}_{\text{DP}}$$
* **Expected Result:** Test whether the rare digit suffers a disproportionately larger privacy cost; do not assume a specific collapse. The paper reported **26.3%** accuracy for digit 8 under its DP-SGD setting.

---

## Phase 3: Diagnose Clipping and Gradient Direction

* **Code:** Log per-class pre-clipping gradient norms, clipping rates, and shrinkage. Also compare each batch’s mean gradient before and after clipping.
* **Record:** Plot clipping rates by digit and cosine similarity between the unclipped and clipped batch updates over the course of training.
* **Interpretation:** Larger norms or higher clipping rates alone do not prove gradient misalignment. A genuine shift in aggregate update direction is the relevant evidence; test rather than presume it.

---

## Phase 4: Test DPSGD-Global-Adapt

* **Action:** Read Section 5 of the paper and the official MNIST experiment code now, before implementation.
* **Implement:** Global-Adapt scales gradients below an upper bound $Z$, clips outliers to $C$, and privately updates $Z$ using a noisy count. Ensure privacy is accounted for across both mechanisms.
* **Re-run:** Compare at matched $(\varepsilon, \delta)$, data, model, and seeds. Report digit-8 and digit-2 privacy costs, overall accuracy, and hyperparameter tuning used by each method.
