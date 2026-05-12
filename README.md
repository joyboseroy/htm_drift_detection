# HTM Drift Detection

Code companion for the paper:

**A Hybrid Framework for Real-Time Data Drift and Anomaly Identification
Using Hierarchical Temporal Memory and Statistical Tests**
Bandyopadhyay S., Bose J., Roychowdhury S.
IJMEMS 2025. [arXiv:2504.18599](https://arxiv.org/abs/2504.18599)

---

## What this is about (plain English)

Machine learning models go stale. The world changes, but your model was
trained on old data. This is called **data drift**.

Most drift detectors raise too many false alarms. The KS test, Wasserstein
distance, and PSI all flag drift constantly on real streaming data, making
them impractical.

This paper proposes a better approach:

1. **HTM layer** — a brain-inspired model that learns normal patterns in
   streaming data and outputs an anomaly score at each time step
2. **SPRT layer** — Sequential Probability Ratio Test, a statistical
   decision framework that converts HTM scores into drift/no-drift decisions
   with controlled false positive and false negative rates

The result: fewer false alarms, no retraining needed, works on streaming
data in real time.

The paper demonstrates this on telecom network KPI data. This repo uses
synthetic data that mimics the same structure.

---

## Architecture

```
Streaming data
      |
      v
  HTM layer (or hybrid statistical detector)
      |
      v  anomaly score per time step
  Rescaling (Section 3.4, equations 3-4)
      |
      v  normalised score
  Binarisation -> Bernoulli sequence {ct}
      |
      v
  SPRT (Theorem 1, equations 1-2)
      |
      v
  Drift detected / No drift / Inconclusive
```

For multivariate data (Section 4): run one HTM per dimension, combine
outputs via neural network or rule-based combiner.

---

## Files

| File | What it does |
|---|---|
| `generate_data.py` | Synthetic telecom time series with injected anomalies |
| `statistical_drift.py` | Hybrid detector: CUSUM + Z-score + IQR + HTM-style likelihood |
| `sprt_detector.py` | SPRT layer from the paper — Theorem 1 implementation |
| `htm_anomaly_detection.ipynb` | Full walkthrough notebook |

---

## Quickstart

```bash
pip install -r requirements.txt

# Generate synthetic data
python generate_data.py

# Run hybrid detector + SPRT
python sprt_detector.py

# Or open the notebook
jupyter notebook htm_anomaly_detection.ipynb
```

---

## Key Results from the Paper

The proposed HTM-SPRT approach vs competing methods on streaming data
with periodically varying mean (Figure 8, paper):

| Method | False Positives | Practical? |
|---|---|---|
| KS test | Very high | No |
| Wasserstein distance | Very high | No |
| PSI | Very low (misses drift) | No |
| HTM + SPRT (proposed) | Balanced | Yes |

---

## SPRT Parameters (from Table 1, paper)

| Scenario | Window | bin_threshold | p_null | p_alt | alpha | beta |
|---|---|---|---|---|---|---|
| Abrupt shift | 15 | 0.65 | 0.45 | 0.50 | 0.05 | 0.005 |
| Slow drift | 35 | 0.65 | 0.45 | 0.50 | 0.05 | 0.005 |
| Periodic mean | 25 | 0.65 | 0.45 | 0.50 | 0.05 | 0.005 |

---

## Note on htm.core

The full HTM pipeline requires `htm.core`:

```bash
pip install htm.core
```

This may require C++ build tools and can be difficult to install.
The notebook (Part 2) and `statistical_drift.py` work without it,
using a hybrid statistical detector as a drop-in replacement for
the HTM anomaly score.

---

## Citation

```bibtex
@article{bandyopadhyay2025hybrid,
  title={A Hybrid Framework for Real-Time Data Drift and Anomaly
         Identification Using Hierarchical Temporal Memory and
         Statistical Tests},
  author={Bandyopadhyay, Subhadip and Bose, Joy and
          Roychowdhury, Sujoy},
  journal={International Journal of Mathematical, Engineering and
           Management Sciences},
  volume={10},
  number={3},
  year={2025},
  doi={10.48550/arXiv.2504.18599}
}
```

---

## Authors

**Subhadip Bandyopadhyay, Joy Bose, Sujoy Roy Chowdhury**
Global AI Accelerator, Ericsson, Bangalore

[Joy Bose on LinkedIn](https://linkedin.com/in/joyboseroy) |
[Google Scholar](https://scholar.google.com/citations?user=1E0YgA4AAAAJ) |
[Personal site](https://joyboseroy.github.io)
