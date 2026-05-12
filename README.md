# From Forecast Accuracy to Inventory Performance

**A Decision-Oriented Evaluation of Statistical and Machine Learning Models in Retail Replenishment**

Master's Thesis · SGH Warsaw School of Economics · Big Data and Advanced Analytics · 2026

---

## Overview

This repository contains the code, thesis document, and defence presentation for an empirical study comparing local statistical (SARIMA) and global machine learning (Random Forest, LightGBM) demand forecasting models in a periodic-review `(R, S)` inventory system.

The study uses the **Walmart M5 dataset** (FOODS category, store `CA_1`, 1,437 SKU-store series) and a **rolling-origin evaluation** to test four hypotheses about how forecasting model choice affects downstream inventory performance.

### Key finding

> **Forecast accuracy is necessary but not sufficient.** Random Forest achieves the lowest WRMSSE (0.668 vs SARIMA's 0.739, a 9.6% improvement) and translates this into 21.6% lower holding cost — but at the price of 0.87 percentage points lower fill rate. The model that minimises forecast error does not maximise service. Model selection is a strategic supply-chain decision, not a pure data-science optimisation.

---

## Research design

A **dual-layer evaluation framework** measures both the accuracy of the forecasts and the operational consequences of feeding those forecasts into a deterministic `(R, S)` replenishment policy.

| Layer | Question | Metric |
|---|---|---|
| **Layer 1 — Accuracy** | Which model produces the most accurate point forecast at the protection-period horizon? | WRMSSE |
| **Layer 2 — Operational** | Which model produces the best cost–service trade-off when its forecast is used to set order-up-to level *S* and safety stock? | Holding cost (lower is better) and fill rate β (higher is better) |

### Models compared

| Paradigm | Model | Notes |
|---|---|---|
| Local statistical (baseline) | **SARIMA** | One model per series, `auto.arima` selection, univariate |
| Global ML (challenger) | **Random Forest** | Cross-learning across all series; lag, price, calendar features |
| Global ML (challenger) | **LightGBM** | Gradient-boosted trees, same feature set as RF |

### Hypotheses

| | Hypothesis | Verdict |
|---|---|---|
| **H1** | Global ML achieves lower WRMSSE than local SARIMA at the protection-period horizon | ✓ Supported |
| **H2** | Accuracy gains do not translate proportionally into both cost AND service | ✓ Supported |
| **H3** | Operational benefit of ML is concentrated in high-volume Class A items | ✓ Supported |
| **H4** | Cost advantage of ML widens non-linearly with target service level α | ✓ Supported |

### Key results

**Layer 1 — WRMSSE (mean across 3 rolling-origin splits):**

| Model | WRMSSE | Std | vs SARIMA |
|---|---|---|---|
| Random Forest | 0.668 | 0.029 | −9.6% |
| LightGBM | 0.686 | 0.019 | −7.2% |
| SARIMA | 0.739 | 0.018 | — |

**Layer 2 — Operational (aggregate, α = 95%):**

| Model | Holding cost | Fill rate β | vs SARIMA cost |
|---|---|---|---|
| Random Forest | 5,932 | 97.36% | −21.6% |
| LightGBM | 6,464 | 97.89% | −14.6% |
| SARIMA | 7,570 | 98.23% | — |

**ABC segment cost gaps (RF vs SARIMA):**

| Segment | SKUs | Demand share | Cost gap |
|---|---|---|---|
| Class A | 540 | 80% | **−25.7%** |
| Class B | 425 | 15% | −12.8% |
| Class C | 472 | 5% | −14.7% |

---

## Repository structure

```
.
├── README.md                                       # This file
├── thesis/
│   └── THAI_QUOC_DAT_THESIS.docx                   # Final thesis document
├── presentation/
│   ├── THAI_QUOC_DAT_DEFENCE_FINAL.pptx            # Defence slides (editable)
│   └── THAI_QUOC_DAT_DEFENCE_FINAL.pdf             # Defence slides (submission PDF)
├── notebooks/
│   ├── 01_forecasting.ipynb                        # SARIMA, RF, LightGBM forecasts
│   └── 02_inventory_simulation.ipynb               # (R, S) policy, cost/service KPIs
├── data/
│   └── README.md                                   # M5 dataset download instructions
└── requirements.txt                                # Python dependencies
```

---

## Getting started

### Prerequisites

- Python 3.9+
- ~8 GB RAM (M5 dataset and rolling-origin loops are memory-intensive)
- The Walmart M5 dataset from the [M5 Forecasting Accuracy Kaggle competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy)

### Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv .venv
source .venv/bin/activate          # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Data download

Place these four files in `data/raw/`:

- `calendar.csv`
- `sales_train_evaluation.csv`
- `sell_prices.csv`
- `sample_submission.csv`

You can download them from Kaggle:

```bash
kaggle competitions download -c m5-forecasting-accuracy
unzip m5-forecasting-accuracy.zip -d data/raw/
```

### Reproduce the results

```bash
jupyter notebook
```

Then run notebooks in order:

1. **`01_forecasting.ipynb`** — fits SARIMA, RF, LightGBM under rolling-origin validation, saves forecast outputs and WRMSSE.
2. **`02_inventory_simulation.ipynb`** — runs the `(R, S)` simulation, computes holding cost and fill rate, generates the z-sweep for the Pareto frontier.

All results in the thesis and presentation can be reproduced with:

```python
RANDOM_SEED = 42
N_SPLITS = 3        # rolling-origin splits
HORIZON = 2         # protection-period (weeks)
TARGET_ALPHA = 0.95 # baseline service level
Z_SWEEP = np.linspace(0.0, 4.0, 15)  # for Pareto frontier
```

---

## Methodology highlights

### Inventory policy

Periodic-review **`(R, S)`** policy with:

- Review period `R = 1 week`
- Lead time `L = 1 week`
- Protection period `h = R + L = 2 weeks`
- Lost sales (FMCG context — no backorders)
- Holding cost computed per period; shortage cost not monetised but service level reported via fill rate β

Order-up-to level:

```
S = μ_d · h + z · σ_d · √h
```

where `σ_d` is proxied by **training-period RMSE** (avoids test leakage) and `z` is the safety factor corresponding to target service level α.

### Validation

- **Rolling-origin evaluation** with `N_SPLITS = 3` to ensure robust generalisation estimates
- **Equal evaluation universe** across models (1,437 series, zero SARIMA fallback)
- **Reproducibility:** `random_state = 42` throughout; deterministic `auto.arima` selection

### ABC classification

SKUs sorted by total demand volume, partitioned by cumulative demand share:

- **Class A:** top 80% of demand (540 SKUs)
- **Class B:** next 15% (425 SKUs)
- **Class C:** bottom 5% (472 SKUs)

---

## Key references

The full bibliography (40+ entries) is in the thesis. The five most cited works:

- **Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022a).** The M5 accuracy competition: Results, findings and conclusions. *International Journal of Forecasting*, 38(4), 1346–1364.
- **Kourentzes, N., Trapero, J. R., & Barrow, D. K. (2020).** Optimising forecasting models for inventory planning. *International Journal of Production Economics*, 225, 107597.
- **Silver, E. A., Pyke, D. F., & Thomas, D. J. (2017).** *Inventory and Production Management in Supply Chains* (4th ed.). CRC Press.
- **Vandeput, N. (2021).** *Data Science for Supply Chain Forecast* (2nd ed.). De Gruyter.
- **Petropoulos, F., et al. (2022).** Forecasting: Theory and practice. *International Journal of Forecasting*, 38(3), 705–871.

---

## Author and supervisor

| | |
|---|---|
| **Author** | Thai Quoc Dat (ID 140098) |
| **Programme** | Big Data and Advanced Analytics, Master's level |
| **Institution** | SGH Warsaw School of Economics |
| **Supervisor** | Mateusz Zawisza, PhD — Department of Decision Support and Analysis |
| **Defence** | July 2026 |

---

## License

This work is submitted as a Master's thesis at SGH Warsaw School of Economics. The code is released under the **MIT License** for academic and non-commercial use. The thesis text and presentation remain the intellectual property of the author and SGH.

The Walmart M5 dataset is distributed by Kaggle under the terms of the [M5 competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy/rules).

---

## Citation

If you use this work, please cite:

```bibtex
@mastersthesis{thai2026forecast,
  title  = {From Forecast Accuracy to Inventory Performance:
            A Decision-Oriented Evaluation of Statistical and Machine
            Learning Models in Retail Replenishment},
  author = {Thai, Quoc Dat},
  school = {SGH Warsaw School of Economics},
  year   = {2026},
  type   = {Master's Thesis},
  note   = {Supervisor: Mateusz Zawisza, PhD}
}
```
