# Heating Demand Modeling Framework

A **policy-centric and physics-consistent framework** for modeling annual residential space-heating demand using RECS 2020 microdata. This framework provides:

- **Accurate predictions** with technology-aware models and physics constraints
- **Robust evaluation** via nested cross-validation without data leakage
- **Policy-relevant analysis** measuring how survey weighting affects retrofit targeting
- **Defensible uncertainty** through replicate-weight jackknife estimation

---

## 🎯 Research Objectives

### Hypotheses

| Hypothesis | Statement | Evidence |
|------------|-----------|----------|
| **H1** | Technology-split models outperform monolithic models | Δ wRMSE = +5,158 kBTU [4,747, 5,569], improvement in 4/5 folds |
| **H2** | Survey weighting changes retrofit targeting composition | Equal-budget Jaccard = 0.36 (substantial list difference) |
| **H3** | Nested CV + replicate weights yield defensible uncertainty | Refit-sensitivity gap < 10% of SE |

### Key Contributions

1. **Technology-aware structure** with physics-motivated constraints (monotonic HDD/Area effects)
2. **Nested CV (5×3)** with stratification by Division × Technology × HDD to eliminate tuning leakage
3. **Survey weighting impact analysis** on retrofit targeting with Jaccard CIs via replicate-weight jackknife
4. **Equity audit** showing systematic underprediction in cold climates and single-family homes
5. **Isotonic post-calibration** to reduce tail underprediction

---

## 📊 Key Results

### Model Performance (Outer-fold, Weighted)

| Metric | Estimate | 95% CI |
|--------|----------|--------|
| wRMSE | 25,197 kBTU | [24,609, 25,786] |
| wMAE | 15,854 kBTU | [15,591, 16,117] |
| wR² | 0.470 | [0.452, 0.487] |
| wBias | -5,977 kBTU | [-6,324, -5,630] |

### H1: Split vs Monolithic Comparison

| Metric | Monolithic | Split | Δ (95% CI) |
|--------|------------|-------|------------|
| wRMSE | 30,355 | 25,197 | +5,158 [4,747, 5,569] |
| wR² | 0.230 | 0.470 | +0.239 [0.220, 0.258] |

### H2: Policy Targeting (Weighted vs Unweighted)

| Score | Jaccard (Percentile) | Jaccard (Equal Budget) | 95% CI |
|-------|---------------------|----------------------|--------|
| High-Use | 0.749 | 0.364 | [0.687, 0.810] |
| High-Intensity | 0.890 | 0.327 | [0.842, 0.937] |
| Excess-Demand | 0.891 | 0.596 | [0.855, 0.927] |

### Physics Diagnostics by Technology

| Technology | n | Target CV (%) | Weighted r | Cold Bias (%) | Tail Bias (%) |
|------------|---|--------------|------------|---------------|---------------|
| Combustion | 11,852 | 76.5 | 0.644 | -21.2 | -46.9 |
| Heat Pump | 2,217 | 113.2 | 0.543 | +26.2 | -30.2 |
| Resistance | 1,376 | 127.0 | 0.563 | +27.0 | -24.8 |
| Hybrid | 2,186 | 122.5 | 0.475 | +23.9 | -44.0 |

---

## 📁 Project Structure

```
heating-demand-modeling/
├── configs/
│   └── default_config.yaml          # Configuration file
├── data/
│   ├── raw/                          # Raw RECS 2020 data
│   └── processed/                    # Processed datasets
├── src/
│   ├── data/
│   │   ├── loader.py                 # RECS data loader
│   │   └── preprocessor.py           # Technology grouping & preprocessing
│   ├── features/
│   │   └── builder.py                # Feature engineering (leakage-safe)
│   ├── models/
│   │   ├── baselines.py              # Physics-based baselines
│   │   └── main_models.py            # LightGBM (Tweedie) and EBM models
│   ├── evaluation/
│   │   ├── metrics.py                # Weighted metrics & physics diagnostics
│   │   ├── nested_cv.py              # Nested cross-validation
│   │   └── model_comparison.py       # Split vs Monolithic comparison
│   ├── policy/
│   │   └── targeting.py              # Policy targeting & overlap metrics
│   ├── uncertainty/
│   │   └── jackknife.py              # Jackknife uncertainty & delta CIs
│   ├── visualization/
│   │   ├── plots.py                  # Publication-quality figures
│   │   └── tables.py                 # Publication-quality tables
│   └── utils/
│       └── helpers.py                # Utility functions
├── outputs/
│   ├── figures/                      # Generated figures (8 files)
│   ├── tables/                       # Generated tables (30+ files)
│   └── models/                       # Saved models
├── tests/
│   └── test_framework.py             # Unit tests (16 tests)
├── run_analysis.py                   # Main analysis script
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Running the Analysis

```bash
# Full analysis with nested CV (5 outer folds)
python run_analysis.py --n-outer-folds 5

# Quick test (skip nested CV)
python run_analysis.py --skip-cv

# Custom configuration
python run_analysis.py --config configs/custom_config.yaml --output results/
```

### Running Tests

```bash
python -m pytest tests/test_framework.py -v
```

---

## 📊 Data

### RECS 2020 Microdata

- **Source:** EIA RECS 2020 public-use microdata (Final Release, version 2023-03)
- **Records:** 17,631 households (after filtering no_heating)
- **Target:** `TOTALBTUSPH` (annual space heating energy in kBTU)
- **Target Distribution:** Heavy-tail (CV = 76-127% by technology)

### Survey Weights

| Weight | Description | Usage |
|--------|-------------|-------|
| `NWEIGHT` | Final survey weight | Training, Evaluation, Targeting |
| `NWEIGHT1`-`NWEIGHT60` | Jackknife replicate weights | Uncertainty estimation |

---

## 🔬 Methodology

### Technology Grouping

Mutually exclusive groups based on primary equipment (`EQUIPM`):

| Group | EQUIPM Codes | Description | n | Pop Share |
|-------|--------------|-------------|---|-----------|
| **Combustion** | 2,4,7,8,9 | Furnace, boiler, wood stove | 11,852 | 63.3% |
| **Heat Pump** | 5,11 | Central/ductless heat pump | 2,217 | 13.8% |
| **Resistance** | 6,10 | Baseboard, portable heaters | 1,376 | 9.1% |
| **Hybrid/Ambiguous** | 3,12 | Mixed systems | 2,186 | 13.9% |

### Nested Cross-Validation (5×3)

```
Outer Loop (5 folds) ─── Evaluate generalization
    │
    └── Inner Loop (3 folds) ─── Tune hyperparameters
            │
            └── Random Search (20 iterations)
```

**Stratification:** Census Division × Technology Group × HDD bins (3 bins)

### Model Architecture

| Component | Implementation |
|-----------|----------------|
| **Main Model** | LightGBM with Tweedie objective (power=1.5) |
| **Constraints** | Monotonic on HDD (+) and Area (+) |
| **Calibration** | Isotonic regression (post-hoc) |
| **Baseline** | Ridge regression (HDD×Area + Area) |
| **Interpretability** | Explainable Boosting Machine (EBM) |

### Policy Scores

| Score | Formula | Interpretation |
|-------|---------|----------------|
| **High-Use** | Ŷ | Absolute predicted consumption |
| **High-Intensity** | Ŷ / Area | Energy use per sq ft (kBTU/ft²) |
| **Excess-Demand** | Ŷ - Ŷ_baseline | Above physics-expected consumption |

### Uncertainty Quantification

**Jackknife variance (RECS SDR method):**
```
Var(θ) = (n-1)/n × Σᵢ(θᵢ - θ̄)²   where n=60
```

**95% Confidence Interval:**
```
CI = θ ± 1.96 × SE(θ)
```

---

## 📈 Outputs

### Tables (30+ files)

| Table | Description |
|-------|-------------|
| `table1_tech_group_descriptives.csv` | Technology group descriptives (n, shares, means) |
| `table2_cv_performance.csv` | Cross-validation performance by fold |
| `table2b_h1_split_vs_mono.csv` | H1 comparison: Split vs Monolithic |
| `table2c_h1_delta_ci.csv` | **NEW:** Delta metrics with replicate-weight CIs |
| `table3_uncertainty.csv` | Uncertainty estimates for all metrics |
| `table_physics_diagnostics.csv` | Physics diagnostics by technology |
| `policy_*_summary.csv` | Policy targeting summaries with Jaccard CIs |
| `policy_*_income.csv` | Composition shifts by income |
| `equity_by_*.csv` | Error equity analysis by group |
| `diagnostics_bias_by_hdd.csv` | Bias by HDD bins |
| `appendix_technical_documentation.md` | Technical appendix |
| `sensitivity_notes.txt` | Sensitivity analysis documentation |

### Figures (8 files)

| Figure | Description |
|--------|-------------|
| `fig1_workflow.png` | Analysis workflow diagram |
| `fig2_pred_vs_obs.png` | Predicted vs Observed (overall) |
| `fig2b_pred_vs_obs_by_tech.png` | Predicted vs Observed by technology |
| `fig3_composition_shift.png` | Composition shift (weighted vs unweighted) |
| `fig4_h1_split_vs_mono.png` | H1: Residual comparison Split vs Mono |
| `fig5_residual_vs_hdd.png` | Residual bias vs HDD by technology |
| `error_equity.png` | Error equity analysis |
| `cv_results.png` | Cross-validation results |

### Q&A Documentation

| File | Description |
|------|-------------|
| `QA_195_questions_Persian.txt` | 195 defense questions with answers (Persian) |
| `QA_195_questions_English.txt` | 195 defense questions with answers (English) |

---

## 🔒 Leakage Prevention

### Blacklisted Features

Features that are direct functions of energy use:

```python
LEAKAGE_BLACKLIST = [
    'TOTALBTUSPH',           # Target variable
    'DOLLARSPH', 'DOLLAREL', # Energy expenditure
    'BTUEL', 'BTUNG',        # End-use components
    'TOTALBTU', 'TOTALDOL',  # Totals
    'NWEIGHT', 'DOEID'       # Survey metadata
]
```

### CV-Internal Preprocessing

All transformations fitted inside inner loop only:
- ✅ Missingness imputation (median from train)
- ✅ Feature encoding (fitted on train)
- ✅ Feature scaling (mean/std from train)
- ✅ Isotonic calibration (fitted on train predictions)

---

## 📚 API Reference

### Complete Pipeline

```python
from src.data.loader import RECSDataLoader
from src.data.preprocessor import RECSPreprocessor
from src.features.builder import FeatureBuilder
from src.models.main_models import LightGBMHeatingModel
from src.evaluation.nested_cv import NestedCrossValidator
from src.policy.targeting import PolicyTargeting
from src.uncertainty.jackknife import JackknifeUncertainty

# 1. Load data
loader = RECSDataLoader('data/raw/recs2020_public_v7.csv')
df = loader.load()

# 2. Preprocess
preprocessor = RECSPreprocessor()
df = preprocessor.fit_transform(df)

# 3. Build features
builder = FeatureBuilder(covid_control_mode='proxy')
X = builder.fit_transform(df)

# 4. Run nested CV
cv = NestedCrossValidator(outer_folds=5, inner_folds=3)
result = cv.run(X, y, weights, model_factory, param_grid)

# 5. Policy analysis
targeting = PolicyTargeting(target_percentile=90)
policy_results = targeting.run_full_analysis(
    y_pred, y_baseline, y_true, weights, metadata
)

# 6. Uncertainty quantification
jackknife = JackknifeUncertainty(n_replicates=60)
uncertainty = jackknife.compute_all_metrics_uncertainty(
    y_true, y_pred, weights, replicate_weights
)

# 7. Delta CIs (Split vs Monolithic)
delta_ci = jackknife.compute_delta_metrics_ci(
    y_true, y_pred_mono, y_pred_split, weights, replicate_weights
)
```

---

## ⚠️ Known Limitations

1. **Tail Underprediction:** Model systematically underpredicts high consumption (tail bias -25% to -47%). Policy targeting may miss some high-use households.

2. **Cold Climate Bias:** Underprediction increases with HDD (-18% for HDD 6-8k, -20% for HDD 8k+).

3. **Electric Subgroup R²:** Low within-group wR² (0.06-0.08) for heat pump/resistance groups due to high target variance (CV > 100%). Weighted correlation (r ≈ 0.5) indicates predictive signal exists.

4. **2020 Anomaly:** Results reflect COVID-era occupancy patterns. Recalibration may be needed for post-pandemic applications.

5. **End-Use Estimation Noise:** TOTALBTUSPH is modeled by EIA (not metered), introducing measurement error that limits achievable R² to ~0.6.

---

## 🧪 Sensitivity Analyses

| Analysis | Result |
|----------|--------|
| **Remove COVID controls** | wRMSE +2% worse |
| **Remove monotonic constraints** | Physics violations increase |
| **Change loss to MAE** | Tail improves, overall RMSE worse |
| **Remove calibration** | Equal-budget Jaccard changes <5% |
| **Remove hybrids** | H1/H2 messages unchanged |

---

## 📋 Reproducibility Checklist

- [x] Exact technology grouping rules (EQUIPM codes)
- [x] Feature list + leakage blacklist
- [x] CV stratification scheme (Division × Tech × HDD)
- [x] Policy score definitions
- [x] Jackknife variance formula (RECS SDR)
- [x] Random seeds (42) and library versioning
- [x] Fold assignment reproducible with same seed
- [x] All preprocessing inside CV loop

---

## 📖 Citation

If you use this framework, please cite:

```bibtex
@software{heating_demand_framework,
  title = {Policy-Centric Heating Demand Modeling Framework for RECS 2020},
  year = {2024},
  note = {Technology-aware models with physics constraints and survey weighting analysis},
  url = {https://github.com/your-repo/heating-demand-modeling}
}
```

---

## 📄 License

MIT License

---

## 📞 Contact

For questions about this framework, please open an issue on GitHub.

---

## 📝 Changelog

### Version 1.0 (December 2024)

- Initial release with complete analysis pipeline
- Technology-split models with Tweedie objective
- Isotonic post-calibration for tail improvement
- Equal-budget policy comparison (fixed N)
- Replicate-weight CIs for Jaccard and Dice
- Delta metric CIs for H1 (more robust than fold-based t-test)
- Physics diagnostics with Target SD, CV, and weighted correlation
- Error equity analysis by climate, income, housing type, tenure
- Technical appendix with 195 Q&A
