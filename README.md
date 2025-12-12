# Heating Demand Modeling Framework

A **policy-centric and physics-consistent framework** to model annual residential space-heating demand using RECS 2020 microdata. This framework quantifies generalization performance without leakage (Nested CV), and measures how survey weighting changes retrofit targeting, with defensible uncertainty bounds.

## 🎯 Research Objectives

### Hypotheses

**H1 (Physics-aware accuracy + plausibility):**
Splitting by heating technology (combustion vs electric; electric split into heat pump vs resistance) and controlling COVID-era occupancy shifts reduces outer-fold generalization error and non-physical behavior vs a monolithic model.

**H2 (Policy sensitivity of weighting):**
Using survey weights changes who gets flagged for retrofit targeting, not only average metrics.

**H3 (Methodological robustness):**
Nested CV prevents optimistic bias from tuning; combining Nested CV with replicate-weight variance yields defensible uncertainty for evaluation and policy metrics.

### Key Contributions

1. **Technology-aware structure** to improve physical plausibility and generalization
2. **Nested CV** to eliminate tuning leakage
3. **Survey weighting impact analysis** on retrofit targeting, with uncertainty via replicate-weight jackknife

## 📁 Project Structure

```
heating-demand-modeling/
├── configs/
│   └── default_config.yaml      # Configuration file
├── data/
│   ├── raw/                     # Raw RECS 2020 data
│   └── processed/               # Processed datasets
├── src/
│   ├── data/
│   │   ├── loader.py            # RECS data loader
│   │   └── preprocessor.py      # Technology grouping & preprocessing
│   ├── features/
│   │   └── builder.py           # Feature engineering (leakage-safe)
│   ├── models/
│   │   ├── baselines.py         # Physics-based baselines
│   │   └── main_models.py       # LightGBM and EBM models
│   ├── evaluation/
│   │   ├── metrics.py           # Weighted metrics & physics diagnostics
│   │   └── nested_cv.py         # Nested cross-validation
│   ├── policy/
│   │   └── targeting.py         # Policy targeting analysis
│   ├── uncertainty/
│   │   └── jackknife.py         # Jackknife uncertainty & refit sensitivity
│   ├── visualization/
│   │   └── plots.py             # Publication-quality figures
│   └── utils/
│       └── helpers.py           # Utility functions
├── outputs/
│   ├── figures/                 # Generated figures
│   ├── tables/                  # Generated tables
│   └── models/                  # Saved models
├── tests/                       # Unit tests
├── run_analysis.py              # Main analysis script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
cd /workspace

# Install dependencies
pip install -r requirements.txt
```

### Running the Analysis

```bash
# Full analysis (includes nested CV)
python run_analysis.py

# Quick test (skip nested CV)
python run_analysis.py --skip-cv

# Custom configuration
python run_analysis.py --config configs/custom_config.yaml --output results/
```

## 📊 Data

### RECS 2020 Microdata

- **Source:** EIA RECS 2020 public-use microdata
- **Records:** ~18,500 households
- **Target:** `TOTALBTUSPH` (annual space heating energy in kBTU)

### Weights

| Weight | Description |
|--------|-------------|
| `NWEIGHT` | Final survey weight |
| `NWEIGHT1`-`NWEIGHT60` | Jackknife replicate weights |

## 🔬 Methodology

### Technology Grouping (Section 3)

Mutually exclusive groups based on primary equipment and fuel:

| Group | Equipment | Fuel |
|-------|-----------|------|
| **Combustion** | Furnace, boiler, room heater, wood stove | Natural gas, propane, fuel oil, wood |
| **Electric Heat Pump** | Heat pump | Electricity |
| **Electric Resistance** | Baseboard, portable heaters | Electricity |
| **Hybrid/Ambiguous** | Other/mixed | Conflicting indicators |

### Nested Cross-Validation (Section 10)

```
Outer Loop (5 folds) - Evaluate generalization
└── Inner Loop (3 folds) - Tune hyperparameters
    └── Random Search (20-30 iterations)
```

**Stratification:** Census Division × Technology Group × HDD bins

### Physics Baselines (Section 6.1)

| Technology | Model |
|------------|-------|
| Combustion | E = a + b·HDD + c·Area + d·HDD·Area |
| Heat Pump | Piecewise linear in HDD (COP variation) |
| Resistance | Linear in HDD (COP = 1) |

### Policy Scores (Section 8)

1. **High-use score:** Predicted Ê_heat
2. **High-intensity score:** Predicted Ê_heat / Area
3. **Excess-demand:** Ê_heat - Ê_baseline (inefficiency proxy)

### Uncertainty Quantification (Section 9)

**Jackknife variance:**
```
Var(θ) = (n-1)/n × Σᵢ(θᵢ - θ̄)²
```

**Refit-sensitivity check:** Refits model on 10/60 replicates to quantify underestimation in metric-only jackknife.

## 📈 Outputs

### Tables (Section 13)

| Table | Description |
|-------|-------------|
| Table 1 | Weighted vs unweighted descriptives by tech group |
| Table 2 | Outer-fold performance (weighted) + runtime |
| Table 3 | Prediction interval coverage + calibration |

### Figures (Section 13)

| Figure | Description |
|--------|-------------|
| Figure 1 | Workflow diagram |
| Figure 2 | Predicted vs observed (hexbin) |
| Figure 3 | Candidate list mismatch + composition shifts |
| Figure 4 | EBM shape functions by technology |
| Figure 5 | Residual bias vs HDD bins |

## 🧪 Sensitivity Analyses (Section 14)

- **Targets:** E vs E/Area vs excess-demand score
- **Climate restriction:** Exclude HDD < 1000
- **COVID controls:** Direct vs proxies vs none
- **Technology assignment:** Primary-only vs hybrid-handling
- **Monotonic constraints:** With vs without

## 📋 Reproducibility Checklist (Section 15)

- [x] Exact technology grouping rules
- [x] Feature list + leakage blacklist
- [x] CV stratification scheme
- [x] Policy score definitions
- [x] Jackknife variance formula
- [x] Random seeds and versioning

## 🔒 Leakage Prevention

### Blacklisted Features (Section 7)

Features that are direct functions of energy use:

```python
LEAKAGE_BLACKLIST = [
    'DOLLAREL', 'DOLLARNG', 'DOLLARLP', 'DOLLARFO',
    'TOTALDOL', 'TOTALDOLSPH',
    'BTUELSPH', 'BTUNGSPH', 'BTULPSPH', 'BTUFOSPH',
    'KWH', 'BTUNG', 'BTULP', 'BTUFO', 'TOTALBTU'
]
```

### CV-Internal Preprocessing

All transformations fitted inside inner loop only:
- Missingness imputation
- Feature encoding
- Transformations (log1p)
- Feature scaling

## 📚 API Reference

### Data Loading

```python
from src.data.loader import RECSDataLoader

loader = RECSDataLoader('data/raw/recs2020_public_v7.csv')
df = loader.load()
```

### Preprocessing

```python
from src.data.preprocessor import preprocess_recs_data

df_processed, preprocessor = preprocess_recs_data(
    df_raw,
    assignment_rule='primary_only',
    exclude_no_heating=True,
    exclude_hybrid=False
)
```

### Feature Engineering

```python
from src.features.builder import FeatureBuilder

builder = FeatureBuilder(
    covid_control_mode='direct',
    include_interactions=True
)
X_transformed = builder.fit_transform(X, y)
```

### Nested CV

```python
from src.evaluation.nested_cv import NestedCrossValidator

cv = NestedCrossValidator(outer_folds=5, inner_folds=3)
result = cv.run(X, y, weights, model_factory, param_grid)
```

### Policy Analysis

```python
from src.policy.targeting import PolicyTargeting

targeting = PolicyTargeting(target_percentile=90)
results = targeting.run_full_analysis(y_pred, y_baseline, y_true, weights, metadata)
```

### Uncertainty Quantification

```python
from src.uncertainty.jackknife import JackknifeUncertainty

jackknife = JackknifeUncertainty(n_replicates=60)
uncertainty_df = jackknife.compute_all_metrics_uncertainty(
    y_true, y_pred, main_weights, replicate_weights
)
```

## ⚠️ Scope Constraints

> **Important:** RECS 2020 reflects 2020 occupancy/behavior; transferring to "normal years" requires recalibration or additional years of data.

## 📖 Citation

If you use this framework, please cite:

```bibtex
@software{heating_demand_framework,
  title = {Heating Demand Modeling Framework for RECS 2020},
  year = {2024},
  note = {Policy-centric modeling with physics-consistent structure}
}
```

## 📄 License

MIT License

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📞 Contact

For questions about this framework, please open an issue on GitHub.
