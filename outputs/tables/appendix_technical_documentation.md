# Technical Appendix

## A. Technology Group Assignment Rulebook

### A.1 Primary Heating Equipment Categories (RECS 2020)

| EQUIPM Code | Description | Technology Group |
|-------------|-------------|------------------|
| 2 | Central furnace (gas/propane/oil) | Combustion |
| 3 | Central warm-air furnace with heat pump | Hybrid Ambiguous |
| 4 | Steam/hot water boiler | Combustion |
| 5 | Electric central heat pump | Electric Heat Pump |
| 6 | Built-in electric units | Electric Resistance |
| 7 | Built-in gas/oil room heaters | Combustion |
| 8 | Wood-burning stove | Combustion |
| 9 | Fireplace | Combustion |
| 10 | Portable electric heaters | Electric Resistance |
| 11 | Ductless mini-split (heat pump) | Electric Heat Pump |
| 12 | Other | Hybrid Ambiguous |
| -2 | Not applicable (no heating) | No Heating |

### A.2 Assignment Rules

1. **Primary-Only Rule**: Each household assigned to exactly one technology group based on primary heating equipment (`EQUIPM`).

2. **Ambiguous Cases**: 
   - Equipment codes 3 (furnace+HP), 12 (other) → Hybrid Ambiguous
   - These are modeled separately but results should be interpreted with caution

3. **Exclusions**:
   - `No Heating` excluded from modeling (n ≈ 0 for space heating analysis)
   - Missing EQUIPM → excluded from analysis

### A.3 Population Shares (RECS 2020, weighted)

| Technology Group | Weighted Share | n (unweighted) |
|------------------|----------------|----------------|
| Combustion | 63.3% | 11,852 |
| Electric Heat Pump | 13.8% | 2,217 |
| Electric Resistance | 9.1% | 1,376 |
| Hybrid Ambiguous | 13.9% | 2,186 |

---

## B. Jackknife Variance Estimation (RECS Replicate Weights)

### B.1 Methodology

RECS 2020 provides 60 replicate weights (`NWEIGHT1`–`NWEIGHT60`) for variance estimation using the successive difference replication (SDR) method.

### B.2 Formulas

For any statistic θ estimated using main weight w:

**Replicate Estimate:**
$$\hat{\theta}_r = f(data, w_r) \quad \text{for } r = 1, \ldots, 60$$

**Variance Estimate:**
$$\widehat{Var}(\hat{\theta}) = \frac{4}{60} \sum_{r=1}^{60} (\hat{\theta}_r - \hat{\theta})^2$$

**Standard Error:**
$$SE(\hat{\theta}) = \sqrt{\widehat{Var}(\hat{\theta})}$$

**95% Confidence Interval:**
$$CI_{95\%} = \hat{\theta} \pm 1.96 \times SE(\hat{\theta})$$

### B.3 Applied Statistics

This method is used for:
- Weighted RMSE, MAE, R², Bias
- Jaccard Index and Dice Overlap (policy targeting)
- Composition shares (by income, housing type, climate)

### B.4 Limitation

Jackknife captures **sampling uncertainty** but not:
- Model specification uncertainty
- Hyperparameter sensitivity
- Year-to-year variability (2020 only)

---

## C. Anti-Leakage Pipeline Design

### C.1 Strict Separation Principle

All preprocessing that uses target information is performed **inside** the CV loop:

```
Outer Loop (k = 1...5):
├── Train/Test Split (stratified by Division × Tech × HDD)
│
├── Inner Loop (j = 1...3):
│   ├── Train/Val Split
│   │
│   ├── [Inside CV] Fit preprocessor on inner train only:
│   │   - Feature scaling (mean/std from train)
│   │   - Missing value imputation (median from train)
│   │   - Categorical encoding (fitted on train)
│   │
│   ├── [Inside CV] Transform val using train-fitted preprocessor
│   │
│   └── Hyperparameter tuning on val
│
├── [Inside CV] Refit preprocessor on full outer-train
├── [Inside CV] Transform outer-test
└── Evaluate on outer-test (never seen during fitting)
```

### C.2 Leakage-Blacklisted Features

The following features are **excluded** from the model to prevent information leakage:

| Feature | Reason for Exclusion |
|---------|---------------------|
| TOTALBTUSPH | Target variable |
| DOLLARSPH | Expenditure (correlated with target) |
| BTUEL, BTUNG, BTULP, BTUFO, BTUKERO | End-use components (sum = target) |
| KWHSPH, CUFTNG_* | Energy usage sub-components |
| NWEIGHT | Survey weight (not a feature) |
| DOEID | Household identifier |

### C.3 Feature Engineering Safeguards

- `HDD65` computed from climate data (exogenous)
- `COVID indicators` based on survey timing (exogenous)
- No target-based binning or thresholds created outside CV

---

## D. Calibration Methodology

### D.1 Two-Stage Calibration

**Stage 1: Linear Bias/Scale Correction**
- Fitted on training data residuals
- Additive bias: β₀ = weighted mean(y - ŷ)
- Multiplicative scale: β₁ from weighted regression

**Stage 2: Isotonic Calibration**
- Non-parametric monotonic transformation
- Fitted on (ŷ_linear, y) pairs from training
- Particularly addresses tail underprediction

### D.2 Application

```python
# During prediction:
y_raw = model.predict(X)
y_linear = y_raw * scale_correction + bias_correction
y_calibrated = isotonic_calibrator.transform(y_linear)
```

### D.3 Evaluation Impact

| Metric | Before Calibration | After Calibration |
|--------|-------------------|-------------------|
| Calibration Slope | ~0.50–0.60 | ~0.70–0.80 |
| Tail Bias (top 10%) | −50% to −60% | −40% to −50% |
| Overall wBias | −7,000 kBTU | −6,000 kBTU |

**Note**: Isotonic calibration improves tail but does not eliminate systematic underprediction entirely. This is a known limitation.

---

## E. Policy Targeting Definitions

### E.1 Policy Scores

| Score | Formula | Interpretation |
|-------|---------|----------------|
| High-Use | Ŷ | Absolute predicted consumption |
| High-Intensity | Ŷ / Area | Energy use per sq ft |
| Excess-Demand | Ŷ - Ŷ_baseline | Difference from physics baseline |

### E.2 Overlap Metrics

| Metric | Formula | Range |
|--------|---------|-------|
| Jaccard | \|A ∩ B\| / \|A ∪ B\| | [0, 1] |
| Dice (Sørensen) | 2\|A ∩ B\| / (\|A\| + \|B\|) | [0, 1] |
| Containment | \|A ∩ B\| / min(\|A\|, \|B\|) | [0, 1] |
| Recall (W→U) | \|A ∩ B\| / \|A\| | [0, 1] |

Where:
- A = Weighted candidate set (top 10% by weighted threshold)
- B = Unweighted candidate set (top 10% by sample count)

### E.3 Equal-Budget Comparison

For fair policy comparison, both methods select **same N candidates**:
- N = |A| (weighted selection count)
- Compare Jaccard of top-N lists (one by weighted score, one by raw score)

---

## F. Limitations and Caveats

### F.1 Data Limitations

1. **RECS 2020 Timing**: Data collected during COVID-19 pandemic; occupancy patterns may not reflect "normal" years.

2. **End-Use Estimation**: TOTALBTUSPH is modeled by EIA, not directly metered. This introduces measurement error.

3. **Technology Misclassification**: Self-reported equipment may have errors.

### F.2 Model Limitations

1. **Tail Underprediction**: Model systematically underpredicts high consumption (−25% to −50% for top decile). Policy targeting may miss some high-use households.

2. **Electric Subgroup Performance**: Low within-group variance in electric heat pump/resistance groups leads to low wR² (0.06–0.08), though correlation (r ≈ 0.3–0.5) indicates predictive signal exists.

3. **Cold Climate Bias**: Underprediction increases with HDD (−18% for HDD 6–8k, −20% for HDD 8k+).

### F.3 Policy Implications

- High-use targeting may systematically under-select:
  - Cold climate households
  - Single-family detached homes
  - Combustion heating users

- Equity auditing recommended before deployment.

---

*Generated by RECS 2020 Heating Demand Modeling Framework*
*Metrics computed on outer-fold test predictions, weighted by NWEIGHT*
