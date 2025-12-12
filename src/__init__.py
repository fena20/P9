"""
Heating Demand Modeling Framework for RECS 2020

A policy-centric and physics-consistent framework to model annual residential 
space-heating demand using RECS 2020 microdata.

Main components:
- data: Data loading and preprocessing
- features: Feature engineering
- models: Baseline and main models (LightGBM, EBM)
- evaluation: Metrics and nested CV
- policy: Targeting analysis
- uncertainty: Jackknife variance estimation
- visualization: Plotting utilities
"""

__version__ = "0.1.0"
__author__ = "Heating Demand Research Team"
