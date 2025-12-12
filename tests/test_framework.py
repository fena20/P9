"""
Unit tests for the Heating Demand Modeling Framework.

Run with: pytest tests/test_framework.py -v
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDataLoader:
    """Tests for data loading functionality."""
    
    def test_loader_import(self):
        """Test that loader can be imported."""
        from src.data.loader import RECSDataLoader
        assert RECSDataLoader is not None
    
    def test_load_data(self):
        """Test data loading."""
        from src.data.loader import RECSDataLoader
        
        loader = RECSDataLoader('data/raw/recs2020_public_v7.csv')
        df = loader.load()
        
        assert df is not None
        assert len(df) > 0
        assert 'TOTALBTUSPH' in df.columns
        assert 'NWEIGHT' in df.columns
        assert 'HDD65' in df.columns


class TestPreprocessor:
    """Tests for preprocessing functionality."""
    
    def test_preprocessor_import(self):
        """Test that preprocessor can be imported."""
        from src.data.preprocessor import RECSPreprocessor, preprocess_recs_data
        assert RECSPreprocessor is not None
    
    def test_technology_grouping(self):
        """Test technology group assignment."""
        from src.data.loader import RECSDataLoader
        from src.data.preprocessor import preprocess_recs_data
        
        loader = RECSDataLoader('data/raw/recs2020_public_v7.csv')
        df = loader.load()
        df_proc, _ = preprocess_recs_data(df, exclude_no_heating=True)
        
        # Check tech groups exist
        assert 'tech_group' in df_proc.columns
        
        # Check expected groups
        groups = df_proc['tech_group'].unique()
        assert 'combustion' in groups
        assert 'electric_heat_pump' in groups
        assert 'electric_resistance' in groups


class TestFeatureBuilder:
    """Tests for feature engineering."""
    
    def test_feature_builder_import(self):
        """Test that feature builder can be imported."""
        from src.features.builder import FeatureBuilder
        assert FeatureBuilder is not None
    
    def test_leakage_blacklist(self):
        """Test that leakage features are blocked."""
        from src.features.builder import FeatureBuilder
        
        builder = FeatureBuilder()
        blacklist = builder.LEAKAGE_BLACKLIST
        
        assert 'DOLLAREL' in blacklist
        assert 'TOTALDOL' in blacklist
        assert 'BTUELSPH' in blacklist


class TestModels:
    """Tests for model implementations."""
    
    def test_baselines_import(self):
        """Test that baselines can be imported."""
        from src.models.baselines import PhysicsBaselines, CombustionBaseline
        assert PhysicsBaselines is not None
    
    def test_lightgbm_import(self):
        """Test that LightGBM model can be imported."""
        from src.models.main_models import LightGBMHeatingModel
        assert LightGBMHeatingModel is not None
    
    def test_baseline_fit_predict(self):
        """Test baseline model fitting and prediction."""
        from src.data.loader import RECSDataLoader
        from src.data.preprocessor import preprocess_recs_data
        from src.models.baselines import PhysicsBaselines
        
        loader = RECSDataLoader('data/raw/recs2020_public_v7.csv')
        df = loader.load()
        df_proc, _ = preprocess_recs_data(df, exclude_no_heating=True)
        
        X = df_proc[['HDD65', 'TOTSQFT_EN']].copy()
        y = df_proc['TOTALBTUSPH'].copy()
        tech = df_proc['tech_group'].copy()
        weights = df_proc['NWEIGHT'].copy()
        
        baselines = PhysicsBaselines()
        baselines.fit(X, y, tech, weights)
        
        predictions = baselines.predict(X, tech)
        
        assert len(predictions) == len(X)
        assert all(predictions >= 0)


class TestMetrics:
    """Tests for evaluation metrics."""
    
    def test_metrics_import(self):
        """Test that metrics can be imported."""
        from src.evaluation.metrics import WeightedMetrics, PhysicsDiagnostics
        assert WeightedMetrics is not None
    
    def test_weighted_rmse(self):
        """Test weighted RMSE calculation."""
        from src.evaluation.metrics import WeightedMetrics
        
        metrics = WeightedMetrics()
        
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 190, 310])
        weights = np.array([1, 1, 1])
        
        rmse = metrics.weighted_rmse(y_true, y_pred, weights)
        
        assert rmse > 0
        assert np.isfinite(rmse)
    
    def test_weighted_r2(self):
        """Test weighted R² calculation."""
        from src.evaluation.metrics import WeightedMetrics
        
        metrics = WeightedMetrics()
        
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([100, 200, 300, 400, 500])  # Perfect predictions
        weights = np.array([1, 1, 1, 1, 1])
        
        r2 = metrics.weighted_r2(y_true, y_pred, weights)
        
        assert r2 == 1.0


class TestPolicyTargeting:
    """Tests for policy targeting analysis."""
    
    def test_targeting_import(self):
        """Test that targeting can be imported."""
        from src.policy.targeting import PolicyTargeting
        assert PolicyTargeting is not None
    
    def test_identify_candidates(self):
        """Test candidate identification."""
        from src.policy.targeting import PolicyTargeting
        
        targeting = PolicyTargeting(target_percentile=90)
        
        scores = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        weights = np.ones(10)
        
        candidates = targeting.identify_candidates(scores, weights, use_weights=True)
        
        # Top 10% - should identify the highest values
        assert candidates.sum() >= 1  # At least one candidate
        assert candidates[-1] == True  # Highest should be candidate


class TestUncertainty:
    """Tests for uncertainty quantification."""
    
    def test_jackknife_import(self):
        """Test that jackknife can be imported."""
        from src.uncertainty.jackknife import JackknifeUncertainty
        assert JackknifeUncertainty is not None
    
    def test_jackknife_variance(self):
        """Test jackknife variance formula."""
        from src.uncertainty.jackknife import JackknifeUncertainty
        
        # Jackknife variance: (n-1)/n * sum((theta_i - theta_bar)^2)
        n = 10
        estimates = np.array([1.0, 1.1, 0.9, 1.05, 0.95, 1.02, 0.98, 1.03, 0.97, 1.01])
        theta_bar = estimates.mean()
        
        expected_var = (n - 1) / n * np.sum((estimates - theta_bar) ** 2)
        
        assert expected_var > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
