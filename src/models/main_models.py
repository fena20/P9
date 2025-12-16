"""
Main Models for Heating Demand Prediction

Implements Section 6.2: Main models
- LightGBM (primary): supports sample weights, robust, fast
- EBM (interpretability): shape functions for physical sanity + policy narratives

Section 6.3: Output non-negativity
- Gamma/Tweedie objective for non-negative predictions
- Post-processing clamp at 0 with reporting

CALIBRATION METHODOLOGY (leakage-free):
----------------------------------------
The isotonic calibration is fitted INSIDE each outer fold of nested CV:
1. For each outer fold, split data into train/test
2. Fit LightGBM on training data
3. Fit isotonic calibrator on TRAINING predictions vs training targets
4. Apply calibrator when predicting on TEST data
This ensures calibration does not leak test information.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.base import BaseEstimator, RegressorMixin, clone
import warnings

from src.utils.helpers import logger, Timer, clip_predictions


class LightGBMHeatingModel(BaseEstimator, RegressorMixin):
    """
    LightGBM model for heating demand prediction.
    
    Features:
    - Gamma objective for non-negative predictions
    - Sample weight support
    - Optional monotonic constraints
    - Hyperparameter configuration
    """
    
    # Default hyperparameters
    # Using Tweedie with power=1.5 (between Poisson=1 and Gamma=2)
    # Better for right-skewed data with many moderate values
    DEFAULT_PARAMS = {
        'objective': 'tweedie',  # Tweedie: handles zero-inflated + heavy tails
        'tweedie_variance_power': 1.5,  # 1=Poisson, 2=Gamma, 1.5=compound
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'n_estimators': 300,
        'max_depth': 6,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Monotonic constraints for physical plausibility
    # 1 = increasing, -1 = decreasing, 0 = no constraint
    MONOTONIC_CONSTRAINTS_MAP = {
        'HDD65': 1,           # More HDD -> More energy
        'HDD30YR_PUB': 1,     # More HDD -> More energy
        'TOTSQFT_EN': 1,      # Larger area -> More energy
        'TOTHSQFT': 1,        # Larger heated area -> More energy
        # Note: ADQINSUL requires knowing its encoding direction
    }
    
    def __init__(self, 
                 params: Optional[Dict] = None,
                 use_monotonic_constraints: bool = False,
                 feature_names: Optional[List[str]] = None,
                 apply_bias_correction: bool = True):
        """
        Initialize LightGBM heating model.
        
        Parameters
        ----------
        params : dict, optional
            LightGBM parameters (merged with defaults)
        use_monotonic_constraints : bool
            Whether to apply monotonic constraints
        feature_names : list, optional
            Feature names for constraint mapping
        apply_bias_correction : bool
            Whether to apply post-hoc bias correction (reduces systematic under/over-prediction)
        """
        self.params = params or {}
        self.use_monotonic_constraints = use_monotonic_constraints
        self.feature_names = feature_names
        self.apply_bias_correction = apply_bias_correction
        
        self.model_ = None
        self.feature_importances_ = None
        self.n_clipped_ = 0
        self.clip_rate_ = 0.0
        self.bias_correction_ = 0.0  # Additive correction
        self.scale_correction_ = 1.0  # Multiplicative correction
        self.isotonic_calibrator_ = None  # For tail calibration
        
    def _get_params(self) -> Dict:
        """Get merged parameters."""
        merged = self.DEFAULT_PARAMS.copy()
        merged.update(self.params)
        return merged
    
    def _build_monotonic_constraints(self, feature_names: List[str]) -> List[int]:
        """Build monotonic constraint list for features."""
        constraints = []
        for feat in feature_names:
            if feat in self.MONOTONIC_CONSTRAINTS_MAP:
                constraints.append(self.MONOTONIC_CONSTRAINTS_MAP[feat])
            else:
                constraints.append(0)  # No constraint
        return constraints
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], 
            y: np.ndarray,
            sample_weight: Optional[np.ndarray] = None,
            eval_set: Optional[List[Tuple]] = None) -> 'LightGBMHeatingModel':
        """
        Fit the LightGBM model.
        
        Parameters
        ----------
        X : DataFrame or array
            Features
        y : array
            Target values
        sample_weight : array, optional
            Sample weights
        eval_set : list, optional
            Evaluation sets for early stopping
            
        Returns
        -------
        self
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm is required. Install with: pip install lightgbm")
        
        params = self._get_params()
        
        # Get feature names
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
            X_array = X.values
        else:
            feature_names = self.feature_names or [f'f{i}' for i in range(X.shape[1])]
            X_array = X
        
        # Apply monotonic constraints if requested
        if self.use_monotonic_constraints:
            constraints = self._build_monotonic_constraints(feature_names)
            params['monotone_constraints'] = constraints
            logger.info(f"Applied monotonic constraints to {sum(c != 0 for c in constraints)} features")
        
        # Handle gamma objective with zero values
        # Gamma requires y > 0, so we need to handle zeros
        y_adjusted = np.maximum(y, 1.0)  # Minimum of 1 BTU
        
        # Create model
        self.model_ = lgb.LGBMRegressor(**params)
        
        # Fit with or without sample weights
        fit_params = {}
        if sample_weight is not None:
            fit_params['sample_weight'] = sample_weight
        
        if eval_set is not None:
            fit_params['eval_set'] = eval_set
            fit_params['callbacks'] = [lgb.early_stopping(50, verbose=False)]
        
        self.model_.fit(X_array, y_adjusted, **fit_params)
        
        # Store feature importances
        self.feature_importances_ = dict(zip(
            feature_names, 
            self.model_.feature_importances_
        ))
        
        # Compute calibration on training data
        if self.apply_bias_correction:
            y_pred_train = self.model_.predict(X_array)
            residuals = y - y_pred_train
            
            if sample_weight is not None:
                # Weighted bias
                self.bias_correction_ = np.sum(sample_weight * residuals) / np.sum(sample_weight)
                
                # Also compute scale correction for calibration
                # Using weighted regression: y = scale * y_pred + bias
                y_pred_centered = y_pred_train - np.average(y_pred_train, weights=sample_weight)
                y_centered = y - np.average(y, weights=sample_weight)
                
                scale_num = np.sum(sample_weight * y_centered * y_pred_centered)
                scale_den = np.sum(sample_weight * y_pred_centered**2)
                
                if abs(scale_den) > 1e-10:
                    self.scale_correction_ = scale_num / scale_den
                    # Clamp scale correction to reasonable range
                    self.scale_correction_ = np.clip(self.scale_correction_, 0.8, 1.25)
                
                # Isotonic regression for non-linear tail calibration
                # Particularly helps with tail underprediction
                try:
                    from sklearn.isotonic import IsotonicRegression
                    
                    # Apply linear correction first
                    y_pred_linear = y_pred_train * self.scale_correction_ + self.bias_correction_
                    
                    # Fit isotonic regression: maps predictions to observed
                    self.isotonic_calibrator_ = IsotonicRegression(
                        y_min=0, y_max=None, out_of_bounds='clip'
                    )
                    # Use sample weights for fitting
                    self.isotonic_calibrator_.fit(y_pred_linear, y, sample_weight=sample_weight)
                    
                    logger.debug("Fitted isotonic calibrator for tail correction")
                except Exception as e:
                    logger.warning(f"Isotonic calibration failed: {e}")
                    self.isotonic_calibrator_ = None
            else:
                self.bias_correction_ = np.mean(residuals)
                self.scale_correction_ = 1.0
            
            logger.debug(f"Bias correction: {self.bias_correction_:.2f}, Scale: {self.scale_correction_:.3f}")
        
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray], 
                apply_correction: bool = True) -> np.ndarray:
        """
        Predict heating demand.
        
        Parameters
        ----------
        X : DataFrame or array
            Features
        apply_correction : bool
            Whether to apply bias/scale correction
            
        Returns
        -------
        array
            Predictions (non-negative)
        """
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        predictions = self.model_.predict(X)
        
        # Apply calibration
        if apply_correction and self.apply_bias_correction:
            # Apply linear correction first
            predictions = predictions * self.scale_correction_ + self.bias_correction_
            
            # Apply isotonic calibration for tail improvement
            if self.isotonic_calibrator_ is not None:
                predictions = self.isotonic_calibrator_.predict(predictions)
        
        # Clip negative predictions (shouldn't happen with gamma but safety check)
        predictions, self.clip_rate_ = clip_predictions(predictions, min_val=0)
        
        if self.clip_rate_ > 0:
            logger.warning(f"Clipped {self.clip_rate_*100:.2f}% of predictions to non-negative")
        
        return predictions
    
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        Get feature importance.
        
        Parameters
        ----------
        importance_type : str
            Type of importance ('gain', 'split')
            
        Returns
        -------
        DataFrame
            Feature importance table
        """
        if self.model_ is None:
            raise ValueError("Model not fitted.")
        
        if importance_type == 'gain':
            importance = self.model_.booster_.feature_importance(importance_type='gain')
        else:
            importance = self.model_.feature_importances_
        
        feature_names = (list(self.feature_importances_.keys()) 
                        if self.feature_importances_ else 
                        [f'f{i}' for i in range(len(importance))])
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        df['importance_pct'] = df['importance'] / df['importance'].sum() * 100
        
        return df


class EBMHeatingModel(BaseEstimator, RegressorMixin):
    """
    Explainable Boosting Machine for heating demand prediction.
    
    Features:
    - Interpretable shape functions
    - Physical plausibility verification via shapes
    - Sample weight support
    """
    
    DEFAULT_PARAMS = {
        'max_bins': 256,
        'interactions': 10,
        'outer_bags': 8,
        'inner_bags': 0,
        'learning_rate': 0.01,
        'validation_size': 0.15,
        'early_stopping_rounds': 50,
        'max_rounds': 10000,
        'random_state': 42
    }
    
    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize EBM heating model.
        
        Parameters
        ----------
        params : dict, optional
            EBM parameters (merged with defaults)
        """
        self.params = params or {}
        self.model_ = None
        self.feature_names_ = None
        
    def _get_params(self) -> Dict:
        """Get merged parameters."""
        merged = self.DEFAULT_PARAMS.copy()
        merged.update(self.params)
        return merged
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray],
            y: np.ndarray,
            sample_weight: Optional[np.ndarray] = None) -> 'EBMHeatingModel':
        """
        Fit the EBM model.
        
        Parameters
        ----------
        X : DataFrame or array
            Features
        y : array
            Target values
        sample_weight : array, optional
            Sample weights
            
        Returns
        -------
        self
        """
        try:
            from interpret.glassbox import ExplainableBoostingRegressor
        except ImportError:
            raise ImportError("interpret is required. Install with: pip install interpret")
        
        params = self._get_params()
        
        # Get feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_array = X.values
        else:
            self.feature_names_ = [f'f{i}' for i in range(X.shape[1])]
            X_array = X
        
        # Create and fit model
        self.model_ = ExplainableBoostingRegressor(
            feature_names=self.feature_names_,
            **params
        )
        
        # Handle sample weights
        if sample_weight is not None:
            self.model_.fit(X_array, y, sample_weight=sample_weight)
        else:
            self.model_.fit(X_array, y)
        
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict heating demand.
        
        Parameters
        ----------
        X : DataFrame or array
            Features
            
        Returns
        -------
        array
            Predictions
        """
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        predictions = self.model_.predict(X)
        
        # Clip negative predictions
        predictions, clip_rate = clip_predictions(predictions, min_val=0)
        
        if clip_rate > 0:
            logger.warning(f"Clipped {clip_rate*100:.2f}% of EBM predictions to non-negative")
        
        return predictions
    
    def get_shape_functions(self) -> Dict[str, Any]:
        """
        Get shape functions for interpretation.
        
        Returns
        -------
        dict
            Dictionary with feature shapes
        """
        if self.model_ is None:
            raise ValueError("Model not fitted.")
        
        shapes = {}
        
        # Get global explanations
        global_exp = self.model_.explain_global()
        
        for i, name in enumerate(global_exp.data()['names']):
            if i < len(self.model_.term_features_):
                term_idx = i
                
                shapes[name] = {
                    'bins': self.model_.bins_[term_idx] if term_idx < len(self.model_.bins_) else None,
                    'scores': self.model_.term_scores_[term_idx] if term_idx < len(self.model_.term_scores_) else None,
                }
        
        return shapes
    
    def verify_hdd_direction(self) -> Dict[str, Any]:
        """
        Verify that HDD shape function has correct direction.
        
        For heating, HDD should have positive effect on energy.
        
        Returns
        -------
        dict
            Verification results
        """
        if self.model_ is None:
            raise ValueError("Model not fitted.")
        
        result = {
            'hdd_found': False,
            'correct_direction': None,
            'correlation': None
        }
        
        # Find HDD feature
        for i, name in enumerate(self.feature_names_):
            if 'HDD' in name.upper():
                result['hdd_found'] = True
                
                if i < len(self.model_.term_scores_):
                    scores = self.model_.term_scores_[i]
                    
                    # Check if scores generally increase
                    if len(scores) > 1:
                        # Compute correlation with index (should be positive)
                        indices = np.arange(len(scores))
                        valid = ~np.isnan(scores)
                        if valid.sum() > 1:
                            corr = np.corrcoef(indices[valid], scores[valid])[0, 1]
                            result['correlation'] = corr
                            result['correct_direction'] = corr > 0
                
                break
        
        return result


class HeatingDemandModels:
    """
    Container for all heating demand models.
    
    Manages:
    - Technology-specific models
    - Monolithic comparison model
    - Model selection and evaluation
    """
    
    def __init__(self, 
                 model_type: str = 'lightgbm',
                 use_monotonic_constraints: bool = False,
                 params: Optional[Dict] = None):
        """
        Initialize heating demand models.
        
        Parameters
        ----------
        model_type : str
            Model type: 'lightgbm' or 'ebm'
        use_monotonic_constraints : bool
            Whether to use monotonic constraints (LightGBM only)
        params : dict, optional
            Model parameters
        """
        self.model_type = model_type
        self.use_monotonic_constraints = use_monotonic_constraints
        self.params = params or {}
        
        self.tech_models_ = {}
        self.monolithic_model_ = None
        self.is_fitted_ = False
        
    def _create_model(self) -> Union[LightGBMHeatingModel, EBMHeatingModel]:
        """Create a model instance."""
        if self.model_type == 'lightgbm':
            return LightGBMHeatingModel(
                params=self.params,
                use_monotonic_constraints=self.use_monotonic_constraints
            )
        elif self.model_type == 'ebm':
            return EBMHeatingModel(params=self.params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit_split(self, X: pd.DataFrame, y: pd.Series,
                  tech_group: pd.Series,
                  sample_weight: Optional[pd.Series] = None) -> 'HeatingDemandModels':
        """
        Fit separate models for each technology group.
        
        Parameters
        ----------
        X : DataFrame
            Features
        y : Series
            Target values
        tech_group : Series
            Technology group labels
        sample_weight : Series, optional
            Sample weights
            
        Returns
        -------
        self
        """
        with Timer("Fitting split models"):
            for group_name in tech_group.unique():
                if group_name in ['no_heating', 'unknown']:
                    continue
                
                mask = tech_group == group_name
                n_samples = mask.sum()
                
                if n_samples < 50:
                    logger.warning(f"Skipping {group_name}: only {n_samples} samples")
                    continue
                
                X_group = X.loc[mask].copy()
                y_group = y.loc[mask].values
                weights_group = sample_weight.loc[mask].values if sample_weight is not None else None
                
                model = self._create_model()
                model.fit(X_group, y_group, sample_weight=weights_group)
                
                self.tech_models_[group_name] = model
                logger.info(f"Fitted {self.model_type} for {group_name} on {n_samples} samples")
        
        self.is_fitted_ = True
        return self
    
    def fit_monolithic(self, X: pd.DataFrame, y: pd.Series,
                       sample_weight: Optional[pd.Series] = None) -> 'HeatingDemandModels':
        """
        Fit a single monolithic model (for comparison).
        
        Parameters
        ----------
        X : DataFrame
            Features
        y : Series
            Target values
        sample_weight : Series, optional
            Sample weights
            
        Returns
        -------
        self
        """
        with Timer("Fitting monolithic model"):
            self.monolithic_model_ = self._create_model()
            
            weights = sample_weight.values if sample_weight is not None else None
            self.monolithic_model_.fit(X, y.values, sample_weight=weights)
            
            logger.info(f"Fitted monolithic {self.model_type} on {len(X)} samples")
        
        return self
    
    def predict_split(self, X: pd.DataFrame, 
                      tech_group: pd.Series,
                      apply_correction: bool = True) -> np.ndarray:
        """
        Predict using technology-specific models.
        
        Parameters
        ----------
        X : DataFrame
            Features
        tech_group : Series
            Technology group labels
        apply_correction : bool
            Whether to apply calibration correction
            
        Returns
        -------
        array
            Predictions
        """
        if not self.tech_models_:
            raise ValueError("Split models not fitted. Call fit_split() first.")
        
        predictions = np.zeros(len(X))
        
        for group_name, model in self.tech_models_.items():
            mask = tech_group == group_name
            if mask.sum() == 0:
                continue
            
            X_group = X.loc[mask].copy()
            # Pass apply_correction if model supports it
            try:
                predictions[mask.values] = model.predict(X_group, apply_correction=apply_correction)
            except TypeError:
                predictions[mask.values] = model.predict(X_group)
        
        # Handle groups without models
        unhandled_mask = ~np.isin(tech_group.values, list(self.tech_models_.keys()))
        if unhandled_mask.sum() > 0:
            # Use average of other predictions or a default
            if predictions[~unhandled_mask].sum() > 0:
                default_pred = np.mean(predictions[~unhandled_mask])
            else:
                default_pred = 0
            predictions[unhandled_mask] = default_pred
            logger.warning(f"Used default prediction for {unhandled_mask.sum()} samples without tech group model")
        
        return predictions
    
    def predict_monolithic(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using monolithic model.
        
        Parameters
        ----------
        X : DataFrame
            Features
            
        Returns
        -------
        array
            Predictions
        """
        if self.monolithic_model_ is None:
            raise ValueError("Monolithic model not fitted. Call fit_monolithic() first.")
        
        return self.monolithic_model_.predict(X)
    
    def get_feature_importance(self, model_name: str = 'monolithic') -> pd.DataFrame:
        """
        Get feature importance from specified model.
        
        Parameters
        ----------
        model_name : str
            'monolithic' or a tech group name
            
        Returns
        -------
        DataFrame
            Feature importance table
        """
        if model_name == 'monolithic':
            if self.monolithic_model_ is None:
                raise ValueError("Monolithic model not fitted.")
            return self.monolithic_model_.get_feature_importance()
        else:
            if model_name not in self.tech_models_:
                raise ValueError(f"No model for tech group: {model_name}")
            return self.tech_models_[model_name].get_feature_importance()
