"""
Nested Cross-Validation Infrastructure

Implements Section 10: Nested CV design (5×3; technology-aware)
- Outer loop: 5 folds, stratified by (Division × Tech group × HDD bins)
- Inner loop: 3 folds, randomized search optimizing Weighted RMSE
- Model selection: choose hyperparameters per outer fold using inner CV only
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from sklearn.model_selection import StratifiedKFold, ParameterSampler
from sklearn.base import clone, BaseEstimator
import warnings
from dataclasses import dataclass
from copy import deepcopy

from src.utils.helpers import logger, Timer, create_stratification_key
from src.evaluation.metrics import WeightedMetrics, PhysicsDiagnostics
from src.features.builder import FeatureBuilder


@dataclass
class CVFoldResult:
    """Results from a single CV fold."""
    fold_idx: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    predictions: np.ndarray
    y_true: np.ndarray
    weights: np.ndarray
    metrics: Dict[str, float]
    best_params: Optional[Dict] = None
    model: Optional[Any] = None
    train_time: float = 0.0


@dataclass
class NestedCVResult:
    """Complete results from nested CV."""
    fold_results: List[CVFoldResult]
    outer_metrics: Dict[str, float]
    outer_metrics_by_fold: pd.DataFrame
    physics_diagnostics: Dict[str, Any]
    runtime: float
    config: Dict[str, Any]


class NestedCrossValidator:
    """
    Nested cross-validation for heating demand modeling.
    
    Implements leakage-proof evaluation:
    - Outer loop evaluates generalization
    - Inner loop tunes hyperparameters
    - All preprocessing inside inner loop
    """
    
    def __init__(self,
                 outer_folds: int = 5,
                 inner_folds: int = 3,
                 n_search_iter: int = 30,
                 random_state: int = 42,
                 stratify_columns: Optional[List[str]] = None):
        """
        Initialize nested CV.
        
        Parameters
        ----------
        outer_folds : int
            Number of outer folds
        inner_folds : int
            Number of inner folds for hyperparameter tuning
        n_search_iter : int
            Number of random search iterations
        random_state : int
            Random seed
        stratify_columns : list, optional
            Columns to use for stratification
        """
        self.outer_folds = outer_folds
        self.inner_folds = inner_folds
        self.n_search_iter = n_search_iter
        self.random_state = random_state
        self.stratify_columns = stratify_columns or ['DIVISION', 'tech_group', 'HDD_bin']
        
        self.fold_results_ = []
        self.outer_predictions_ = None
        self.is_run_ = False
        
    def _create_stratification_key(self, df: pd.DataFrame) -> pd.Series:
        """Create stratification key from multiple columns."""
        available_cols = [c for c in self.stratify_columns if c in df.columns]
        
        if not available_cols:
            logger.warning("No stratification columns available, using random splits")
            return pd.Series(np.zeros(len(df)), index=df.index)
        
        # Combine columns into single key
        key = df[available_cols[0]].astype(str)
        for col in available_cols[1:]:
            key = key + '_' + df[col].astype(str)
        
        return key
    
    def _reduce_stratification_groups(self, strat_key: pd.Series, 
                                       min_samples: int = 5) -> pd.Series:
        """
        Reduce stratification groups that have too few samples.
        
        Groups smaller than min_samples are merged into an 'other' category.
        """
        counts = strat_key.value_counts()
        small_groups = counts[counts < min_samples].index
        
        if len(small_groups) > 0:
            logger.info(f"Merging {len(small_groups)} small stratification groups")
            strat_key = strat_key.replace(small_groups, 'other')
        
        return strat_key
    
    def run(self,
            X: pd.DataFrame,
            y: pd.Series,
            weights: pd.Series,
            model_factory: Callable[..., BaseEstimator],
            param_grid: Dict[str, List],
            feature_builder: Optional[FeatureBuilder] = None,
            tech_group: Optional[pd.Series] = None,
            additional_metadata: Optional[pd.DataFrame] = None) -> NestedCVResult:
        """
        Run nested cross-validation.
        
        Parameters
        ----------
        X : DataFrame
            Raw features (preprocessing done inside CV)
        y : Series
            Target values
        weights : Series
            Sample weights
        model_factory : callable
            Function that returns a new model instance
        param_grid : dict
            Hyperparameter search space
        feature_builder : FeatureBuilder, optional
            Feature builder for preprocessing
        tech_group : Series, optional
            Technology group labels
        additional_metadata : DataFrame, optional
            Additional columns for diagnostics (HDD65, DIVISION, etc.)
            
        Returns
        -------
        NestedCVResult
            Complete results
        """
        import time
        start_time = time.time()
        
        with Timer(f"Running {self.outer_folds}×{self.inner_folds} nested CV"):
            
            # Create stratification key
            if additional_metadata is not None:
                strat_df = additional_metadata.copy()
                if tech_group is not None:
                    strat_df['tech_group'] = tech_group
            else:
                strat_df = pd.DataFrame({'tech_group': tech_group}) if tech_group is not None else X.copy()
            
            strat_key = self._create_stratification_key(strat_df)
            strat_key = self._reduce_stratification_groups(strat_key, min_samples=self.outer_folds)
            
            # Encode stratification key for sklearn
            strat_encoded = pd.factorize(strat_key)[0]
            
            # Outer CV split
            outer_cv = StratifiedKFold(
                n_splits=self.outer_folds,
                shuffle=True,
                random_state=self.random_state
            )
            
            # Store results
            self.fold_results_ = []
            all_predictions = np.zeros(len(X))
            metrics_calc = WeightedMetrics()
            
            for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, strat_encoded)):
                logger.info(f"\n{'='*60}")
                logger.info(f"Outer Fold {fold_idx + 1}/{self.outer_folds}")
                logger.info(f"{'='*60}")
                logger.info(f"Train: {len(train_idx)}, Test: {len(test_idx)}")
                
                fold_start = time.time()
                
                # Split data
                X_train = X.iloc[train_idx].copy()
                X_test = X.iloc[test_idx].copy()
                y_train = y.iloc[train_idx].values
                y_test = y.iloc[test_idx].values
                w_train = weights.iloc[train_idx].values
                w_test = weights.iloc[test_idx].values
                
                tech_train = tech_group.iloc[train_idx] if tech_group is not None else None
                tech_test = tech_group.iloc[test_idx] if tech_group is not None else None
                
                # Preprocess features INSIDE fold (no leakage)
                if feature_builder is not None:
                    fb = deepcopy(feature_builder)
                    X_train_proc = fb.fit_transform(X_train, y_train)
                    X_test_proc = fb.transform(X_test)
                else:
                    X_train_proc = X_train
                    X_test_proc = X_test
                
                # Inner CV for hyperparameter tuning
                best_params = self._inner_cv(
                    X_train_proc, y_train, w_train,
                    model_factory, param_grid,
                    tech_group=tech_train
                )
                
                logger.info(f"Best params: {best_params}")
                
                # Fit final model on full training set with best params
                final_model = model_factory(**best_params)
                
                if hasattr(final_model, 'fit_split') and tech_train is not None:
                    final_model.fit_split(X_train_proc, pd.Series(y_train), 
                                          tech_train, pd.Series(w_train))
                    predictions = final_model.predict_split(X_test_proc, tech_test)
                else:
                    if hasattr(final_model, 'fit'):
                        final_model.fit(X_train_proc, y_train, sample_weight=w_train)
                    predictions = final_model.predict(X_test_proc)
                
                # Store predictions
                all_predictions[test_idx] = predictions
                
                # Compute fold metrics
                fold_metrics = metrics_calc.compute_all_metrics(y_test, predictions, w_test)
                
                fold_time = time.time() - fold_start
                
                fold_result = CVFoldResult(
                    fold_idx=fold_idx,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    predictions=predictions,
                    y_true=y_test,
                    weights=w_test,
                    metrics=fold_metrics,
                    best_params=best_params,
                    model=final_model,
                    train_time=fold_time
                )
                
                self.fold_results_.append(fold_result)
                
                logger.info(f"Fold {fold_idx + 1} metrics: RMSE={fold_metrics['weighted_rmse']:.2f}, "
                           f"MAE={fold_metrics['weighted_mae']:.2f}, R²={fold_metrics['weighted_r2']:.4f}")
            
            # Compute overall metrics
            self.outer_predictions_ = all_predictions
            overall_metrics = metrics_calc.compute_all_metrics(
                y.values, all_predictions, weights.values
            )
            
            # Compile fold metrics
            fold_metrics_df = pd.DataFrame([fr.metrics for fr in self.fold_results_])
            fold_metrics_df['fold'] = range(len(self.fold_results_))
            
            # Physics diagnostics
            physics = PhysicsDiagnostics()
            hdd = additional_metadata['HDD65'].values if additional_metadata is not None and 'HDD65' in additional_metadata else None
            division = additional_metadata['DIVISION'].values if additional_metadata is not None and 'DIVISION' in additional_metadata else None
            
            physics_results = {}
            if hdd is not None:
                physics_results = physics.run_all_diagnostics(
                    y.values, all_predictions, hdd, weights.values,
                    division=division,
                    tech_group=tech_group.values if tech_group is not None else None
                )
            
            total_time = time.time() - start_time
            
            self.is_run_ = True
            
            return NestedCVResult(
                fold_results=self.fold_results_,
                outer_metrics=overall_metrics,
                outer_metrics_by_fold=fold_metrics_df,
                physics_diagnostics=physics_results,
                runtime=total_time,
                config={
                    'outer_folds': self.outer_folds,
                    'inner_folds': self.inner_folds,
                    'n_search_iter': self.n_search_iter,
                    'random_state': self.random_state
                }
            )
    
    def _inner_cv(self,
                  X_train: pd.DataFrame,
                  y_train: np.ndarray,
                  w_train: np.ndarray,
                  model_factory: Callable,
                  param_grid: Dict[str, List],
                  tech_group: Optional[pd.Series] = None) -> Dict:
        """
        Inner CV for hyperparameter tuning.
        
        Parameters
        ----------
        X_train : DataFrame
            Training features
        y_train : array
            Training targets
        w_train : array
            Training weights
        model_factory : callable
            Model factory function
        param_grid : dict
            Hyperparameter grid
        tech_group : Series, optional
            Technology group labels
            
        Returns
        -------
        dict
            Best hyperparameters
        """
        # Random search
        param_list = list(ParameterSampler(
            param_grid, 
            n_iter=self.n_search_iter,
            random_state=self.random_state
        ))
        
        if len(param_list) == 0:
            return {}
        
        # Inner CV
        inner_cv = StratifiedKFold(
            n_splits=self.inner_folds,
            shuffle=True,
            random_state=self.random_state + 1
        )
        
        # Create simple stratification for inner loop
        if tech_group is not None:
            inner_strat = pd.factorize(tech_group.astype(str))[0]
        else:
            inner_strat = np.zeros(len(y_train), dtype=int)
        
        best_score = float('inf')
        best_params = param_list[0]
        
        metrics_calc = WeightedMetrics()
        
        for params in param_list:
            fold_scores = []
            
            try:
                for inner_train_idx, inner_val_idx in inner_cv.split(X_train, inner_strat):
                    # Split
                    X_inner_train = X_train.iloc[inner_train_idx]
                    X_inner_val = X_train.iloc[inner_val_idx]
                    y_inner_train = y_train[inner_train_idx]
                    y_inner_val = y_train[inner_val_idx]
                    w_inner_train = w_train[inner_train_idx]
                    w_inner_val = w_train[inner_val_idx]
                    
                    # Fit model
                    model = model_factory(**params)
                    
                    if hasattr(model, 'fit_split') and tech_group is not None:
                        tech_inner_train = tech_group.iloc[inner_train_idx]
                        tech_inner_val = tech_group.iloc[inner_val_idx]
                        model.fit_split(X_inner_train, pd.Series(y_inner_train),
                                       tech_inner_train, pd.Series(w_inner_train))
                        preds = model.predict_split(X_inner_val, tech_inner_val)
                    else:
                        model.fit(X_inner_train, y_inner_train, sample_weight=w_inner_train)
                        preds = model.predict(X_inner_val)
                    
                    # Compute weighted RMSE
                    rmse = metrics_calc.weighted_rmse(y_inner_val, preds, w_inner_val)
                    fold_scores.append(rmse)
                
                mean_score = np.mean(fold_scores)
                
                if mean_score < best_score:
                    best_score = mean_score
                    best_params = params
                    
            except Exception as e:
                logger.warning(f"Error with params {params}: {e}")
                continue
        
        logger.info(f"Inner CV best RMSE: {best_score:.2f}")
        return best_params
    
    def get_fold_predictions(self) -> pd.DataFrame:
        """
        Get predictions from all folds.
        
        Returns
        -------
        DataFrame
            Predictions with fold information
        """
        if not self.is_run_:
            raise ValueError("CV not run yet. Call run() first.")
        
        results = []
        for fr in self.fold_results_:
            for i, idx in enumerate(fr.test_idx):
                results.append({
                    'idx': idx,
                    'fold': fr.fold_idx,
                    'y_true': fr.y_true[i],
                    'y_pred': fr.predictions[i],
                    'weight': fr.weights[i]
                })
        
        return pd.DataFrame(results).sort_values('idx')
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Get summary statistics across folds.
        
        Returns
        -------
        DataFrame
            Summary statistics
        """
        if not self.is_run_:
            raise ValueError("CV not run yet. Call run() first.")
        
        metrics_df = pd.DataFrame([fr.metrics for fr in self.fold_results_])
        
        summary = {
            'metric': [],
            'mean': [],
            'std': [],
            'min': [],
            'max': []
        }
        
        for col in metrics_df.columns:
            if col == 'n_samples':
                continue
            summary['metric'].append(col)
            summary['mean'].append(metrics_df[col].mean())
            summary['std'].append(metrics_df[col].std())
            summary['min'].append(metrics_df[col].min())
            summary['max'].append(metrics_df[col].max())
        
        return pd.DataFrame(summary)


def compare_split_vs_monolithic(X: pd.DataFrame,
                                 y: pd.Series,
                                 weights: pd.Series,
                                 tech_group: pd.Series,
                                 split_model_factory: Callable,
                                 mono_model_factory: Callable,
                                 param_grid: Dict,
                                 feature_builder: Optional[FeatureBuilder] = None,
                                 metadata: Optional[pd.DataFrame] = None,
                                 n_outer_folds: int = 5,
                                 random_state: int = 42) -> Dict[str, Any]:
    """
    Compare split models vs monolithic model (H1 test).
    
    Parameters
    ----------
    X : DataFrame
        Features
    y : Series
        Target
    weights : Series
        Weights
    tech_group : Series
        Technology groups
    split_model_factory : callable
        Factory for split models
    mono_model_factory : callable
        Factory for monolithic model
    param_grid : dict
        Hyperparameter grid
    feature_builder : FeatureBuilder, optional
        Feature builder
    metadata : DataFrame, optional
        Metadata for diagnostics
    n_outer_folds : int
        Number of outer folds
    random_state : int
        Random seed
        
    Returns
    -------
    dict
        Comparison results
    """
    # Run split model CV
    split_cv = NestedCrossValidator(
        outer_folds=n_outer_folds,
        random_state=random_state
    )
    
    split_result = split_cv.run(
        X, y, weights,
        split_model_factory,
        param_grid,
        feature_builder=feature_builder,
        tech_group=tech_group,
        additional_metadata=metadata
    )
    
    # Run monolithic model CV
    mono_cv = NestedCrossValidator(
        outer_folds=n_outer_folds,
        random_state=random_state
    )
    
    mono_result = mono_cv.run(
        X, y, weights,
        mono_model_factory,
        param_grid,
        feature_builder=feature_builder,
        additional_metadata=metadata
    )
    
    # Compare results
    comparison = {
        'split_metrics': split_result.outer_metrics,
        'mono_metrics': mono_result.outer_metrics,
        'split_by_fold': split_result.outer_metrics_by_fold,
        'mono_by_fold': mono_result.outer_metrics_by_fold,
        'split_physics': split_result.physics_diagnostics,
        'mono_physics': mono_result.physics_diagnostics,
    }
    
    # Paired comparison
    split_rmse = split_result.outer_metrics_by_fold['weighted_rmse'].values
    mono_rmse = mono_result.outer_metrics_by_fold['weighted_rmse'].values
    
    comparison['rmse_difference'] = mono_rmse.mean() - split_rmse.mean()
    comparison['rmse_improvement_pct'] = (mono_rmse.mean() - split_rmse.mean()) / mono_rmse.mean() * 100
    
    # Simple paired t-test
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(split_rmse, mono_rmse)
    comparison['paired_t_statistic'] = t_stat
    comparison['paired_p_value'] = p_value
    
    return comparison
