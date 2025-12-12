"""
Policy Targeting Analysis

Implements Section 8 Policy Metrics and H2 testing:
- Define candidates using weighted 90th percentile of policy score
- Compare weighted vs unweighted candidate lists
- Jaccard index, overlap rate, composition shifts
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

from src.utils.helpers import logger, compute_weighted_quantile, Timer


class PolicyTargeting:
    """
    Policy targeting analysis for retrofit prioritization.
    
    Implements H2 test: Using survey weights changes who gets flagged
    for retrofit targeting, not only average metrics.
    
    Policy scores:
    - High-use score: predicted E_heat
    - High-intensity score: predicted E_heat / Area
    - Excess-demand (inefficiency proxy): residual-like score
    """
    
    def __init__(self, target_percentile: float = 90):
        """
        Initialize policy targeting analysis.
        
        Parameters
        ----------
        target_percentile : float
            Percentile threshold for targeting (default 90 = top 10%)
        """
        self.target_percentile = target_percentile
        self.quantile = target_percentile / 100
        
    def compute_policy_scores(self,
                              y_pred: np.ndarray,
                              y_baseline: np.ndarray,
                              area: np.ndarray) -> pd.DataFrame:
        """
        Compute policy scores for targeting.
        
        Parameters
        ----------
        y_pred : array
            Predicted energy values
        y_baseline : array
            Baseline energy predictions (for excess demand)
        area : array
            Floor area (square feet)
            
        Returns
        -------
        DataFrame
            Policy scores
        """
        scores = pd.DataFrame({
            'high_use': y_pred,
            'high_intensity': y_pred / np.maximum(area, 1),
            'excess_demand': y_pred - y_baseline
        })
        
        return scores
    
    def identify_candidates(self,
                            scores: np.ndarray,
                            weights: np.ndarray,
                            use_weights: bool = True) -> np.ndarray:
        """
        Identify candidates above threshold.
        
        Parameters
        ----------
        scores : array
            Policy scores
        weights : array
            Sample weights
        use_weights : bool
            Whether to use weights for threshold calculation
            
        Returns
        -------
        array
            Boolean mask of candidates
        """
        if use_weights:
            threshold = compute_weighted_quantile(scores, weights, self.quantile)
        else:
            threshold = np.quantile(scores[~np.isnan(scores)], self.quantile)
        
        return scores >= threshold
    
    def compare_weighted_vs_unweighted(self,
                                        scores: pd.DataFrame,
                                        weights: np.ndarray,
                                        metadata: pd.DataFrame) -> Dict[str, Any]:
        """
        Compare weighted vs unweighted targeting.
        
        Parameters
        ----------
        scores : DataFrame
            Policy scores (high_use, high_intensity, excess_demand)
        weights : array
            Sample weights
        metadata : DataFrame
            Metadata for composition analysis (income, housing type, etc.)
            
        Returns
        -------
        dict
            Comparison results
        """
        results = {}
        
        for score_name in scores.columns:
            score_values = scores[score_name].values
            
            # Identify candidates with and without weights
            candidates_weighted = self.identify_candidates(score_values, weights, use_weights=True)
            candidates_unweighted = self.identify_candidates(score_values, weights, use_weights=False)
            
            # Compute overlap metrics
            overlap_metrics = self._compute_overlap_metrics(
                candidates_weighted, candidates_unweighted
            )
            
            # Composition analysis
            composition = self._analyze_composition_shift(
                candidates_weighted, candidates_unweighted,
                weights, metadata
            )
            
            results[score_name] = {
                'overlap': overlap_metrics,
                'composition': composition,
                'n_weighted_candidates': candidates_weighted.sum(),
                'n_unweighted_candidates': candidates_unweighted.sum(),
                'weighted_threshold': compute_weighted_quantile(score_values, weights, self.quantile),
                'unweighted_threshold': np.quantile(score_values[~np.isnan(score_values)], self.quantile)
            }
        
        return results
    
    def _compute_overlap_metrics(self,
                                  candidates_a: np.ndarray,
                                  candidates_b: np.ndarray) -> Dict[str, float]:
        """
        Compute overlap metrics between two candidate sets.
        
        Parameters
        ----------
        candidates_a : array
            First candidate mask (e.g., weighted)
        candidates_b : array
            Second candidate mask (e.g., unweighted)
            
        Returns
        -------
        dict
            Overlap metrics
        """
        # Jaccard index: intersection / union
        intersection = np.sum(candidates_a & candidates_b)
        union = np.sum(candidates_a | candidates_b)
        jaccard = intersection / union if union > 0 else 0
        
        # Overlap rate: intersection / min(|A|, |B|)
        min_size = min(candidates_a.sum(), candidates_b.sum())
        overlap_rate = intersection / min_size if min_size > 0 else 0
        
        # Proportion only in weighted
        only_weighted = np.sum(candidates_a & ~candidates_b)
        # Proportion only in unweighted
        only_unweighted = np.sum(~candidates_a & candidates_b)
        
        return {
            'jaccard_index': jaccard,
            'overlap_rate': overlap_rate,
            'intersection': intersection,
            'union': union,
            'only_weighted': only_weighted,
            'only_unweighted': only_unweighted,
            'pct_only_weighted': only_weighted / candidates_a.sum() * 100 if candidates_a.sum() > 0 else 0,
            'pct_only_unweighted': only_unweighted / candidates_b.sum() * 100 if candidates_b.sum() > 0 else 0
        }
    
    def _analyze_composition_shift(self,
                                    candidates_weighted: np.ndarray,
                                    candidates_unweighted: np.ndarray,
                                    weights: np.ndarray,
                                    metadata: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Analyze composition shifts by demographic/housing groups.
        
        Parameters
        ----------
        candidates_weighted : array
            Weighted candidate mask
        candidates_unweighted : array
            Unweighted candidate mask
        weights : array
            Sample weights
        metadata : DataFrame
            Metadata columns for grouping
            
        Returns
        -------
        dict
            Composition analysis by group type
        """
        results = {}
        
        # Analyze by available grouping variables
        grouping_vars = {
            'income': 'MONEYPY',
            'housing_type': 'TYPEHUQ',
            'tenure': 'KOWNRENT',
            'division': 'DIVISION',
            'climate': 'HDD_bin'
        }
        
        for group_name, col_name in grouping_vars.items():
            if col_name not in metadata.columns:
                continue
            
            group_col = metadata[col_name].values
            
            composition_df = self._compute_group_composition(
                candidates_weighted, candidates_unweighted,
                weights, group_col, group_name
            )
            
            results[group_name] = composition_df
        
        return results
    
    def _compute_group_composition(self,
                                    candidates_weighted: np.ndarray,
                                    candidates_unweighted: np.ndarray,
                                    weights: np.ndarray,
                                    groups: np.ndarray,
                                    group_name: str) -> pd.DataFrame:
        """
        Compute composition by group.
        """
        results = []
        
        for group_val in np.unique(groups):
            if pd.isna(group_val):
                continue
            
            group_mask = groups == group_val
            
            # Weighted share in weighted candidates
            w_in_weighted_cands = weights[candidates_weighted & group_mask].sum()
            total_w_weighted = weights[candidates_weighted].sum()
            share_weighted = w_in_weighted_cands / total_w_weighted * 100 if total_w_weighted > 0 else 0
            
            # Share in unweighted candidates
            n_in_unweighted = (candidates_unweighted & group_mask).sum()
            total_unweighted = candidates_unweighted.sum()
            share_unweighted = n_in_unweighted / total_unweighted * 100 if total_unweighted > 0 else 0
            
            # Population share (weighted)
            pop_share = weights[group_mask].sum() / weights.sum() * 100
            
            results.append({
                f'{group_name}': group_val,
                'share_weighted_candidates': share_weighted,
                'share_unweighted_candidates': share_unweighted,
                'share_difference': share_weighted - share_unweighted,
                'population_share': pop_share,
                'n_in_group': group_mask.sum()
            })
        
        return pd.DataFrame(results)
    
    def run_full_analysis(self,
                          y_pred: np.ndarray,
                          y_baseline: np.ndarray,
                          y_true: np.ndarray,
                          weights: np.ndarray,
                          metadata: pd.DataFrame) -> Dict[str, Any]:
        """
        Run full policy targeting analysis.
        
        Parameters
        ----------
        y_pred : array
            Model predictions
        y_baseline : array
            Baseline predictions
        y_true : array
            True values
        weights : array
            Sample weights
        metadata : DataFrame
            Metadata (must include TOTSQFT_EN)
            
        Returns
        -------
        dict
            Complete analysis results
        """
        with Timer("Running policy targeting analysis"):
            
            # Compute policy scores
            area = metadata['TOTSQFT_EN'].values if 'TOTSQFT_EN' in metadata else np.ones(len(y_pred))
            scores = self.compute_policy_scores(y_pred, y_baseline, area)
            
            # Compare weighted vs unweighted
            comparison = self.compare_weighted_vs_unweighted(scores, weights, metadata)
            
            # Add actual energy analysis (who we're actually missing)
            actual_high_use = y_true >= compute_weighted_quantile(y_true, weights, self.quantile)
            
            # Coverage analysis - how well do predicted candidates cover actual high users?
            coverage = {}
            for score_name in scores.columns:
                pred_candidates = self.identify_candidates(scores[score_name].values, weights, True)
                
                # True positive rate (sensitivity)
                tp = np.sum(pred_candidates & actual_high_use)
                coverage[score_name] = {
                    'true_positive_rate': tp / actual_high_use.sum() * 100 if actual_high_use.sum() > 0 else 0,
                    'false_positive_rate': (pred_candidates.sum() - tp) / pred_candidates.sum() * 100 if pred_candidates.sum() > 0 else 0
                }
            
            return {
                'scores': scores,
                'weighted_vs_unweighted': comparison,
                'coverage_analysis': coverage,
                'target_percentile': self.target_percentile
            }


class TargetingUncertainty:
    """
    Uncertainty quantification for policy targeting using replicate weights.
    """
    
    def __init__(self, n_replicates: int = 60):
        self.n_replicates = n_replicates
        
    def compute_targeting_uncertainty(self,
                                       scores: np.ndarray,
                                       weights: np.ndarray,
                                       replicate_weights: pd.DataFrame,
                                       target_percentile: float = 90) -> Dict[str, Any]:
        """
        Compute uncertainty in targeting metrics using jackknife.
        
        Parameters
        ----------
        scores : array
            Policy scores
        weights : array
            Main weights
        replicate_weights : DataFrame
            Replicate weight columns (NWEIGHT1-NWEIGHT60)
        target_percentile : float
            Target percentile
            
        Returns
        -------
        dict
            Uncertainty estimates
        """
        quantile = target_percentile / 100
        
        # Main estimate
        main_threshold = compute_weighted_quantile(scores, weights, quantile)
        main_candidates = scores >= main_threshold
        main_n_candidates = main_candidates.sum()
        
        # Replicate estimates
        rep_thresholds = []
        rep_n_candidates = []
        rep_jaccard_vs_main = []
        
        for col in replicate_weights.columns:
            rep_w = replicate_weights[col].values
            rep_threshold = compute_weighted_quantile(scores, rep_w, quantile)
            rep_candidates = scores >= rep_threshold
            
            rep_thresholds.append(rep_threshold)
            rep_n_candidates.append(rep_candidates.sum())
            
            # Jaccard vs main
            intersection = np.sum(main_candidates & rep_candidates)
            union = np.sum(main_candidates | rep_candidates)
            jaccard = intersection / union if union > 0 else 0
            rep_jaccard_vs_main.append(jaccard)
        
        rep_thresholds = np.array(rep_thresholds)
        rep_n_candidates = np.array(rep_n_candidates)
        rep_jaccard_vs_main = np.array(rep_jaccard_vs_main)
        
        # Jackknife variance estimation
        # Var = (n-1)/n * sum((theta_i - theta_bar)^2)
        n = len(rep_thresholds)
        
        threshold_var = (n - 1) / n * np.sum((rep_thresholds - rep_thresholds.mean()) ** 2)
        threshold_se = np.sqrt(threshold_var)
        
        n_candidates_var = (n - 1) / n * np.sum((rep_n_candidates - rep_n_candidates.mean()) ** 2)
        n_candidates_se = np.sqrt(n_candidates_var)
        
        jaccard_var = (n - 1) / n * np.sum((rep_jaccard_vs_main - rep_jaccard_vs_main.mean()) ** 2)
        jaccard_se = np.sqrt(jaccard_var)
        
        # 95% CI
        z = 1.96
        
        return {
            'threshold': {
                'estimate': main_threshold,
                'se': threshold_se,
                'ci_lower': main_threshold - z * threshold_se,
                'ci_upper': main_threshold + z * threshold_se
            },
            'n_candidates': {
                'estimate': main_n_candidates,
                'se': n_candidates_se,
                'ci_lower': main_n_candidates - z * n_candidates_se,
                'ci_upper': main_n_candidates + z * n_candidates_se
            },
            'jaccard_stability': {
                'mean': rep_jaccard_vs_main.mean(),
                'se': jaccard_se,
                'min': rep_jaccard_vs_main.min(),
                'max': rep_jaccard_vs_main.max()
            }
        }


def create_targeting_summary_table(analysis_results: Dict[str, Any],
                                    score_name: str = 'high_use') -> pd.DataFrame:
    """
    Create summary table for policy targeting results.
    
    Parameters
    ----------
    analysis_results : dict
        Results from PolicyTargeting.run_full_analysis()
    score_name : str
        Which score to summarize
        
    Returns
    -------
    DataFrame
        Summary table
    """
    if score_name not in analysis_results['weighted_vs_unweighted']:
        raise ValueError(f"Score {score_name} not found in results")
    
    score_results = analysis_results['weighted_vs_unweighted'][score_name]
    
    summary = {
        'Metric': [],
        'Value': []
    }
    
    # Overlap metrics
    overlap = score_results['overlap']
    summary['Metric'].extend([
        'Jaccard Index',
        'Overlap Rate',
        'Candidates Only in Weighted',
        'Candidates Only in Unweighted',
        'Weighted Threshold',
        'Unweighted Threshold'
    ])
    summary['Value'].extend([
        f"{overlap['jaccard_index']:.3f}",
        f"{overlap['overlap_rate']:.3f}",
        f"{overlap['only_weighted']} ({overlap['pct_only_weighted']:.1f}%)",
        f"{overlap['only_unweighted']} ({overlap['pct_only_unweighted']:.1f}%)",
        f"{score_results['weighted_threshold']:,.0f}",
        f"{score_results['unweighted_threshold']:,.0f}"
    ])
    
    return pd.DataFrame(summary)
