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
            
            # Equal-budget comparison (fixed number of candidates)
            # This is the "fair" policy comparison
            n_budget = candidates_weighted.sum()  # Use weighted count as budget
            
            # Select top-N by weighted ranking
            weighted_ranks = np.argsort(-score_values * weights / weights.sum())
            candidates_weighted_topN = np.zeros(len(score_values), dtype=bool)
            candidates_weighted_topN[weighted_ranks[:n_budget]] = True
            
            # Select top-N by unweighted ranking  
            unweighted_ranks = np.argsort(-score_values)
            candidates_unweighted_topN = np.zeros(len(score_values), dtype=bool)
            candidates_unweighted_topN[unweighted_ranks[:n_budget]] = True
            
            # Overlap for equal-budget comparison
            overlap_equal_budget = self._compute_overlap_metrics(
                candidates_weighted_topN, candidates_unweighted_topN
            )
            
            results[score_name] = {
                'overlap': overlap_metrics,
                'overlap_equal_budget': overlap_equal_budget,  # NEW: equal-budget comparison
                'composition': composition,
                'n_weighted_candidates': candidates_weighted.sum(),
                'n_unweighted_candidates': candidates_unweighted.sum(),
                'n_budget': n_budget,
                'weighted_threshold': compute_weighted_quantile(score_values, weights, self.quantile),
                'unweighted_threshold': np.quantile(score_values[~np.isnan(score_values)], self.quantile),
                'target_pct': 100 - self.target_percentile  # 10% for top 10%
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
        # Counts
        n_a = candidates_a.sum()
        n_b = candidates_b.sum()
        intersection = np.sum(candidates_a & candidates_b)
        union = np.sum(candidates_a | candidates_b)
        
        # Jaccard index: |A ∩ B| / |A ∪ B|
        jaccard = intersection / union if union > 0 else 0
        
        # Dice coefficient (Sørensen–Dice): 2|A ∩ B| / (|A| + |B|)
        # Measures overlap as harmonic mean of precision and recall
        dice = 2 * intersection / (n_a + n_b) if (n_a + n_b) > 0 else 0
        
        # Recall (from perspective of weighted candidates)
        # What fraction of weighted candidates are also in unweighted?
        recall_weighted = intersection / n_a if n_a > 0 else 0
        
        # Precision (from perspective of unweighted candidates)
        # What fraction of unweighted candidates are also in weighted?
        precision_unweighted = intersection / n_b if n_b > 0 else 0
        
        # Containment coefficient: |A ∩ B| / min(|A|, |B|)
        # Equals 1.0 if smaller set is fully contained in larger
        min_size = min(n_a, n_b)
        containment = intersection / min_size if min_size > 0 else 0
        
        # Proportion only in weighted
        only_weighted = np.sum(candidates_a & ~candidates_b)
        # Proportion only in unweighted
        only_unweighted = np.sum(~candidates_a & candidates_b)
        
        return {
            'jaccard_index': jaccard,
            'dice_coefficient': dice,  # F1-like: harmonic mean of precision/recall
            'recall_weighted': recall_weighted,  # Fraction of weighted also in unweighted
            'precision_unweighted': precision_unweighted,  # Fraction of unweighted also in weighted
            'containment': containment,  # Containment coefficient
            'overlap_rate': dice,  # Alias for backward compatibility (now Dice)
            'intersection': intersection,
            'union': union,
            'n_weighted': n_a,
            'n_unweighted': n_b,
            'only_weighted': only_weighted,
            'only_unweighted': only_unweighted,
            'pct_only_weighted': only_weighted / n_a * 100 if n_a > 0 else 0,
            'pct_only_unweighted': only_unweighted / n_b * 100 if n_b > 0 else 0
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
        Compute composition by group with selection rates and representation ratios.
        
        Includes:
        - Candidate share among all candidates
        - Within-group selection rate (% of group selected)
        - Representation ratio (candidate share / population share)
        """
        results = []
        total_weight = weights.sum()
        total_weighted_candidates = weights[candidates_weighted].sum()
        total_unweighted_candidates = candidates_unweighted.sum()
        
        for group_val in np.unique(groups):
            if pd.isna(group_val):
                continue
            
            group_mask = groups == group_val
            group_weight = weights[group_mask].sum()
            n_in_group = group_mask.sum()
            
            # Population share (weighted)
            pop_share = group_weight / total_weight * 100
            
            # Weighted share in weighted candidates
            w_in_weighted_cands = weights[candidates_weighted & group_mask].sum()
            share_weighted = w_in_weighted_cands / total_weighted_candidates * 100 if total_weighted_candidates > 0 else 0
            
            # Share in unweighted candidates
            n_in_unweighted_cands = (candidates_unweighted & group_mask).sum()
            share_unweighted = n_in_unweighted_cands / total_unweighted_candidates * 100 if total_unweighted_candidates > 0 else 0
            
            # Within-group selection rate (weighted)
            # What % of this group gets selected as candidates?
            within_group_selection_rate = w_in_weighted_cands / group_weight * 100 if group_weight > 0 else 0
            
            # Representation ratio (candidate share / population share)
            # >1 = overrepresented, <1 = underrepresented
            representation_ratio = share_weighted / pop_share if pop_share > 0 else 0
            
            results.append({
                f'{group_name}': group_val,
                'n_in_group': n_in_group,
                'population_share': pop_share,
                'share_weighted_candidates': share_weighted,
                'share_unweighted_candidates': share_unweighted,
                'share_difference': share_weighted - share_unweighted,
                'within_group_selection_rate': within_group_selection_rate,
                'representation_ratio': representation_ratio
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
    
    Computes 95% CIs for:
    - Jaccard index
    - Overlap rate
    - Subgroup shares
    - Threshold values
    """
    
    def __init__(self, n_replicates: int = 60, confidence_level: float = 0.95):
        self.n_replicates = n_replicates
        self.confidence_level = confidence_level
        self.z_score = 1.96 if confidence_level == 0.95 else 1.645
        
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
        z = self.z_score
        
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
            'jaccard': {
                'estimate': rep_jaccard_vs_main.mean(),
                'se': jaccard_se,
                'ci_lower': max(0, rep_jaccard_vs_main.mean() - z * jaccard_se),
                'ci_upper': min(1, rep_jaccard_vs_main.mean() + z * jaccard_se),
                'min': rep_jaccard_vs_main.min(),
                'max': rep_jaccard_vs_main.max()
            }
        }
    
    def compute_jaccard_overlap_with_ci(self,
                                         scores: np.ndarray,
                                         weights: np.ndarray,
                                         replicate_weights: pd.DataFrame,
                                         target_percentile: float = 90) -> Dict[str, Any]:
        """
        Compute Jaccard index and Overlap rate with 95% CI.
        
        Compares weighted vs unweighted candidate lists.
        
        Parameters
        ----------
        scores : array
            Policy scores
        weights : array
            Main weights (NWEIGHT)
        replicate_weights : DataFrame
            Replicate weights (NWEIGHT1-60)
        target_percentile : float
            Target percentile
            
        Returns
        -------
        dict
            Jaccard and Overlap with 95% CIs
        """
        quantile = target_percentile / 100
        z = self.z_score
        
        # Main estimates
        threshold_w = compute_weighted_quantile(scores, weights, quantile)
        threshold_uw = np.quantile(scores[~np.isnan(scores)], quantile)
        
        candidates_w = scores >= threshold_w
        candidates_uw = scores >= threshold_uw
        
        intersection = np.sum(candidates_w & candidates_uw)
        union = np.sum(candidates_w | candidates_uw)
        main_jaccard = intersection / union if union > 0 else 0
        
        min_size = min(candidates_w.sum(), candidates_uw.sum())
        main_overlap = intersection / min_size if min_size > 0 else 0
        
        # Replicate estimates
        rep_jaccards = []
        rep_overlaps = []
        
        for col in replicate_weights.columns[:self.n_replicates]:
            rep_w = replicate_weights[col].values
            rep_threshold = compute_weighted_quantile(scores, rep_w, quantile)
            rep_candidates = scores >= rep_threshold
            
            rep_inter = np.sum(rep_candidates & candidates_uw)
            rep_union = np.sum(rep_candidates | candidates_uw)
            rep_jaccard = rep_inter / rep_union if rep_union > 0 else 0
            
            rep_min = min(rep_candidates.sum(), candidates_uw.sum())
            rep_overlap = rep_inter / rep_min if rep_min > 0 else 0
            
            rep_jaccards.append(rep_jaccard)
            rep_overlaps.append(rep_overlap)
        
        rep_jaccards = np.array(rep_jaccards)
        rep_overlaps = np.array(rep_overlaps)
        n = len(rep_jaccards)
        
        # Jackknife variance
        jaccard_var = (n - 1) / n * np.sum((rep_jaccards - rep_jaccards.mean()) ** 2)
        jaccard_se = np.sqrt(jaccard_var)
        
        overlap_var = (n - 1) / n * np.sum((rep_overlaps - rep_overlaps.mean()) ** 2)
        overlap_se = np.sqrt(overlap_var)
        
        return {
            'jaccard': {
                'estimate': main_jaccard,
                'se': jaccard_se,
                'ci_lower': max(0, main_jaccard - z * jaccard_se),
                'ci_upper': min(1, main_jaccard + z * jaccard_se)
            },
            'overlap': {
                'estimate': main_overlap,
                'se': overlap_se,
                'ci_lower': max(0, main_overlap - z * overlap_se),
                'ci_upper': min(1, main_overlap + z * overlap_se)
            },
            'n_weighted_candidates': int(candidates_w.sum()),
            'n_unweighted_candidates': int(candidates_uw.sum())
        }


def create_targeting_summary_table(analysis_results: Dict[str, Any],
                                    score_name: str = 'high_use',
                                    uncertainty_results: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Create summary table for policy targeting results with uncertainty.
    
    Parameters
    ----------
    analysis_results : dict
        Results from PolicyTargeting.run_full_analysis()
    score_name : str
        Which score to summarize
    uncertainty_results : dict, optional
        Uncertainty results from TargetingUncertainty
        
    Returns
    -------
    DataFrame
        Summary table with CIs
    """
    if score_name not in analysis_results['weighted_vs_unweighted']:
        raise ValueError(f"Score {score_name} not found in results")
    
    score_results = analysis_results['weighted_vs_unweighted'][score_name]
    overlap = score_results['overlap']
    
    rows = []
    
    # Jaccard Index with CI if available
    jaccard_val = f"{overlap['jaccard_index']:.3f}"
    if uncertainty_results and 'jaccard' in uncertainty_results:
        jac = uncertainty_results['jaccard']
        jaccard_val = f"{jac['estimate']:.3f} [{jac['ci_lower']:.3f}, {jac['ci_upper']:.3f}]"
    rows.append({'Metric': 'Jaccard Index', 'Value': jaccard_val, 
                 'Description': 'Intersection / Union of candidate sets'})
    
    # Overlap Rate with CI if available
    overlap_val = f"{overlap['overlap_rate']:.3f}"
    if uncertainty_results and 'overlap' in uncertainty_results:
        ovl = uncertainty_results['overlap']
        overlap_val = f"{ovl['estimate']:.3f} [{ovl['ci_lower']:.3f}, {ovl['ci_upper']:.3f}]"
    rows.append({'Metric': 'Overlap Rate', 'Value': overlap_val,
                 'Description': 'Dice coefficient: 2×|A∩B| / (|A|+|B|)'})
    
    # Recall and Containment metrics
    if 'recall_weighted' in overlap:
        rows.append({'Metric': 'Recall (Weighted)', 
                    'Value': f"{overlap['recall_weighted']:.3f}",
                    'Description': 'Fraction of weighted candidates also in unweighted'})
    if 'containment' in overlap:
        rows.append({'Metric': 'Containment',
                    'Value': f"{overlap['containment']:.3f}",
                    'Description': 'Intersection / min(|A|, |B|)'})
    
    # Other metrics
    rows.extend([
        {'Metric': 'Only in Weighted', 
         'Value': f"{overlap['only_weighted']} ({overlap['pct_only_weighted']:.1f}%)",
         'Description': 'Candidates selected only with weights'},
        {'Metric': 'Only in Unweighted',
         'Value': f"{overlap['only_unweighted']} ({overlap['pct_only_unweighted']:.1f}%)",
         'Description': 'Candidates selected only without weights'},
    ])
    
    # Thresholds with correct units
    if 'high_intensity' in score_name:
        threshold_unit = 'kBTU/ft²'
    else:
        threshold_unit = 'kBTU'
    
    rows.extend([
        {'Metric': 'Weighted Threshold',
         'Value': f"{score_results['weighted_threshold']:,.0f} {threshold_unit}",
         'Description': f"Weighted 90th percentile cutoff"},
        {'Metric': 'Unweighted Threshold',
         'Value': f"{score_results['unweighted_threshold']:,.0f} {threshold_unit}",
         'Description': f"Unweighted 90th percentile cutoff"}
    ])
    
    result_df = pd.DataFrame(rows)
    result_df.attrs['note'] = (
        f"Policy score: {score_name}. Top 10% candidates defined by weighted/unweighted "
        "90th percentile. 95% CIs from replicate-weight jackknife (n=60)."
    )
    
    return result_df
