#!/usr/bin/env python3
"""
Heating Demand Modeling Framework - Main Analysis Script

This script runs the complete analysis pipeline for RECS 2020 heating demand modeling:
1. Data loading and preprocessing
2. Technology grouping
3. Feature engineering
4. Physics baselines
5. Nested cross-validation
6. Policy targeting analysis
7. Uncertainty quantification
8. Diagnostics and visualization

Usage:
    python run_analysis.py [--config CONFIG_PATH] [--output OUTPUT_DIR]
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.helpers import (
    load_config, ensure_dir, set_random_seed, 
    print_section_header, Timer, logger
)
from src.data.loader import RECSDataLoader
from src.data.preprocessor import RECSPreprocessor, preprocess_recs_data
from src.features.builder import FeatureBuilder, create_feature_matrix
from src.models.baselines import PhysicsBaselines, MonolithicBaseline
from src.models.main_models import HeatingDemandModels, LightGBMHeatingModel
from src.evaluation.metrics import WeightedMetrics, PhysicsDiagnostics, ErrorEquityAnalysis
from src.evaluation.nested_cv import NestedCrossValidator, compare_split_vs_monolithic
from src.evaluation.model_comparison import create_split_vs_monolithic_table, create_h1_summary_table
from src.policy.targeting import PolicyTargeting
from src.uncertainty.jackknife import JackknifeUncertainty, RefitSensitivity, CombinedUncertainty
from src.visualization.plots import HeatingDemandVisualizer, create_workflow_diagram

warnings.filterwarnings('ignore')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run heating demand modeling analysis')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output', type=str, default='outputs/',
                       help='Output directory')
    parser.add_argument('--skip-cv', action='store_true',
                       help='Skip nested CV (for quick testing)')
    parser.add_argument('--n-outer-folds', type=int, default=5,
                       help='Number of outer CV folds')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    return parser.parse_args()


def run_data_loading_and_preprocessing(config: dict) -> tuple:
    """
    Step 1-2: Load data and preprocess.
    
    Returns
    -------
    df : DataFrame
        Preprocessed data
    preprocessor : RECSPreprocessor
        Fitted preprocessor
    """
    print_section_header("STEP 1-2: DATA LOADING AND PREPROCESSING")
    
    # Load data
    data_path = config.get('data', {}).get('raw_path', 'data/raw/recs2020_public_v7.csv')
    loader = RECSDataLoader(data_path)
    df_raw = loader.load()
    
    # Preprocess
    df, preprocessor = preprocess_recs_data(
        df_raw,
        assignment_rule='primary_only',
        exclude_no_heating=True,
        exclude_hybrid=False,  # Keep for sensitivity analysis
        min_hdd=None
    )
    
    # Get technology group summary
    tech_summary = preprocessor.get_tech_group_summary(df)
    print("\nTechnology Group Summary:")
    print(tech_summary.to_string())
    
    # Create verification table
    verification = loader.create_verification_table()
    print("\nVerification Statistics by Division:")
    print(verification.to_string())
    
    return df, preprocessor, loader


def run_feature_engineering(df: pd.DataFrame, config: dict) -> tuple:
    """
    Step 3: Feature engineering.
    
    Returns
    -------
    X : DataFrame
        Feature matrix
    y : Series
        Target
    weights : Series
        Sample weights
    feature_builder : FeatureBuilder
        Fitted feature builder
    """
    print_section_header("STEP 3: FEATURE ENGINEERING")
    
    # Create feature builder
    feature_builder = FeatureBuilder(
        covid_control_mode='direct',  # Include COVID controls
        include_interactions=True,
        scale_continuous=False,  # Don't scale for tree models
        encode_categorical='ordinal'  # Ordinal for LightGBM
    )
    
    # Get feature columns
    feature_cols = feature_builder.get_feature_columns(include_tech_group=False)
    available_cols = [c for c in feature_cols if c in df.columns]
    
    print(f"Selected {len(available_cols)} features")
    
    # Create feature matrix
    X = df[available_cols].copy()
    y = df['TOTALBTUSPH'].copy()
    weights = df['NWEIGHT'].copy()
    
    # Fit feature builder
    X_transformed = feature_builder.fit_transform(X, y)
    
    print(f"Transformed feature matrix shape: {X_transformed.shape}")
    
    return X, y, weights, feature_builder


def run_physics_baselines(df: pd.DataFrame) -> tuple:
    """
    Step 4: Fit physics baselines.
    
    Returns
    -------
    baselines : PhysicsBaselines
        Fitted baselines
    baseline_results : DataFrame
        Baseline evaluation results
    """
    print_section_header("STEP 4: PHYSICS BASELINES")
    
    # Prepare data
    X = df[['HDD65', 'TOTSQFT_EN']].copy()
    y = df['TOTALBTUSPH'].copy()
    tech_group = df['tech_group'].copy()
    weights = df['NWEIGHT'].copy()
    
    # Fit baselines
    baselines = PhysicsBaselines()
    baselines.fit(X, y, tech_group, weights)
    
    # Evaluate
    baseline_results = baselines.evaluate(X, y, tech_group, weights)
    print("\nBaseline Performance by Technology Group:")
    print(baseline_results.to_string())
    
    # Get baseline predictions for excess demand calculation
    baseline_predictions = baselines.predict(X, tech_group)
    
    return baselines, baseline_results, baseline_predictions


def run_nested_cv(df: pd.DataFrame, 
                  feature_builder: FeatureBuilder,
                  config: dict,
                  n_outer_folds: int = 5) -> dict:
    """
    Step 5: Run nested cross-validation.
    
    Returns
    -------
    cv_results : dict
        Cross-validation results
    """
    print_section_header("STEP 5: NESTED CROSS-VALIDATION")
    
    # Prepare data
    X = df[[c for c in feature_builder.get_feature_columns() if c in df.columns]].copy()
    y = df['TOTALBTUSPH'].copy()
    weights = df['NWEIGHT'].copy()
    tech_group = df['tech_group'].copy()
    
    # Metadata for diagnostics
    metadata = df[['HDD65', 'DIVISION', 'TYPEHUQ', 'KOWNRENT', 'MONEYPY', 'TOTSQFT_EN', 'HDD_bin']].copy()
    
    # Define model factory
    def model_factory(**params):
        return LightGBMHeatingModel(params=params, use_monotonic_constraints=False)
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [15, 31, 63],
        'min_child_samples': [10, 20, 50]
    }
    
    # Run nested CV
    cv = NestedCrossValidator(
        outer_folds=n_outer_folds,
        inner_folds=3,
        n_search_iter=20,
        random_state=42
    )
    
    cv_result = cv.run(
        X, y, weights,
        model_factory=model_factory,
        param_grid=param_grid,
        feature_builder=feature_builder,
        tech_group=tech_group,
        additional_metadata=metadata
    )
    
    # Print results
    print("\nOverall CV Metrics:")
    for metric, value in cv_result.outer_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nMetrics by Fold:")
    print(cv_result.outer_metrics_by_fold.to_string())
    
    print("\nPhysics Diagnostics:")
    print(f"  Negative prediction rate: {cv_result.physics_diagnostics.get('negative_rate', 'N/A')}")
    if 'hdd_sensitivity' in cv_result.physics_diagnostics:
        hdd_sens = cv_result.physics_diagnostics['hdd_sensitivity']
        print(f"  HDD correlation: {hdd_sens.get('correlation', 'N/A'):.3f}")
        print(f"  Correct HDD direction: {hdd_sens.get('is_correct_direction', 'N/A')}")
    
    # Also run monolithic model for H1 comparison
    print("\nRunning Monolithic model for H1 comparison...")
    
    def mono_model_factory(**params):
        return LightGBMHeatingModel(params=params, use_monotonic_constraints=False)
    
    mono_cv = NestedCrossValidator(
        outer_folds=n_outer_folds,
        inner_folds=3,
        n_search_iter=20,
        random_state=42
    )
    
    mono_result = mono_cv.run(
        X, y, weights,
        model_factory=mono_model_factory,
        param_grid=param_grid,
        feature_builder=feature_builder,
        additional_metadata=metadata
        # Note: no tech_group = monolithic
    )
    
    # Paired comparison for H1 test
    from scipy import stats
    split_rmse = cv_result.outer_metrics_by_fold['weighted_rmse'].values
    mono_rmse = mono_result.outer_metrics_by_fold['weighted_rmse'].values
    
    t_stat, p_value = stats.ttest_rel(split_rmse, mono_rmse)
    
    h1_comparison = {
        'split_by_fold': cv_result.outer_metrics_by_fold,
        'mono_by_fold': mono_result.outer_metrics_by_fold,
        'split_metrics': cv_result.outer_metrics,
        'mono_metrics': mono_result.outer_metrics,
        'rmse_difference': mono_rmse.mean() - split_rmse.mean(),
        'rmse_improvement_pct': (mono_rmse.mean() - split_rmse.mean()) / mono_rmse.mean() * 100,
        'paired_t_statistic': t_stat,
        'paired_p_value': p_value,
        'mono_predictions': mono_cv.outer_predictions_
    }
    
    print(f"\nH1 Test (Split vs Monolithic):")
    print(f"  Split wRMSE: {split_rmse.mean():.0f} ± {split_rmse.std():.0f}")
    print(f"  Mono wRMSE: {mono_rmse.mean():.0f} ± {mono_rmse.std():.0f}")
    print(f"  Δ wRMSE: {h1_comparison['rmse_difference']:.0f} ({h1_comparison['rmse_improvement_pct']:.1f}%)")
    print(f"  Paired t-test p-value: {p_value:.4f}")
    
    return {
        'cv_result': cv_result,
        'predictions': cv.outer_predictions_,
        'fold_results': cv.fold_results_,
        'h1_comparison': h1_comparison,
        'mono_predictions': mono_cv.outer_predictions_
    }


def run_policy_analysis(df: pd.DataFrame,
                        predictions: np.ndarray,
                        baseline_predictions: np.ndarray) -> dict:
    """
    Step 6: Policy targeting analysis.
    
    Returns
    -------
    policy_results : dict
        Policy analysis results
    """
    print_section_header("STEP 6: POLICY TARGETING ANALYSIS")
    
    # Initialize targeting analyzer
    targeting = PolicyTargeting(target_percentile=90)
    
    # Run full analysis
    policy_results = targeting.run_full_analysis(
        y_pred=predictions,
        y_baseline=baseline_predictions,
        y_true=df['TOTALBTUSPH'].values,
        weights=df['NWEIGHT'].values,
        metadata=df
    )
    
    # Print results
    for score_name, results in policy_results['weighted_vs_unweighted'].items():
        print(f"\n{score_name.upper()} Score:")
        overlap = results['overlap']
        print(f"  Jaccard Index: {overlap['jaccard_index']:.3f}")
        print(f"  Overlap Rate: {overlap['overlap_rate']:.3f}")
        print(f"  Only in Weighted: {overlap['only_weighted']} ({overlap['pct_only_weighted']:.1f}%)")
        print(f"  Only in Unweighted: {overlap['only_unweighted']} ({overlap['pct_only_unweighted']:.1f}%)")
    
    return policy_results


def run_uncertainty_quantification(df: pd.DataFrame,
                                    predictions: np.ndarray,
                                    cv_results: dict,
                                    is_outer_fold: bool = True) -> dict:
    """
    Step 7: Uncertainty quantification.
    
    Returns
    -------
    uncertainty_results : dict
        Uncertainty quantification results
    """
    print_section_header("STEP 7: UNCERTAINTY QUANTIFICATION")
    
    # Get replicate weights
    replicate_cols = [f'NWEIGHT{i}' for i in range(1, 61)]
    available_rep_cols = [c for c in replicate_cols if c in df.columns]
    
    if not available_rep_cols:
        logger.warning("No replicate weights available, skipping uncertainty quantification")
        return {}
    
    replicate_weights = df[available_rep_cols].copy()
    
    # Compute jackknife uncertainty
    jackknife = JackknifeUncertainty(n_replicates=60)
    
    uncertainty_df = jackknife.compute_all_metrics_uncertainty(
        y_true=df['TOTALBTUSPH'].values,
        y_pred=predictions,
        main_weights=df['NWEIGHT'].values,
        replicate_weights=replicate_weights
    )
    
    print("\nMetric Uncertainty (Jackknife):")
    for _, row in uncertainty_df.iterrows():
        print(f"  {row['metric']}: {row['estimate']:.3f} ± {row['se']:.3f} "
              f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]")
    
    return {
        'uncertainty_df': uncertainty_df,
        'jackknife': jackknife,
        'is_outer_fold': is_outer_fold
    }


def run_diagnostics(df: pd.DataFrame,
                    predictions: np.ndarray,
                    cv_results: dict) -> dict:
    """
    Step 8: Diagnostics and error equity analysis.
    
    Returns
    -------
    diagnostic_results : dict
        Diagnostic results
    """
    print_section_header("STEP 8: DIAGNOSTICS AND ERROR EQUITY")
    
    y_true = df['TOTALBTUSPH'].values
    weights = df['NWEIGHT'].values
    
    # Physics diagnostics
    physics = PhysicsDiagnostics()
    physics_results = physics.run_all_diagnostics(
        y_true=y_true,
        y_pred=predictions,
        hdd=df['HDD65'].values,
        weights=weights,
        division=df['DIVISION'].values,
        tech_group=df['tech_group'].values
    )
    
    print("\nResidual Bias by HDD:")
    if 'bias_by_hdd' in physics_results:
        print(physics_results['bias_by_hdd'].to_string())
    
    # Error equity analysis
    equity = ErrorEquityAnalysis()
    equity_results = equity.run_full_audit(
        y_true=y_true,
        y_pred=predictions,
        weights=weights,
        metadata=df
    )
    
    print("\nError by Housing Type:")
    if 'by_housing_type' in equity_results:
        print(equity_results['by_housing_type'].to_string())
    
    print("\nError by Tenure:")
    if 'by_tenure' in equity_results:
        print(equity_results['by_tenure'].to_string())
    
    return {
        'physics': physics_results,
        'equity': equity_results
    }


def create_visualizations(df: pd.DataFrame,
                          predictions: np.ndarray,
                          cv_results: dict,
                          policy_results: dict,
                          diagnostic_results: dict,
                          output_dir: str) -> None:
    """
    Step 9: Create publication-quality visualizations.
    
    All figures follow publication standards:
    - Clear labeling of weighted vs unweighted, outer-fold vs in-sample
    - Human-readable labels for categorical codes
    - Uncertainty quantification where applicable
    """
    print_section_header("STEP 9: CREATING VISUALIZATIONS")
    
    viz = HeatingDemandVisualizer()
    figures_dir = Path(output_dir) / 'figures'
    ensure_dir(str(figures_dir))
    
    # Determine if results are from CV
    is_outer_fold = cv_results is not None and 'cv_result' in cv_results
    
    # Figure 1: Workflow diagram (FIXED: no dangling arrows, weights explicit)
    create_workflow_diagram(str(figures_dir / 'fig1_workflow.png'))
    
    # Figure 2: Predicted vs Observed (FIXED: calibration line, log-log, metrics labeled)
    fig = viz.plot_predicted_vs_observed(
        y_true=df['TOTALBTUSPH'].values,
        y_pred=predictions,
        weights=df['NWEIGHT'].values,
        tech_group=df['tech_group'].values,
        title="Predicted vs Observed Heating Energy",
        is_outer_fold=is_outer_fold
    )
    viz.save_figure(fig, 'fig2_pred_vs_obs.png', str(figures_dir))
    
    # Figure 2b: By technology group (supports H1)
    fig = viz.plot_predicted_vs_observed_by_tech(
        y_true=df['TOTALBTUSPH'].values,
        y_pred=predictions,
        weights=df['NWEIGHT'].values,
        tech_group=df['tech_group'].values,
        is_outer_fold=is_outer_fold
    )
    viz.save_figure(fig, 'fig2b_pred_vs_obs_by_tech.png', str(figures_dir))
    
    # Figure 3: Composition shifts (FIXED: human labels, Jaccard shown, differences)
    if 'weighted_vs_unweighted' in policy_results:
        high_use_results = policy_results['weighted_vs_unweighted'].get('high_use', {})
        if 'composition' in high_use_results and 'overlap' in high_use_results:
            fig = viz.plot_composition_shift(
                high_use_results['composition'],
                jaccard_index=high_use_results['overlap']['jaccard_index'],
                overlap_rate=high_use_results['overlap']['overlap_rate'],
                title="Composition Shift: Weighted vs Unweighted Targeting (High-Use Score)"
            )
            viz.save_figure(fig, 'fig3_composition_shift.png', str(figures_dir))
    
    # Figure 5: Residual vs HDD (FIXED: common y-axis, uncertainty bands, bin support)
    fig = viz.plot_residual_vs_hdd(
        y_true=df['TOTALBTUSPH'].values,
        y_pred=predictions,
        hdd=df['HDD65'].values,
        weights=df['NWEIGHT'].values,
        tech_group=df['tech_group'].values,
        title="Residuals vs HDD by Technology"
    )
    viz.save_figure(fig, 'fig5_residual_vs_hdd.png', str(figures_dir))
    
    # CV Results
    if cv_results and 'cv_result' in cv_results:
        fold_metrics = cv_results['cv_result'].outer_metrics_by_fold
        fig = viz.plot_cv_results(fold_metrics, model_name="LightGBM", 
                                   title="Nested Cross-Validation Results")
        viz.save_figure(fig, 'cv_results.png', str(figures_dir))
    
    # NEW: Figure 4 - H1 Comparison: Split vs Monolithic Residuals
    if cv_results and 'h1_comparison' in cv_results and 'mono_predictions' in cv_results:
        fig = viz.plot_residual_vs_hdd_comparison(
            y_true=df['TOTALBTUSPH'].values,
            y_pred_split=predictions,
            y_pred_mono=cv_results['mono_predictions'],
            hdd=df['HDD65'].values,
            weights=df['NWEIGHT'].values,
            tech_group=df['tech_group'].values,
            title="H1 Test: Residuals vs HDD — Split vs Monolithic"
        )
        viz.save_figure(fig, 'fig4_h1_split_vs_mono.png', str(figures_dir))
    
    # Error equity (FIXED: separate panels, nMAE, group sizes, no typos)
    if 'equity' in diagnostic_results:
        fig = viz.plot_error_equity(
            diagnostic_results['equity'],
            title="Error Equity Analysis"
        )
        viz.save_figure(fig, 'error_equity.png', str(figures_dir))
    
    logger.info(f"Saved all figures to {figures_dir}")


def save_results(df: pd.DataFrame,
                 preprocessor: RECSPreprocessor,
                 cv_results: dict,
                 policy_results: dict,
                 uncertainty_results: dict,
                 diagnostic_results: dict,
                 output_dir: str) -> None:
    """
    Save all results to publication-quality tables.
    
    All tables:
    - Remove spreadsheet artifacts (Unnamed columns)
    - Replace codes with human-readable labels
    - Declare weighted vs unweighted, out-of-sample vs in-sample
    - Include metric definitions and units
    """
    print_section_header("SAVING RESULTS")
    
    from src.visualization.tables import (
        create_table1_descriptives, create_uncertainty_table,
        create_policy_targeting_table, create_composition_table,
        create_equity_table, create_hdd_diagnostics_table,
        create_h1_comparison_table, create_physics_diagnostics_table,
        save_table_with_note
    )
    
    tables_dir = Path(output_dir) / 'tables'
    ensure_dir(str(tables_dir))
    
    # Table 1: Comprehensive technology group descriptives
    table1 = create_table1_descriptives(df, df['NWEIGHT'])
    save_table_with_note(
        table1, 
        str(tables_dir / 'table1_tech_group_descriptives.csv'),
        "Table 1: Descriptive statistics by technology group. All statistics are weighted "
        "using NWEIGHT unless otherwise noted."
    )
    print("\nTable 1: Technology Group Descriptives")
    print(table1.to_string())
    
    # Table 2: CV performance (if available)
    if cv_results and 'cv_result' in cv_results:
        fold_df = cv_results['cv_result'].outer_metrics_by_fold.copy()
        fold_df = fold_df.loc[:, ~fold_df.columns.str.contains('^Unnamed')]
        
        # Add summary row
        summary_row = {col: f"{fold_df[col].mean():.3f} ± {fold_df[col].std():.3f}" 
                      for col in fold_df.columns if col != 'fold'}
        summary_row['fold'] = 'Mean ± SD'
        fold_df = pd.concat([fold_df, pd.DataFrame([summary_row])], ignore_index=True)
        
        save_table_with_note(
            fold_df,
            str(tables_dir / 'table2_cv_performance.csv'),
            "Table 2: Nested CV performance (outer-fold, weighted metrics). "
            "wRMSE and wMAE in kBTU. All metrics computed on out-of-sample predictions."
        )
    
    # NEW: Table 2b: H1 Comparison (Monolithic vs Split)
    if cv_results and 'h1_comparison' in cv_results:
        h1_table = create_h1_comparison_table(cv_results['h1_comparison'])
        save_table_with_note(
            h1_table,
            str(tables_dir / 'table2b_h1_split_vs_mono.csv'),
            h1_table.attrs.get('note', '')
        )
        print("\nTable 2b: H1 Comparison (Split vs Monolithic)")
        print(h1_table.to_string())
    
    # NEW: Table - Physics Diagnostics by Technology
    if cv_results and 'cv_result' in cv_results:
        predictions = cv_results.get('predictions', np.zeros(len(df)))
        physics_table = create_physics_diagnostics_table(
            y_true=df['TOTALBTUSPH'].values,
            y_pred=predictions,
            weights=df['NWEIGHT'].values,
            tech_group=df['tech_group'].values,
            hdd=df['HDD65'].values
        )
        save_table_with_note(
            physics_table,
            str(tables_dir / 'table_physics_diagnostics.csv'),
            physics_table.attrs.get('note', '')
        )
        print("\nPhysics Diagnostics by Technology:")
        print(physics_table.to_string())
    
    # Table 3: Uncertainty with proper formatting
    if uncertainty_results and 'uncertainty_df' in uncertainty_results:
        table3 = create_uncertainty_table(uncertainty_results['uncertainty_df'])
        # Use correct note based on whether CV was run
        is_oof = uncertainty_results.get('is_outer_fold', False)
        sample_type = "outer-fold test predictions" if is_oof else "in-sample predictions (CV skipped)"
        note = (
            f"All metrics computed on {sample_type} with NWEIGHT. "
            "SE and 95% CI from RECS replicate-weight jackknife (n=60). "
            "MAPE excluded (unstable for small denominators)."
        )
        save_table_with_note(
            table3,
            str(tables_dir / 'table3_uncertainty.csv'),
            note
        )
    
    # Policy targeting tables
    if policy_results and 'weighted_vs_unweighted' in policy_results:
        for score_name in ['high_use', 'high_intensity', 'excess_demand']:
            if score_name in policy_results['weighted_vs_unweighted']:
                # Summary table
                summary_table = create_policy_targeting_table(policy_results, score_name)
                if len(summary_table) > 0:
                    save_table_with_note(
                        summary_table,
                        str(tables_dir / f'policy_{score_name}_summary.csv'),
                        summary_table.attrs.get('note', '')
                    )
                
                # Composition tables with human-readable labels
                results = policy_results['weighted_vs_unweighted'][score_name]
                if 'composition' in results:
                    for group_name, comp_df in results['composition'].items():
                        if comp_df is not None and len(comp_df) > 0:
                            formatted_comp = create_composition_table(comp_df, group_name)
                            save_table_with_note(
                                formatted_comp,
                                str(tables_dir / f'policy_{score_name}_{group_name}.csv'),
                                f"Composition shift for {score_name} targeting by {group_name}. "
                                "Representation Ratio = Share among candidates / Population share."
                            )
    
    # Equity tables with normalized metrics
    if diagnostic_results and 'equity' in diagnostic_results:
        for name, equity_df in diagnostic_results['equity'].items():
            if equity_df is not None and len(equity_df) > 0:
                group_type = name.replace('by_', '')
                formatted_equity = create_equity_table(equity_df, group_type)
                save_table_with_note(
                    formatted_equity,
                    str(tables_dir / f'equity_{name}.csv'),
                    formatted_equity.attrs.get('note', '')
                )
    
    # HDD diagnostics with bin support
    if diagnostic_results and 'physics' in diagnostic_results:
        if 'bias_by_hdd' in diagnostic_results['physics']:
            hdd_table = create_hdd_diagnostics_table(diagnostic_results['physics']['bias_by_hdd'])
            save_table_with_note(
                hdd_table,
                str(tables_dir / 'diagnostics_bias_by_hdd.csv'),
                hdd_table.attrs.get('note', '')
            )
    
    logger.info(f"Saved all tables to {tables_dir}")


def main():
    """Main entry point."""
    args = parse_args()
    
    print_section_header("HEATING DEMAND MODELING FRAMEWORK", char="=", width=80)
    print("RECS 2020 Analysis")
    print("="*80)
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Load config
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logger.warning(f"Config file not found: {args.config}, using defaults")
        config = {}
    
    # Create output directories
    ensure_dir(args.output)
    ensure_dir(f"{args.output}/figures")
    ensure_dir(f"{args.output}/tables")
    ensure_dir(f"{args.output}/models")
    
    # Run pipeline
    with Timer("Complete analysis pipeline"):
        
        # Step 1-2: Data loading and preprocessing
        df, preprocessor, loader = run_data_loading_and_preprocessing(config)
        
        # Step 3: Feature engineering
        X, y, weights, feature_builder = run_feature_engineering(df, config)
        
        # Step 4: Physics baselines
        baselines, baseline_results, baseline_predictions = run_physics_baselines(df)
        
        # Step 5: Nested CV (can be skipped for quick testing)
        if not args.skip_cv:
            cv_results = run_nested_cv(df, feature_builder, config, args.n_outer_folds)
            predictions = cv_results['predictions']
            is_outer_fold = True  # Proper outer-fold out-of-sample predictions
        else:
            logger.info("Skipping nested CV (--skip-cv flag)")
            # Use simple train/predict for testing
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
                X, y, weights, test_size=0.2, random_state=42
            )
            model = LightGBMHeatingModel()
            X_train_t = feature_builder.fit_transform(X_train)
            X_test_t = feature_builder.transform(X_test)
            model.fit(X_train_t, y_train.values, sample_weight=w_train.values)
            predictions = np.zeros(len(df))
            predictions[X_test.index] = model.predict(X_test_t)
            predictions[X_train.index] = model.predict(X_train_t)
            cv_results = None
            is_outer_fold = False  # In-sample predictions (CV skipped)
        
        # Step 6: Policy analysis
        policy_results = run_policy_analysis(df, predictions, baseline_predictions)
        
        # Step 7: Uncertainty quantification
        uncertainty_results = run_uncertainty_quantification(df, predictions, cv_results, 
                                                              is_outer_fold=is_outer_fold)
        
        # Step 8: Diagnostics
        diagnostic_results = run_diagnostics(df, predictions, cv_results)
        
        # Step 9: Visualizations
        create_visualizations(df, predictions, cv_results, policy_results, 
                             diagnostic_results, args.output)
        
        # Save results
        save_results(df, preprocessor, cv_results, policy_results,
                    uncertainty_results, diagnostic_results, args.output)
    
    print_section_header("ANALYSIS COMPLETE", char="=", width=80)
    print(f"Results saved to: {args.output}")
    print("="*80)


if __name__ == '__main__':
    main()
