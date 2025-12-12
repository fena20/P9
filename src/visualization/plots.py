"""
Visualization Utilities for Heating Demand Modeling

Creates publication-quality figures as specified in Section 13.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any

from src.utils.helpers import logger


class HeatingDemandVisualizer:
    """
    Visualization utilities for heating demand analysis.
    
    Planned figures (Section 13):
    - Figure 1: Workflow diagram
    - Figure 2: Predicted vs observed (hexbin) + y=x
    - Figure 3: Candidate list mismatch (Jaccard/overlap) + composition shifts
    - Figure 4: EBM shapes (HDD, insulation, vintage) by tech
    - Figure 5: Residual bias vs HDD bins (by tech)
    """
    
    def __init__(self, 
                 figsize: Tuple[int, int] = (10, 8),
                 dpi: int = 300,
                 style: str = 'seaborn-v0_8-whitegrid'):
        """
        Initialize visualizer.
        
        Parameters
        ----------
        figsize : tuple
            Default figure size
        dpi : int
            Resolution for saving
        style : str
            Matplotlib style
        """
        self.figsize = figsize
        self.dpi = dpi
        self.style = style
        
        # Set style
        try:
            plt.style.use(style)
        except:
            plt.style.use('seaborn-whitegrid')
        
        # Color palette for tech groups
        self.tech_colors = {
            'combustion': '#E24A33',
            'electric_heat_pump': '#348ABD',
            'electric_resistance': '#988ED5',
            'hybrid_ambiguous': '#777777'
        }
        
    def plot_predicted_vs_observed(self,
                                    y_true: np.ndarray,
                                    y_pred: np.ndarray,
                                    weights: Optional[np.ndarray] = None,
                                    tech_group: Optional[np.ndarray] = None,
                                    title: str = "Predicted vs Observed Heating Energy",
                                    ax: Optional[plt.Axes] = None) -> plt.Figure:
        """
        Create predicted vs observed scatter/hexbin plot.
        
        Parameters
        ----------
        y_true : array
            True values
        y_pred : array
            Predicted values
        weights : array, optional
            Sample weights for sizing
        tech_group : array, optional
            Technology group labels for coloring
        title : str
            Plot title
        ax : Axes, optional
            Matplotlib axes
            
        Returns
        -------
        Figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure
        
        # Determine plot type based on sample size
        n_points = len(y_true)
        
        if n_points > 5000:
            # Use hexbin for large datasets
            hb = ax.hexbin(y_true, y_pred, gridsize=50, cmap='Blues', mincnt=1)
            plt.colorbar(hb, ax=ax, label='Count')
        else:
            # Use scatter for smaller datasets
            if tech_group is not None:
                for tech in np.unique(tech_group):
                    mask = tech_group == tech
                    ax.scatter(y_true[mask], y_pred[mask], 
                              alpha=0.5, label=tech, s=20,
                              c=self.tech_colors.get(tech, 'gray'))
                ax.legend(title='Technology')
            else:
                ax.scatter(y_true, y_pred, alpha=0.5, s=20)
        
        # Add y=x line
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect prediction')
        
        ax.set_xlabel('Observed Energy (kBTU)')
        ax.set_ylabel('Predicted Energy (kBTU)')
        ax.set_title(title)
        
        # Add R² annotation
        from src.evaluation.metrics import WeightedMetrics
        metrics = WeightedMetrics()
        if weights is not None:
            r2 = metrics.weighted_r2(y_true, y_pred, weights)
            rmse = metrics.weighted_rmse(y_true, y_pred, weights)
        else:
            r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2)
            rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        
        ax.annotate(f'R² = {r2:.3f}\nRMSE = {rmse:,.0f}', 
                   xy=(0.05, 0.95), xycoords='axes fraction',
                   fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def plot_residual_vs_hdd(self,
                             y_true: np.ndarray,
                             y_pred: np.ndarray,
                             hdd: np.ndarray,
                             weights: np.ndarray,
                             tech_group: Optional[np.ndarray] = None,
                             title: str = "Residuals vs HDD",
                             ax: Optional[plt.Axes] = None) -> plt.Figure:
        """
        Plot residuals vs HDD with binned means.
        
        Parameters
        ----------
        y_true : array
            True values
        y_pred : array
            Predicted values
        hdd : array
            Heating degree days
        weights : array
            Sample weights
        tech_group : array, optional
            Technology groups for separate panels
        title : str
            Plot title
        ax : Axes, optional
            Matplotlib axes
            
        Returns
        -------
        Figure
        """
        residuals = y_pred - y_true
        
        if tech_group is not None:
            unique_groups = [g for g in np.unique(tech_group) 
                           if g not in ['no_heating', 'unknown']]
            n_groups = len(unique_groups)
            
            fig, axes = plt.subplots(1, n_groups, figsize=(5*n_groups, 5))
            if n_groups == 1:
                axes = [axes]
            
            for ax, group in zip(axes, unique_groups):
                mask = tech_group == group
                
                ax.scatter(hdd[mask], residuals[mask], 
                          alpha=0.3, s=10, c=self.tech_colors.get(group, 'gray'))
                ax.axhline(y=0, color='red', linestyle='--', lw=2)
                
                # Binned means
                hdd_bins = pd.cut(hdd[mask], bins=8)
                bin_means = pd.DataFrame({
                    'hdd': hdd[mask],
                    'residual': residuals[mask],
                    'weight': weights[mask],
                    'bin': hdd_bins
                }).groupby('bin').apply(
                    lambda x: np.average(x['residual'], weights=x['weight'])
                )
                
                bin_centers = [b.mid for b in bin_means.index]
                ax.plot(bin_centers, bin_means.values, 'ko-', lw=2, ms=8, label='Weighted mean')
                
                ax.set_xlabel('HDD')
                ax.set_ylabel('Residual (Pred - Obs)')
                ax.set_title(f'{group}')
                ax.legend()
            
            fig.suptitle(title, fontsize=14)
            
        else:
            if ax is None:
                fig, ax = plt.subplots(figsize=self.figsize)
            else:
                fig = ax.figure
            
            ax.scatter(hdd, residuals, alpha=0.3, s=10)
            ax.axhline(y=0, color='red', linestyle='--', lw=2)
            
            ax.set_xlabel('HDD')
            ax.set_ylabel('Residual (Pred - Obs)')
            ax.set_title(title)
        
        plt.tight_layout()
        return fig
    
    def plot_composition_shift(self,
                               composition_results: Dict[str, pd.DataFrame],
                               title: str = "Composition Shift: Weighted vs Unweighted Targeting",
                               ax: Optional[plt.Axes] = None) -> plt.Figure:
        """
        Plot composition shifts between weighted and unweighted targeting.
        
        Parameters
        ----------
        composition_results : dict
            Results from PolicyTargeting composition analysis
        title : str
            Plot title
        ax : Axes, optional
            Matplotlib axes
            
        Returns
        -------
        Figure
        """
        n_groups = len(composition_results)
        fig, axes = plt.subplots(1, n_groups, figsize=(5*n_groups, 5))
        
        if n_groups == 1:
            axes = [axes]
        
        for ax, (group_name, df) in zip(axes, composition_results.items()):
            if df is None or len(df) == 0:
                continue
            
            x = np.arange(len(df))
            width = 0.35
            
            ax.bar(x - width/2, df['share_weighted_candidates'], width, 
                   label='Weighted', color='steelblue')
            ax.bar(x + width/2, df['share_unweighted_candidates'], width,
                   label='Unweighted', color='coral')
            
            ax.set_xlabel(group_name.replace('_', ' ').title())
            ax.set_ylabel('Share of Candidates (%)')
            ax.set_title(f'By {group_name.replace("_", " ").title()}')
            ax.set_xticks(x)
            
            # Get labels from first column
            label_col = df.columns[0]
            ax.set_xticklabels(df[label_col].astype(str), rotation=45, ha='right')
            ax.legend()
        
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        return fig
    
    def plot_ebm_shape(self,
                       feature_name: str,
                       bins: np.ndarray,
                       scores: np.ndarray,
                       title: Optional[str] = None,
                       ax: Optional[plt.Axes] = None) -> plt.Figure:
        """
        Plot EBM shape function.
        
        Parameters
        ----------
        feature_name : str
            Feature name
        bins : array
            Bin edges
        scores : array
            Shape function scores
        title : str, optional
            Plot title
        ax : Axes, optional
            Matplotlib axes
            
        Returns
        -------
        Figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.figure
        
        # Handle bins formatting
        if bins is not None and len(bins) > 0:
            if len(bins) == len(scores) + 1:
                # Bin edges
                bin_centers = (bins[:-1] + bins[1:]) / 2
            else:
                bin_centers = np.arange(len(scores))
        else:
            bin_centers = np.arange(len(scores))
        
        ax.plot(bin_centers, scores, 'b-', lw=2)
        ax.fill_between(bin_centers, 0, scores, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', lw=1)
        
        ax.set_xlabel(feature_name)
        ax.set_ylabel('Contribution to Prediction')
        ax.set_title(title or f'Shape Function: {feature_name}')
        
        plt.tight_layout()
        return fig
    
    def plot_tech_group_summary(self,
                                summary_df: pd.DataFrame,
                                title: str = "Technology Group Summary") -> plt.Figure:
        """
        Plot summary statistics by technology group.
        
        Parameters
        ----------
        summary_df : DataFrame
            Summary statistics by tech group
        title : str
            Plot title
            
        Returns
        -------
        Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        groups = summary_df['tech_group'].values
        x = np.arange(len(groups))
        
        # Weighted mean energy
        ax = axes[0, 0]
        colors = [self.tech_colors.get(g, 'gray') for g in groups]
        ax.bar(x, summary_df['weighted_mean_energy'], color=colors)
        ax.set_ylabel('Weighted Mean Energy (kBTU)')
        ax.set_title('Mean Heating Energy')
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=45, ha='right')
        
        # Sample sizes
        ax = axes[0, 1]
        ax.bar(x, summary_df['n_households'], color=colors)
        ax.set_ylabel('Number of Households')
        ax.set_title('Sample Size')
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=45, ha='right')
        
        # Mean HDD
        ax = axes[1, 0]
        ax.bar(x, summary_df['weighted_mean_hdd'], color=colors)
        ax.set_ylabel('Weighted Mean HDD')
        ax.set_title('Climate (HDD)')
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=45, ha='right')
        
        # Mean sqft
        ax = axes[1, 1]
        ax.bar(x, summary_df['weighted_mean_sqft'], color=colors)
        ax.set_ylabel('Weighted Mean Sqft')
        ax.set_title('Floor Area')
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=45, ha='right')
        
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        return fig
    
    def plot_cv_results(self,
                        fold_metrics: pd.DataFrame,
                        title: str = "Cross-Validation Results") -> plt.Figure:
        """
        Plot CV results across folds.
        
        Parameters
        ----------
        fold_metrics : DataFrame
            Metrics by fold
        title : str
            Plot title
            
        Returns
        -------
        Figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        metrics_to_plot = ['weighted_rmse', 'weighted_mae', 'weighted_r2']
        
        for ax, metric in zip(axes, metrics_to_plot):
            if metric not in fold_metrics.columns:
                continue
            
            folds = fold_metrics['fold'].values if 'fold' in fold_metrics.columns else np.arange(len(fold_metrics))
            values = fold_metrics[metric].values
            
            ax.bar(folds, values, color='steelblue', alpha=0.7)
            ax.axhline(y=values.mean(), color='red', linestyle='--', lw=2, label=f'Mean: {values.mean():.3f}')
            
            ax.set_xlabel('Fold')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(metric.replace('_', ' ').title())
            ax.legend()
        
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        return fig
    
    def plot_error_equity(self,
                          equity_results: Dict[str, pd.DataFrame],
                          title: str = "Error Equity Analysis") -> plt.Figure:
        """
        Plot error equity analysis.
        
        Parameters
        ----------
        equity_results : dict
            Results from ErrorEquityAnalysis
        title : str
            Plot title
            
        Returns
        -------
        Figure
        """
        n_groups = len(equity_results)
        fig, axes = plt.subplots(1, n_groups, figsize=(5*n_groups, 5))
        
        if n_groups == 1:
            axes = [axes]
        
        for ax, (group_name, df) in zip(axes, equity_results.items()):
            if df is None or len(df) == 0:
                continue
            
            # Get first column as labels
            label_col = df.columns[0]
            x = np.arange(len(df))
            
            # Plot bias and MAE
            width = 0.35
            ax.bar(x - width/2, df['weighted_bias'], width, label='Bias', color='coral')
            ax.bar(x + width/2, df['weighted_mae'], width, label='MAE', color='steelblue')
            
            ax.axhline(y=0, color='gray', linestyle='--', lw=1)
            
            ax.set_xlabel(group_name.replace('_', ' ').title())
            ax.set_ylabel('Error (kBTU)')
            ax.set_title(f'By {group_name.replace("_", " ").title()}')
            ax.set_xticks(x)
            ax.set_xticklabels(df[label_col].astype(str), rotation=45, ha='right')
            ax.legend()
        
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        return fig
    
    def save_figure(self, fig: plt.Figure, filename: str, 
                    output_dir: str = "outputs/figures/") -> None:
        """Save figure to file."""
        from pathlib import Path
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        filepath = Path(output_dir) / filename
        fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Saved figure: {filepath}")
        plt.close(fig)


def create_workflow_diagram(output_path: str = "outputs/figures/workflow.png") -> None:
    """
    Create workflow diagram for the paper.
    
    Shows: split → nested CV → policy → uncertainty
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Boxes
    boxes = [
        (1, 6, 2.5, 1.5, 'RECS 2020\nMicrodata'),
        (4, 6, 2.5, 1.5, 'Technology\nGrouping'),
        (7.5, 6, 2.5, 1.5, 'Feature\nEngineering'),
        (11, 6, 2.5, 1.5, 'Nested CV\n(5×3)'),
        (1, 3, 2.5, 1.5, 'Physics\nBaselines'),
        (4, 3, 2.5, 1.5, 'LightGBM\nEBM'),
        (7.5, 3, 2.5, 1.5, 'Policy\nTargeting'),
        (11, 3, 2.5, 1.5, 'Uncertainty\nJackknife'),
        (5.75, 0.5, 2.5, 1.5, 'Results &\nDiagnostics')
    ]
    
    for x, y, w, h, text in boxes:
        rect = plt.Rectangle((x, y), w, h, fill=True, 
                             facecolor='lightblue', edgecolor='navy', lw=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrows
    arrows = [
        (3.5, 6.75, 0.4, 0),
        (6.5, 6.75, 0.9, 0),
        (10, 6.75, 0.9, 0),
        (2.25, 6, 0, -0.4),
        (5.25, 6, 0, -0.4),
        (8.75, 6, 0, -0.4),
        (12.25, 6, 0, -0.4),
        (3.5, 3.75, 0.4, 0),
        (6.5, 3.75, 0.9, 0),
        (10, 3.75, 0.9, 0),
        (7, 3, 0, -0.9),
    ]
    
    for x, y, dx, dy in arrows:
        ax.annotate('', xy=(x+dx, y+dy), xytext=(x, y),
                   arrowprops=dict(arrowstyle='->', color='navy', lw=2))
    
    ax.set_title('Heating Demand Modeling Framework Workflow', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    from pathlib import Path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved workflow diagram: {output_path}")
