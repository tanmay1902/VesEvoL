import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mystyle
import utils

mystyle.apply_custom_style()

DATA_ROOT = r"F:\Budding Dynamics Data\CSV data for paper"
conc_order = ["1uM", "1.5uM", "2uM", "2.5uM"]
lipid_order = ["POPC", "POPC:Chol"]
metrics = ["MSS", "MLS", "B_low"]

def build_2d_heatmap_grid(data_tree, metric, use_normalized=False):
    """
    Generates a 2x4 layout of 2D density histograms mapping out the most 
    statistically populated membrane curvature evolution pathways.
    """
    fig, axes = plt.subplots(2, 4, figsize=(14.0, 7.0), sharex='col', sharey='row')
    time_col = 'time_norm' if use_normalized else 'time_min'
    
    for r_idx, lipid in enumerate(lipid_order):
        for c_idx, conc in enumerate(conc_order):
            ax = axes[r_idx, c_idx]
            files = data_tree.get(lipid, {}).get(conc, [])
            
            if not files:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center', color='gray', fontsize=8)
                continue
                
            trajectories = utils.load_vesicle_trajectories(files)
            if not trajectories:
                continue
                
            # Pool all individual coordinate frames for this condition panel
            df_pooled = pd.concat(trajectories, ignore_index=True).dropna(subset=[time_col, metric])
            if df_pooled.empty:
                continue
                
            x_vals = df_pooled[time_col].values
            y_vals = df_pooled[metric].values
            
            # Setup dynamic ranges based on absolute metric boundaries
            x_bins = np.linspace(x_vals.min(), x_vals.max(), 35)
            y_bins = np.linspace(y_vals.min(), y_vals.max(), 35)
            
            # Compute the two-dimensional matrix distribution density
            counts, xedges, yedges = np.histogram2d(x_vals, y_vals, bins=[x_bins, y_bins], density=True)
            
            # Draw the 2D color mesh grid plane
            mesh = ax.pcolormesh(xedges, yedges, counts.T, cmap="Blues", shading="auto", zorder=2)
            
            # Subplot structural framework polishing
            if r_idx == 0:
                ax.set_title(f"Conc: {conc}")
            if c_idx == 0:
                ax.set_ylabel(f"{lipid}\n${metric}\ (\mu m^{{-1}})$")
            if r_idx == 1:
                ax.set_xlabel("Normalized Time ($t/t_{max}$)" if use_normalized else "Time (min)", fontsize=8)
                
            # Add a subtle interior mini-colorbar to show intensity scale
            cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params()
            cbar.set_label("Relative Density")
            
    fig.tight_layout()
    return fig

def main():
    data_tree = utils.parse_dataset_tree(DATA_ROOT)
    os.makedirs("output_analysis_plots/final", exist_ok=True)
    
    for metric in metrics:
        print(f"Processing 2D Trajectory Heatmaps for: {metric}")
        fig = build_2d_heatmap_grid(data_tree, metric, use_normalized=False)
        fig.savefig(f"output_analysis_plots/final/Grid_2DHeatmap_RealTime_{metric}.png", dpi=600)
        fig.savefig(f"output_analysis_plots/final/Grid_2DHeatmap_RealTime_{metric}.pdf", dpi=600)
        plt.close(fig)
        
        fig = build_2d_heatmap_grid(data_tree, metric, use_normalized=True)
        fig.savefig(f"output_analysis_plots/final/Grid_2DHeatmap_NormTime_{metric}.pdf", dpi=600)
        fig.savefig(f"output_analysis_plots/final/Grid_2DHeatmap_NormTime_{metric}.png", dpi=600)
        plt.close(fig)

if __name__ == "__main__":
    main()