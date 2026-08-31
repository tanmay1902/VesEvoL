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

# Distinct palette to trace each individual vesicle uniquely inside its subplot panel
VESICLE_PALETTE = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1", "#ff9da7"]

def build_matrix_grid_figure(data_tree, metric, is_binned=False, use_normalized=False):
    fig, axes = plt.subplots(2, 4, figsize=(13.0, 6.5), sharex='col', sharey='row')
    
    for r_idx, lipid in enumerate(lipid_order):
        for c_idx, conc in enumerate(conc_order):
            ax = axes[r_idx, c_idx]
            files = data_tree.get(lipid, {}).get(conc, [])
            
            if not files:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center', color='gray', fontsize=8)
                continue
                
            trajectories = utils.load_vesicle_trajectories(files)
            
            for v_idx, df_v in enumerate(trajectories):
                v_color = VESICLE_PALETTE[v_idx % len(VESICLE_PALETTE)]
                v_label = df_v['vesicle_name'].iloc[0]
                
                if is_binned:
                    # -------------------------------------------------------------
                    # INDIVIDUAL VESICLES - BINNED IN 2-MINUTE BLOCKS (DISCRETE DOTS)
                    # -------------------------------------------------------------
                    t_bin, y_bin, dy_bin = utils.bin_individual_vesicle_2min(
                        df_v, metric, use_normalized=use_normalized
                    )
                    
                    if len(t_bin) > 0:
                        ax.errorbar(t_bin, y_bin, yerr=dy_bin, fmt='o', ms=3.5, capsize=1.5, 
                                    elinewidth=0.8, color=v_color, label=v_label, alpha=0.9)
                else:
                    # -------------------------------------------------------------
                    # INDIVIDUAL VESICLES - UNBINNED (RAW CONTINUOUS LINES)
                    # -------------------------------------------------------------
                    time_col = 'time_norm' if use_normalized else 'time_min'
                    t_arr = df_v[time_col].values
                    y_arr = df_v[metric].values
                    dy_arr = df_v[f"d{metric}"].values if f"d{metric}" in df_v.columns else np.zeros_like(y_arr)
                    
                    ax.plot(t_arr, y_arr, color=v_color, lw=1.0, alpha=0.85, label=v_label)
                    ax.errorbar(t_arr, y_arr, yerr=dy_arr, fmt='none', ecolor=v_color, elinewidth=0.4, alpha=0.2)
            
            # Show the legend to track individual vesicles (V1, V2, V3) in each condition panel
            ax.legend(frameon=True, fontsize=6, loc='upper right', handlelength=1.2, borderpad=0.2)
            
            # Framework labeling parameters
            if r_idx == 0:
                ax.set_title(f"{conc}")
            if c_idx == 0:
                if metric == "MSS":
                    metric_formatted = r"M_{SS}"
                elif metric == "MLS":
                    metric_formatted = r"M_{LS}"
                elif metric == "B_low":
                    metric_formatted = r"B_{Low}"
                ax.set_ylabel(f"{lipid}\n${metric_formatted}\ (\mu m^{{-1}})$")
            if r_idx == 1:
                ax.set_xlabel("Normalized Time ($t/t_{max}$)" if use_normalized else "Time (min)")
            roman = ["(I)", "(II)", "(III)", "(IV)"]
            ax.set_title(roman[c_idx], loc="left")
                
    fig.tight_layout()
    return fig

def main():
    data_tree = utils.parse_dataset_tree(DATA_ROOT)
    if not data_tree:
        print("[ERROR] Mapping routine returned empty directory structure.")
        sys.exit(1)
        
    os.makedirs("output_matrix_plots/final", exist_ok=True)
    
    for metric in metrics:
        print(f"\nProcessing grid matrix visualizations for: {metric}")
        
        # 1. Individual Vesicles: Unbinned (Raw Lines) - Real Time Scale
        fig = build_matrix_grid_figure(data_tree, metric, is_binned=False, use_normalized=False)
        fig.savefig(f"output_matrix_plots/final/Grid_Unbinned_RealTime_{metric}.png", dpi=600)
        fig.savefig(f"output_matrix_plots/final/Grid_Unbinned_RealTime_{metric}.pdf", dpi=600)
        fig.savefig(f"output_matrix_plots/final/Grid_Unbinned_RealTime_{metric}.svg", format='svg')
        plt.close(fig)
        
        # 2. Individual Vesicles: Unbinned (Raw Lines) - Normalized Time Scale
        fig = build_matrix_grid_figure(data_tree, metric, is_binned=False, use_normalized=True)
        fig.savefig(f"output_matrix_plots/final/Grid_Unbinned_NormTime_{metric}.png", dpi=600)
        fig.savefig(f"output_matrix_plots/final/Grid_Unbinned_RealTime_{metric}.pdf", dpi=600)
        fig.savefig(f"output_matrix_plots/final/Grid_Unbinned_NormTime_{metric}.svg", format='svg')
        plt.close(fig)
        
        # 4. Individual Vesicles: 2-Minute Binned (Discrete Dots) - Normalized Time Scale
        fig = build_matrix_grid_figure(data_tree, metric, is_binned=True, use_normalized=True)
        fig.savefig(f"output_matrix_plots/final/Grid_Binned_NormTime_{metric}.png", dpi=600)
        fig.savefig(f"output_matrix_plots/final/Grid_Binned_NormTime_{metric}.pdf", dpi=600)
        fig.savefig(f"output_matrix_plots/final/Grid_Binned_NormTime_{metric}.svg", format='svg')
        plt.close(fig)

        
        
        
    print("\nProcessing complete. All independent per-vesicle grids generated successfully.")

if __name__ == "__main__":
    main()