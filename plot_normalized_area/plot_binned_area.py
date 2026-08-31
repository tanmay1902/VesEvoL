import os
import glob
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import internal plotting toolbox submodules
from mystyle import apply_style, square_axes, panel_label, format_axis, save_pdf_png
from utils import load_csv, normalize_vesicle_data, bin_vesicle_data, aggregate_condition

# ==============================================================================
# CONFIGURATION: Set your root data path here
# ==============================================================================
DATA_ROOT = r"F:\Budding Dynamics Data\CSV data for paper"


def collect_and_process_data_auto(root_dir: str) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Automatically crawls the directory tree structure to load and parse 
    'global_area_vs_time_raw.csv' files for every vesicle replicate.
    """
    # Map the folder name naming variations to match your target keys
    conditions = ["POPC", "POPC-Chol"]
    
    # Matching folder names exactly as they are named on disk (e.g., '1uM' vs label '1 uM')
    conc_mapping = {
        "1uM": "1 uM",
        "1.5uM": "1.5 uM",
        "2uM": "2 uM",
        "2.5uM": "2.5 uM"
    }
    
    processed_dataset: Dict[str, Dict[str, pd.DataFrame]] = {
        cond: {clean_conc: pd.DataFrame() for clean_conc in conc_mapping.values()} 
        for cond in conditions
    }
    
    print(f"Scanning data root directory: {root_dir}")
    
    for cond in conditions:
        # Handle directory naming swap for POPC:Chol if folder is named with a colon or underscore
        # If your folder is named "POPC_Chol" instead of "POPC:Chol", update this string lookup.
        cond_folder = cond.replace(":", "_") if not os.path.exists(os.path.join(root_dir, cond)) else cond
        
        for folder_conc, clean_conc in conc_mapping.items():
            # Build search string match pattern: ROOT/Lipid/Concentration/V*/global_area_vs_time_raw.csv
            search_pattern = os.path.join(root_dir, cond_folder, folder_conc, "V*", "global_area_vs_time_raw.csv")
            matching_files = glob.glob(search_pattern)
            
            vesicle_list = []
            print(f"Processing [ {cond} | {clean_conc} ] - Found {len(matching_files)} vesicles.")
            
            for f_path in matching_files:
                try:
                    # Extract vesicle ID name out from the absolute folder string structure
                    vesicle_id = os.path.basename(os.path.dirname(f_path)) 

                    raw_df = load_csv(f_path)
                    norm_df = normalize_vesicle_data(raw_df)
                    binned_df = bin_vesicle_data(norm_df)
                    vesicle_list.append(binned_df)
                    
                    print(f"  -> Successfully compiled replica: {vesicle_id}")
                except Exception as err:
                    print(f"  [ERROR] Failed to parse target spreadsheet: {f_path}. Error: {err}")
            
            if vesicle_list:
                processed_dataset[cond][clean_conc] = aggregate_condition(vesicle_list)
            else:
                print(f"  [NOTICE] No data metrics extracted for group: {cond} at {clean_conc}")
                
    return processed_dataset


def generate_figure_1(dataset: Dict[str, Dict[str, pd.DataFrame]]) -> plt.Figure:
    """Creates a 2x4 grid comparing conditions across concentration states."""
    conditions = ["POPC", "POPC-Chol"]
    concentrations = ["1 uM", "1.5 uM", "2 uM", "2.5 uM"]
    
    fig, axes_grid = plt.subplots(2, 4, figsize=(10, 5.5), sharex=True, sharey=True)
    c_mother, c_daughter = "#1f77b4", "#ff7f0e"
    
    for row_idx, cond in enumerate(conditions):
        for col_idx, conc in enumerate(concentrations):
            ax = axes_grid[row_idx, col_idx]
            square_axes(ax)
            
            df = dataset[cond][conc]
            
            if df is not None and not df.empty:
                ax.errorbar(
                    df['time_norm'], df['mother_mean'], yerr=df['mother_std'],
                    color=c_mother, linestyle='-', label='Mother', elinewidth=0.8, capsize=1.5
                )
                ax.errorbar(
                    df['time_norm'], df['bud_mean'], yerr=df['bud_std'],
                    color=c_daughter, linestyle='-', label='Bud', elinewidth=0.8, capsize=1.5
                )
            
            is_bottom_row = (row_idx == 1)
            is_left_col = (col_idx == 0)
            
            x_lbl = "Normalized Time ($t / t_{max}$)" if is_bottom_row else ""
            y_lbl = "Normalized Area ($A / A_{0}$)" if is_left_col else ""
            format_axis(ax, x_label=x_lbl, y_label=y_lbl)
            
            if is_left_col:
                ax.set_ylabel(f"{cond}\n{y_lbl}".strip())
            if row_idx == 0:
                ax.set_title(conc)
                
    handles, labels = axes_grid[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.05))
        
    panel_label(axes_grid[0, 0], "A")
    panel_label(axes_grid[1, 0], "B")
    
    return fig


def generate_figure_2(dataset: Dict[str, Dict[str, pd.DataFrame]]) -> plt.Figure:
    """Creates a 1x2 summary layout comparing conditions across concentrations."""
    conditions = ["POPC", "POPC-Chol"]
    concentrations = ["1 uM", "1.5 uM", "2 uM", "2.5 uM"]
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    fig, axes_list = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    
    for idx, cond in enumerate(conditions):
        ax = axes_list[idx]
        square_axes(ax)
        
        for color_idx, conc in enumerate(concentrations):
            df = dataset[cond][conc]
            if df is not None and not df.empty:
                color = palette[color_idx % len(palette)]
                ax.plot(df['time_norm'], df['mother_mean'], color=color, linestyle='-', label=f'{conc} Mother')
                ax.plot(df['time_norm'], df['bud_mean'], color=color, linestyle='--', label=f'{conc} Bud')
                
        format_axis(ax, x_label="Normalized Time ($t / t_{max}$)", y_label="Normalized Area ($A / A_{0}$)" if idx == 0 else "")
        ax.set_title(cond)
        
    handles, labels = axes_list[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='center right', ncol=1, frameon=False, bbox_to_anchor=(1.22, 0.5))
        
    panel_label(axes_list[0], "A")
    panel_label(axes_list[1], "B")
    
    return fig


if __name__ == "__main__":
    apply_style()
    
    # Run the completely hands-free directory crawler parse execution routine
    study_data = collect_and_process_data_auto(DATA_ROOT)
    
    f1 = generate_figure_1(study_data)
    f2 = generate_figure_2(study_data)
    
    print("\nExporting publication-grade visualization output bundles...")
    save_pdf_png(f1, "figure_1_binned_profiles")
    save_pdf_png(f2, "figure_2_overlay_comparison")
    
    print("Processing runs completed successfully.")