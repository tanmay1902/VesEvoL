import os
import glob
import re
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Import internal plotting toolbox submodules
from mystyle import apply_style, square_axes, panel_label, format_axis, save_pdf_png
from utils import load_csv, bin_vesicle_data

# ==============================================================================
# CONFIGURATION: Set your root data path here
# ==============================================================================
DATA_ROOT = r"F:\Budding Dynamics Data\CSV data for paper"

# All generated outputs (parameter CSV + figures) are written into this folder
OUTPUT_DIR = "final"


# ==============================================================================
# MATHEMATICAL MODELS & FITTING
# ==============================================================================
def single_exponential(t: np.ndarray, a: float, k: float, c: float) -> np.ndarray:
    """Standard single-exponential decay model equation."""
    return a * np.exp(-k * t) + c


def fit_single_exponential(t: np.ndarray, y: np.ndarray) -> Tuple[Any, Any]:
    """
    Fits a single-exponential decay model to the given data, using the same
    physically-motivated bounds/rate-ceiling philosophy as the previous
    bi-exponential fitter (rate capped at 10 s^-1, plateau pinned near the
    trajectory's final value).

    Returns: (fitted_y, k) or (None, None) if fitting fails.
    """
    if len(t) < 3:
        return None, None

    final_y = y[-1]

    # Initial guess: amplitude ~ (y0 - final), moderate decay rate, plateau at final_y
    p0 = [max(y[0] - final_y, 0.1), 1.0, final_y]

    bounds = (
        [-2.0, 0.001, max(0.0, final_y - 0.10)],
        [ 2.0, 10.00,  min(1.1, final_y + 0.10)]
    )

    try:
        popt, _ = curve_fit(single_exponential, t, y, p0=p0, bounds=bounds, maxfev=10000)
        fitted_y = single_exponential(t, *popt)
        k = popt[1]
        return fitted_y, k
    except Exception:
        return None, None


# ==============================================================================
# DATA PIPELINE (WITH EXACT ARCHITECTURE COMPATIBILITY)
# ==============================================================================
def strictly_self_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes trajectories using the 95th percentile of the area as the true
    pre-budding baseline reference (A_0). This prevents initial growth transients
    from causing fitting artifacts.
    """
    df_norm = df.copy()

    # 1. Chronological sorting
    df_norm = df_norm.sort_values("time_min").reset_index(drop=True)

    # 2. Filter out extreme tracking dropouts (near-zero artifacts)
    median_area = df_norm["mother_area"].median()
    df_filtered = df_norm[df_norm["mother_area"] > 0.2 * median_area].copy()
    df_filtered = df_norm[df_norm["bud_area"] < np.percentile(df_norm["bud_area"], 75)].copy()

    if df_filtered.empty:
        return df_norm

    # 3. Define A_0 as the 95th percentile of the clean area trajectory
    # This captures the true maximum resting surface area before/during early dynamics
    v_a0 = np.percentile(df_filtered["mother_area"], 95)
    v_ab = max(df_filtered["bud_area"])
    # 4. Remove any remaining non-physical tracking spikes above baseline
    df_filtered = df_filtered[df_filtered["mother_area"] <= 1.25 * v_a0]

    # 5. Apply normalization using the stable reference state
    df_filtered["mother_norm"] = df_filtered["mother_area"] / v_a0
    df_filtered["bud_norm"] = df_filtered["bud_area"] / v_ab

    # Scale timeline fractionally from 0 to 1
    max_t = df_filtered["time_min"].max()
    df_filtered["time_norm"] = df_filtered["time_min"] / max_t if max_t > 0 else 0.0

    return df_filtered.reset_index(drop=True)


def collect_and_process_data_auto(root_dir: str) -> Tuple[Dict[str, Dict[str, List[Dict[str, Any]]]], pd.DataFrame]:
    """
    Crawls directory structures, applies strict self-normalization to each
    individual replicate trajectory, and returns binned datasets and parameters.
    """
    conditions = ["POPC", "POPC-Chol"]
    conc_mapping = {"1uM": "1 uM", "1.5uM": "1.5 uM", "2uM": "2 uM", "2.5uM": "2.5 uM"}

    processed_dataset = {cond: {clean_conc: [] for clean_conc in conc_mapping.values()} for cond in conditions}
    all_individual_parameters = []

    print(f"Scanning data root directory: {root_dir}")

    for cond in conditions:
        cond_folder = cond.replace(":", "_") if not os.path.exists(os.path.join(root_dir, cond)) else cond

        for folder_conc, clean_conc in conc_mapping.items():
            search_pattern = os.path.join(root_dir, cond_folder, folder_conc, "V*", "global_area_vs_time_raw.csv")
            matching_files = glob.glob(search_pattern)

            print(f"Processing [ {cond} | {clean_conc} ] - Found {len(matching_files)} vesicles.")

            for f_path in matching_files:
                try:
                    vesicle_id = os.path.basename(os.path.dirname(f_path))
                    raw_df = load_csv(f_path)

                    if raw_df.empty:
                        continue

                    # 1. Apply strict self-normalization yielding the expected columns
                    norm_df = strictly_self_normalize(raw_df)

                    # 2. Process through your original utility binning logic safely
                    binned_df = bin_vesicle_data(norm_df)

                    # 3. Dynamically discover output column labels generated by utils.py
                    time_col = [c for c in binned_df.columns if 'time' in c.lower()][0]
                    mother_col = [c for c in binned_df.columns if 'mother' in c.lower() or 'area_mean' in c.lower()][0]
                    bud_col = [c for c in binned_df.columns if 'bud' in c.lower() or 'daughter' in c.lower() or 'area' in c.lower()][-1]

                    t_data = binned_df[time_col].values
                    m_data = binned_df[mother_col].values

                    # Fit self-normalized values with a single-exponential model
                    _, m_k = fit_single_exponential(t_data, m_data)

                    processed_dataset[cond][clean_conc].append({
                        "vesicle_id": vesicle_id,
                        "df": binned_df,
                        "time_col": time_col,
                        "mother_col": mother_col,
                        "bud_col": bud_col
                    })

                    if m_k is not None:
                        all_individual_parameters.append({
                            "Condition": cond,
                            "Concentration": clean_conc,
                            "Vesicle_ID": vesicle_id,
                            "Mother_k": m_k,
                            "Mother_tau": 1 / m_k,
                        })

                except Exception as err:
                    print(f"  [ERROR] Failed to parse vesicle file: {f_path}. Error: {err}")

    param_df = pd.DataFrame(all_individual_parameters)
    return processed_dataset, param_df


# ==============================================================================
# PLOTTING GENERATORS
# ==============================================================================
def generate_fitted_figure_1(dataset: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> plt.Figure:
    """Creates a 2x4 grid plotting individual vesicles with distinct replicate tracking colors."""
    conditions = ["POPC", "POPC-Chol"]
    concentrations = ["1 uM", "1.5 uM", "2 uM", "2.5 uM"]

    vesicle_colors = {
        1: "#1f77b4",  # Blue
        2: "#ff7f0e",  # Orange
        3: "#2ca02c"   # Green
    }
    fallback_colors = ["#d62728", "#9467bd", "#8c564b"]

    fig, axes_grid = plt.subplots(2, 4, figsize=(12, 6.5), sharex=True, sharey=True)
    legend_tracker = {}

    for row_idx, cond in enumerate(conditions):
        for col_idx, conc in enumerate(concentrations):
            ax = axes_grid[row_idx, col_idx]
            square_axes(ax)

            vesicle_data_list = dataset[cond][conc]

            for color_loop_idx, v_meta in enumerate(vesicle_data_list):
                v_id = v_meta["vesicle_id"]
                df = v_meta["df"]

                num_match = re.search(r'V(\d+)', v_id, re.IGNORECASE)
                v_num = int(num_match.group(1)) if num_match else (color_loop_idx + 1)

                plot_color = vesicle_colors.get(v_num, fallback_colors[color_loop_idx % len(fallback_colors)])

                t_arr = df[v_meta["time_col"]].values
                m_data = df[v_meta["mother_col"]].values
                b_data = df[v_meta["bud_col"]].values

                # Fit the binned self-normalized values (single-exponential)
                m_fit, _ = fit_single_exponential(t_arr, m_data)
                b_fit, _ = fit_single_exponential(t_arr, b_data)

                m_key = f"V{v_num} Mother"
                b_key = f"V{v_num} Bud"

                # Plot data scatters to show true anchored y(0) = 1.0 positions
                scat_m = ax.scatter(t_arr, m_data, color=plot_color, alpha=0.5, s=10, marker="o", edgecolors='none')
                scat_b = ax.scatter(t_arr, b_data, color=plot_color, alpha=0.5, s=10, marker='d', edgecolors='none')
                if m_key not in legend_tracker:
                        legend_tracker[m_key] = scat_m
                if b_key not in legend_tracker:
                        legend_tracker[b_key] = scat_b

                '''if m_fit is not None:
                    line_m, = ax.plot(t_arr, m_fit, color=plot_color, linestyle='-', lw=1.8)
                    if m_key not in legend_tracker:
                        legend_tracker[m_key] = line_m

                if b_fit is not None:
                    line_b, = ax.plot(t_arr, b_fit, color=plot_color, linestyle='--', lw=1.8)
                    if b_key not in legend_tracker:
                        legend_tracker[b_key] = line_b'''

            is_bottom_row = (row_idx == 1)
            is_left_col = (col_idx == 0)

            x_lbl = "Normalized Time ($t / t_{max}$)" if is_bottom_row else ""
            y_lbl = "Normalized Area ($A / A_{0}$)" if is_left_col else ""
            format_axis(ax, x_label=x_lbl, y_label=y_lbl)

            if is_left_col:
                ax.set_ylabel(f"{cond}\n{y_lbl}".strip())
            # Roman numeral on EVERY subplot
            roman = ["(I)", "(II)", "(III)", "(IV)"]
            ax.set_title(roman[col_idx], loc="left")

            # Concentration only on the top row
            if row_idx == 0:
                clean_title = conc.replace(" uM", r"\ \mu\mathrm{M}")
                ax.set_title(f"${clean_title}$", loc="center")

    sorted_keys = sorted(legend_tracker.keys())
    handles = [legend_tracker[k] for k in sorted_keys]

    if handles:
        fig.legend(handles, sorted_keys, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.07), fontsize=12)

    panel_label(axes_grid[0, 0], "A")
    panel_label(axes_grid[1, 0], "B")

    return fig


def generate_parameter_plots(param_df: pd.DataFrame) -> plt.Figure:
    """Plots the mean trend line of the single-exponential decay time constant across concentrations."""
    fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
    if param_df.empty:
        return fig

    conditions = ["POPC", "POPC-Chol"]
    colors = {"POPC": "#2ca02c", "POPC-Chol": "#9467bd"}

    summary = param_df.groupby(["Condition", "Concentration"]).agg(
        m_tau_mean=("Mother_tau", "mean"),
        m_tau_sem=("Mother_tau", lambda x: x.std() / np.sqrt(len(x)) if len(x) > 1 else 0)
    ).reset_index()

    # Professional math formatting for concentration ticks
    latex_ticks = ["1 $\\mu$M", "1.5 $\\mu$M", "2 $\\mu$M", "2.5 $\\mu$M"]

    for cond in conditions:
        cond_data = summary[summary["Condition"] == cond]
        ax.errorbar(cond_data["Concentration"], cond_data["m_tau_mean"], yerr=cond_data["m_tau_sem"],
                     marker='o', ls='-', color=colors[cond], label=cond, capsize=3)

    format_axis(ax, x_label="Concentration", y_label=r"$\tau$ ( normalized time )")
    ax.set_xticklabels(latex_ticks)
    ax.legend(frameon=False)
    return fig


def generate_all_vesicles_scatter(param_df: pd.DataFrame) -> plt.Figure:
    """Creates a single comparative scatter plot split side-by-side between lipid conditions."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    if param_df.empty:
        return fig

    conditions = ["POPC", "POPC-Chol"]
    concs = ["1 uM", "1.5 uM", "2 uM", "2.5 uM"]
    latex_ticks = ["1 $\\mu$M", "1.5 $\\mu$M", "2 $\\mu$M", "2.5 $\\mu$M"]
    x_positions = {conc: i for i, conc in enumerate(concs)}

    for idx, cond in enumerate(conditions):
        ax = axes[idx]
        sub_df = param_df[param_df["Condition"] == cond].copy()

        if not sub_df.empty:
            sub_df["x_idx"] = sub_df["Concentration"].map(x_positions)
            jitter = np.random.uniform(-0.15, 0.15, size=len(sub_df))
            ax.scatter(sub_df["x_idx"] + jitter, sub_df["Mother_tau"],
                       alpha=0.7, edgecolors='k', linewidths=0.5, s=40, color="#d62728")

        ax.set_xticks(list(x_positions.values()))
        ax.set_xticklabels(latex_ticks)
        format_axis(ax, x_label="Concentration Condition", y_label=r"$\tau$ (normalized time)" if idx == 0 else "")
        ax.set_title(f"All Replicates: {cond}")
        square_axes(ax)

    return fig


if __name__ == "__main__":
    apply_style()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Execute full tree directory aggregation & single-exponential parsing loop
    study_data, parameter_dataframe = collect_and_process_data_auto(DATA_ROOT)

    if not parameter_dataframe.empty:
        param_csv_path = os.path.join(OUTPUT_DIR, "vesicle_single_exponential_parameters.csv")
        parameter_dataframe.to_csv(param_csv_path, index=False)
        print(f"\n[INFO] Complete parameter printout saved to '{param_csv_path}'")

    # Visual Generation Steps
    f1 = generate_fitted_figure_1(study_data)
    f3 = generate_parameter_plots(parameter_dataframe)
    f4 = generate_all_vesicles_scatter(parameter_dataframe)

    print("\nExporting publication-grade visualization output bundles...")
    save_pdf_png(f1, os.path.join(OUTPUT_DIR, "Figure 5"))
    save_pdf_png(f3, os.path.join(OUTPUT_DIR, "Figure 10"))
    save_pdf_png(f4, os.path.join(OUTPUT_DIR, "figure_4_all_replicates_scatter_new"))

    print("Processing runs completed successfully.")