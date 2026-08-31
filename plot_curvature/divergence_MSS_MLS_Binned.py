import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
# Reuse your existing custom styling setup safely
import mystyle
mystyle.apply_custom_style()

# Load ONLY the binned MSS and MLS Gaussian-mixture results.
# B_low and all unbinned results are intentionally excluded.
BASE_DIR = Path(".")

DATA_FILES = {
    "MSS": BASE_DIR /"output_histogram_plots"/ "GaussianMixture_MSS_Binned_2Min.csv",
    "MLS": BASE_DIR /"output_histogram_plots"/ "GaussianMixture_MLS_Binned_2Min.csv",
}


def load_binned_metric(metric):
    """Load one binned metric and convert it to the peak-wise format used for plotting."""
    path = DATA_FILES[metric]
    if not path.exists():
        raise FileNotFoundError(f"Could not find: {path}")

    data = pd.read_csv(path)

    # Convert 1uM, 1.5uM, ... -> numeric concentration for plotting.
    data["conc"] = (
        data["Concentration"]
        .astype(str)
        .str.replace("uM", "", regex=False)
        .astype(float)
    )

    rows = []
    for _, row in data.iterrows():
        rows.append({
            "metric": metric,
            "lipid": row["Lipid"],
            "conc": row["conc"],
            "vesicle": row["Vesicle"],
            "peak": "Peak 1",
            "mu": row["Mean1"],
            "sigma": row["Sigma1"],
            "weight": row["Weight1"],
        })
        rows.append({
            "metric": metric,
            "lipid": row["Lipid"],
            "conc": row["conc"],
            "vesicle": row["Vesicle"],
            "peak": "Peak 2",
            "mu": row["Mean2"],
            "sigma": row["Sigma2"],
            "weight": row["Weight2"],
        })

    return pd.DataFrame(rows)


# Only these two datasets are used.
df_mss = load_binned_metric("MSS")
df_mls = load_binned_metric("MLS")

def generate_peak_divergence_plot():
    #                 POPC              POPC:Chol
    # MSS             [0,0]             [0,1]
    # MLS             [1,0]             [1,1]
    fig, axes = plt.subplots(
        2, 2,
        figsize=(10.0, 8.0),
        sharex=True,
        sharey="row",
    )

    lipids = ["POPC", "POPC:Chol"]
    metrics = ["MSS", "MLS"]

    # Keep the same peak convention as the original MSS plot.
    peak_colors = {
        "Peak 1": "#d62728",
        "Peak 2": "#1f77b4",
    }

    metric_labels = {
        "MSS": r"|$M_{SS}|\ (\mu m^{-1})$",
        "MLS": r"|$M_{LS}|\ (\mu m^{-1})$",
    }

    metric_data = {
        "MSS": df_mss,
        "MLS": df_mls,
    }

    for row_idx, metric in enumerate(metrics):
        df_metric = metric_data[metric]

        for col_idx, lipid in enumerate(lipids):
            ax = axes[row_idx, col_idx]
            df_sub = df_metric[df_metric["lipid"] == lipid]

            for peak in ["Peak 1", "Peak 2"]:
                df_peak = df_sub[df_sub["peak"] == peak]

                # Mean ± SD across vesicles at each concentration.
                summary = (
                    df_peak.groupby("conc")["mu"]
                    .agg(["mean", "std"])
                    .reset_index()
                )
                summary["std"] = summary["std"].fillna(0.0)

                # Divergence is plotted as the absolute value of the fitted mean.
                y = np.abs(summary["mean"])

                ax.plot(
                    summary["conc"],
                    y,
                    color=peak_colors[peak],
                    ls="--",
                    lw=1.0,
                    alpha=0.7,
                )

                ax.errorbar(
                    summary["conc"],
                    y,
                    yerr=summary["std"],
                    fmt="o",
                    ms=5.5,
                    capsize=3.0,
                    elinewidth=1.2,
                    color=peak_colors[peak],
                    label=f"Population {peak}",
                    zorder=3,
                )

                # Individual vesicles.
                ax.scatter(
                    df_peak["conc"],
                    np.abs(df_peak["mu"]),
                    color=peak_colors[peak],
                    s=12.0,
                    alpha=0.3,
                    edgecolors="none",
                    zorder=1,
                )

            # Column titles only on the top row.
            if row_idx == 0:
                ax.set_title(f"Composition: {lipid}")

            ax.set_xticks([1.0, 1.5, 2.0, 2.5])
            ax.set_xlabel("Concentration ($\\mu$M)")

            # Row-specific y-axis label.
            if col_idx == 0:
                ax.set_ylabel(metric_labels[metric])

            # Legend only once.
            if row_idx == 0 and col_idx == 0:
                ax.legend(frameon=True, loc="upper right")

    fig.tight_layout()

    out_dir = Path("output_analysis_plots/final")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "Figure 9.png"
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"[SUCCESS] MSS + MLS binned divergence plot saved to: {out_path}")


if __name__ == "__main__":
    generate_peak_divergence_plot()
