import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.mixture import GaussianMixture
from scipy.stats import lognorm
import mystyle
import utils

mystyle.apply_custom_style()

DATA_ROOT = r"F:\Budding Dynamics Data\CSV data for paper"
conc_order = ["1uM", "1.5uM", "2uM", "2.5uM"]
lipid_order = ["POPC", "POPC:Chol"]
metrics = ["MSS", "MLS", "B_low"]

VESICLE_PALETTE = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1", "#ff9da7"]

def dual_gaussian(x, a1, mu1, sigma1, a2, mu2, sigma2):
    """Mathematical function representing a two-peak Gaussian mixture model."""
    g1 = (a1 / (sigma1 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu1) / sigma1) ** 2)
    g2 = (a2 / (sigma2 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu2) / sigma2) ** 2)
    return g1 + g2

def lognormal_model(x, sigma, loc, scale):
    return lognorm.pdf(x, s=sigma, loc=loc, scale=scale)

def estimate_initial_params(y_vals):
    """Generates robust initial parameter guesses by splitting data into quantiles."""
    mu_init = np.median(y_vals)
    sigma_init = np.std(y_vals) if np.std(y_vals) > 0 else 0.1
    
    low_peaks = y_vals[y_vals < mu_init]
    high_peaks = y_vals[y_vals >= mu_init]
    
    mu1_guess = np.mean(low_peaks) if len(low_peaks) > 0 else mu_init - 0.1
    mu2_guess = np.mean(high_peaks) if len(high_peaks) > 0 else mu_init + 0.1
    
    sigma1_guess = np.std(low_peaks) if len(low_peaks) > 0 and np.std(low_peaks) > 0 else sigma_init * 0.5
    sigma2_guess = np.std(high_peaks) if len(high_peaks) > 0 and np.std(high_peaks) > 0 else sigma_init * 0.5
    
    return [0.5, mu1_guess, sigma1_guess, 0.5, mu2_guess, sigma2_guess]


def build_fitted_histogram_grid(data_tree, metric, use_binned_data=False):

    mode_str = "Binned_2Min" if use_binned_data else "Unbinned"

    fig, axes = plt.subplots(
        2, 4,
        figsize=(14, 7),
        sharex="col",
        sharey="row"
    )

    gaussian_results = []
    lognormal_results = []

    os.makedirs("output_histogram_plots/GaussianFit", exist_ok=True)
    os.makedirs("output_histogram_plots/LogNormalFit", exist_ok=True)

    print(f"\n========== {metric} ({mode_str}) ==========")

    for r_idx, lipid in enumerate(lipid_order):

        for c_idx, conc in enumerate(conc_order):

            ax = axes[r_idx, c_idx]

            files = data_tree.get(lipid, {}).get(conc, [])

            if len(files) == 0:
                ax.text(
                    0.5,
                    0.5,
                    "No Data",
                    ha="center",
                    va="center"
                )
                continue

            trajectories = utils.load_vesicle_trajectories(files)

            for v_idx, df_v in enumerate(trajectories):

                color = VESICLE_PALETTE[v_idx % len(VESICLE_PALETTE)]
                vesicle = df_v["vesicle_name"].iloc[0]

                if use_binned_data:
                    _, y_vals, _ = utils.bin_individual_vesicle_2min(
                        df_v,
                        metric,
                        use_normalized=False
                    )
                else:
                    y_vals = df_v[metric].dropna().values

                y_vals = np.asarray(y_vals)

                if len(y_vals) < 10:
                    continue

                ##################################################
                # Histogram
                ##################################################

                counts, edges = np.histogram(
                    y_vals,
                    bins="auto"
                )

                counts = counts.astype(float)

                if counts.max() == 0:
                    continue

                counts /= counts.max()

                ax.stairs(
                    counts,
                    edges,
                    fill=True,
                    alpha=0.30,
                    color=color
                )

                ax.stairs(
                    counts,
                    edges,
                    fill=False,
                    lw=0.8,
                    color=color
                )

                xmin = y_vals.min()
                xmax = y_vals.max()
                pad = 0.05 * (xmax - xmin)

                x = np.linspace(
                    xmin - pad,
                    xmax + pad,
                    500
                )

                ##################################################
                # Gaussian Mixture
                ##################################################

                try:

                    gmm = GaussianMixture(
                        n_components=2,
                        covariance_type="full",
                        random_state=0
                    )

                    gmm.fit(y_vals.reshape(-1,1))

                    weights = gmm.weights_

                    means = gmm.means_.flatten()

                    sigmas = np.sqrt(
                        gmm.covariances_.flatten()
                    )

                    order = np.argsort(means)

                    weights = weights[order]
                    means = means[order]
                    sigmas = sigmas[order]

                    component1 = (
                        weights[0]
                        * np.exp(-(x-means[0])**2/(2*sigmas[0]**2))
                        /(sigmas[0]*np.sqrt(2*np.pi))
                    )

                    component2 = (
                        weights[1]
                        * np.exp(-(x-means[1])**2/(2*sigmas[1]**2))
                        /(sigmas[1]*np.sqrt(2*np.pi))
                    )

                    mixture = component1 + component2

                    component1 /= mixture.max()
                    component2 /= mixture.max()
                    mixture /= mixture.max()

                    ax.plot(
                        x,
                        mixture,
                        color=color,
                        lw=1.8
                    )

                    ##################################################
                    # Save Gaussian figure
                    ##################################################

                    gfig, gax = plt.subplots(figsize=(4,3))

                    gax.stairs(counts, edges, fill=True, alpha=.35)

                    gax.plot(
                        x,
                        mixture,
                        color="black",
                        lw=2,
                        label="Mixture"
                    )

                    gax.plot(
                        x,
                        component1,
                        "--",
                        lw=1.3,
                        label="Gaussian 1"
                    )

                    gax.plot(
                        x,
                        component2,
                        "--",
                        lw=1.3,
                        label="Gaussian 2"
                    )

                    gax.legend()

                    gax.set_title(
                        f"{vesicle}"
                    )

                    gfig.tight_layout()

                    gfig.savefig(
                        f"output_histogram_plots/GaussianFit/"
                        f"{metric}_{mode_str}_{lipid}_{conc}_{vesicle}.png",
                        dpi=300
                    )

                    plt.close(gfig)

                    gaussian_results.append({

                        "Metric": metric,
                        "Mode": mode_str,
                        "Lipid": lipid,
                        "Concentration": conc,
                        "Vesicle": vesicle,

                        "Weight1": weights[0],
                        "Mean1": means[0],
                        "Sigma1": sigmas[0],

                        "Weight2": weights[1],
                        "Mean2": means[1],
                        "Sigma2": sigmas[1],

                        "AIC": gmm.aic(y_vals.reshape(-1,1)),
                        "BIC": gmm.bic(y_vals.reshape(-1,1))

                    })

                except Exception:
                    pass

                ##################################################
                # Log-normal
                ##################################################

                positive = y_vals[y_vals > 0]

                if len(positive) > 5:

                    try:

                        shape, loc, scale = lognorm.fit(
                            positive,
                            floc=0
                        )

                        xlog = np.linspace(
                            positive.min(),
                            positive.max(),
                            500
                        )

                        logpdf = lognorm.pdf(
                            xlog,
                            shape,
                            loc=loc,
                            scale=scale
                        )

                        logpdf /= logpdf.max()

                        ax.plot(
                            xlog,
                            logpdf,
                            "--",
                            color=color,
                            lw=1.4
                        )

                        ##################################################
                        # Save Lognormal figure
                        ##################################################

                        lfig, lax = plt.subplots(figsize=(4,3))

                        lax.stairs(
                            counts,
                            edges,
                            fill=True,
                            alpha=.35
                        )

                        lax.plot(
                            xlog,
                            logpdf,
                            color="black",
                            lw=2
                        )

                        lax.set_title(
                            vesicle
                        )

                        lfig.tight_layout()

                        lfig.savefig(
                            f"output_histogram_plots/LogNormalFit/"
                            f"{metric}_{mode_str}_{lipid}_{conc}_{vesicle}.png",
                            dpi=300
                        )

                        plt.close(lfig)

                        ll = np.sum(
                            lognorm.logpdf(
                                positive,
                                shape,
                                loc=loc,
                                scale=scale
                            )
                        )

                        k = 3

                        lognormal_results.append({

                            "Metric": metric,
                            "Mode": mode_str,
                            "Lipid": lipid,
                            "Concentration": conc,
                            "Vesicle": vesicle,

                            "Shape": shape,
                            "Loc": loc,
                            "Scale": scale,

                            "AIC": 2*k - 2*ll,
                            "BIC": np.log(len(positive))*k - 2*ll

                        })

                    except Exception:
                        pass

            if r_idx == 0:
                ax.set_title(conc)

            if c_idx == 0:
                ax.set_ylabel(
                    f"{lipid}\nNormalized Count"
                )

            if r_idx == 1:
                ax.set_xlabel(
                    rf"${metric}\ (\mu m^{{-1}})$"
                )

    fig.tight_layout()

    pd.DataFrame(gaussian_results).to_csv(
        f"output_histogram_plots/GaussianMixture_{metric}_{mode_str}.csv",
        index=False
    )

    pd.DataFrame(lognormal_results).to_csv(
        f"output_histogram_plots/LogNormal_{metric}_{mode_str}.csv",
        index=False
    )

    return fig


def main():
    data_tree = utils.parse_dataset_tree(DATA_ROOT)
    if not data_tree:
        print("[ERROR] Mapping routine returned empty matrix.")
        sys.exit(1)
        
    os.makedirs("output_histogram_plots/final", exist_ok=True)
    
    for metric in metrics:
        # 1. Processing Raw Unbinned Frame Profiles
        fig = build_fitted_histogram_grid(data_tree, metric, use_binned_data=False)
        fig.savefig(f"output_histogram_plots/final/Grid_FittedHist_Unbinned_{metric}.png", dpi=600)
        fig.savefig(f"output_histogram_plots/final/Grid_FittedHist_Unbinned_{metric}.svg", format='svg')
        plt.close(fig)
        
        # 2. Processing 2-Minute Binned Data points
        fig = build_fitted_histogram_grid(data_tree, metric, use_binned_data=True)
        fig.savefig(f"output_histogram_plots/final/Grid_FittedHist_Binned_2Min_{metric}.png", dpi=600)
        fig.savefig(f"output_histogram_plots/final/Grid_FittedHist_Binned_2Min_{metric}.svg", format='svg')
        plt.close(fig)

    print("\nProcessing complete. Dual-peak parameters written successfully to console.")

if __name__ == "__main__":
    main()