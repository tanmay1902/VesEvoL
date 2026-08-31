import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mystyle


# ============================================================
# Apply publication style
# ============================================================

mystyle.apply_style()


# ============================================================
# DATA
# ============================================================

# Load all data from the single CSV file.
# Keep the CSV in the same directory as this script.
DATA_FILE = "pipeline_validation_data.csv"

data = pd.read_csv(DATA_FILE)


def get_data(metric, method=None):
    """Return noise, value, and error arrays for a metric."""
    df = data[data["metric"] == metric].copy()

    if method is not None:
        df = df[df["method"] == method]

    df = df.sort_values("noise_sigma")

    # Missing values are represented as NaN in the CSV.
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["error"] = pd.to_numeric(df["error"], errors="coerce")

    return (
        df["noise_sigma"].to_numpy(),
        df["value"].to_numpy(),
        df["error"].to_numpy(),
    )


def get_method_data(metric):
    """Return {method: (noise, value)} for a multi-method metric."""
    result = {}

    df = data[data["metric"] == metric].copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    for method in df["method"].dropna().unique():
        sub = df[df["method"] == method].sort_values("noise_sigma")

        result[method] = (
            sub["noise_sigma"].to_numpy(),
            sub["value"].to_numpy(),
        )

    return result


# ------------------------------------------------------------
# IoU: model comparison
# ------------------------------------------------------------

iou_data = get_method_data("IoU")


# ------------------------------------------------------------
# Downstream validation of selected StarDist pipeline
# ------------------------------------------------------------

noise_precision, daughter_precision, daughter_precision_err = get_data(
    "Daughter precision",
    method="stardist",
)

_, mother_area_error, mother_area_error_err = get_data(
    "Mother area error",
    method="stardist",
)

_, mother_area_agreement, _ = get_data(
    "Mother area agreement",
    method="stardist",
)


# ------------------------------------------------------------
# Supplementary data
# ------------------------------------------------------------

# Mother centroid error
noise_centroid, mother_centroid_error, mother_centroid_error_err = get_data(
    "Mother centroid error",
    method="stardist",
)

# The original CSV contains missing error values for the final two
# centroid-error measurements. Matplotlib cannot plot a partially
# missing yerr array, so use error bars only where an error is available.
centroid_error_mask = np.isfinite(mother_centroid_error_err)


# Hausdorff discrepancy
hausdorff_data = get_method_data("Hausdorff discrepancy")


# Vesicle quantification deviation
vesicle_quantification_data = get_method_data(
    "Vesicle quantification deviation"
)


# ============================================================
# COLORS
# ============================================================

# StarDist is highlighted consistently as the selected model.

stardist_color = "#CC79A7"

# Colour used for the mother-area agreement curve
mother_color = "#0072B2"

method_colors = {
    "watershed":      "#56B4E9",
    "active_contour": "#E69F00",
    "cellpose":       "#009E73",
    "stardist":       stardist_color,
}

method_markers = {
    "watershed": "o",
    "active_contour": "x",
    "cellpose": "s",
    "stardist": "P",
}

method_linestyles = {
    "watershed": "-",
    "active_contour": "--",
    "cellpose": ":",
    "stardist": "--",
}


# ============================================================
# MAIN FIGURE
#
# (a) IoU comparison
# (b) Daughter precision + Mother area agreement
# ============================================================

fig, axes = plt.subplots(
    1, 2,
    figsize=(4.5, 2.2)
)

ax1, ax2 = axes


# ============================================================
# (a) IoU MODEL COMPARISON
# ============================================================

for method, (noise, values) in iou_data.items():

    if method == "stardist":
        linewidth = 2.0
        markersize = 5.5
        alpha = 1.0
        zorder = 5
    else:
        linewidth = 1.2
        markersize = 4.0
        alpha = 0.65
        zorder = 2

    ax1.plot(
        noise,
        values,

        marker=method_markers[method],
        linestyle=method_linestyles[method],

        linewidth=linewidth,
        markersize=markersize,

        color=method_colors[method],
        alpha=alpha,
        zorder=zorder,

        label=method.replace("_", " "),
    )


mystyle.format_axis(
    ax1,
    x_label=r"Noise $\sigma$",
    y_label="Intersection over Union"
)

ax1.set_xlim(-0.05, 1.05)
ax1.set_ylim(-0.02, 0.58)

ax1.set_xticks([0.0, 0.5, 1.0])
ax1.set_yticks(np.arange(0, 0.51, 0.1))

ax1.legend(
    frameon=False,
    fontsize=6.5,
    loc="upper right",
    handlelength=1.8,
)

mystyle.panel_label(ax1, "a", x_offset=0.5, y_offset=1.1)


# ============================================================
# (b) DOWNSTREAM PERFORMANCE
# ============================================================

# ------------------------------------------------------------
# Daughter precision
# ------------------------------------------------------------

ax2.errorbar(
    noise_precision,
    daughter_precision,
    yerr=daughter_precision_err,

    fmt='o-',

    color=stardist_color,
    markerfacecolor=stardist_color,
    markeredgecolor=stardist_color,

    markeredgewidth=1.0,
    linewidth=1.7,
    markersize=4.8,

    capsize=2.5,
    elinewidth=1.1,

    label="Daughter precision",
)


# ------------------------------------------------------------
# Mother area agreement
# ------------------------------------------------------------

ax2.plot(
    noise_precision,
    mother_area_agreement,

    's--',

    color=mother_color,
    markerfacecolor=mother_color,
    markeredgecolor=mother_color,

    linewidth=1.6,
    markersize=4.5,

    label="Mother area agreement",
)


mystyle.format_axis(
    ax2,
    x_label=r"Noise $\sigma$",
    y_label="Performance"
)

ax2.set_xlim(-0.02, 0.52)
ax2.set_ylim(0, 1.05)

ax2.set_xticks(np.arange(0, 0.51, 0.1))
ax2.set_yticks(np.arange(0, 1.01, 0.2))

ax2.legend(
    frameon=False,
    fontsize=6.5,
    loc="upper right",
    handlelength=1.8,
)

mystyle.panel_label(ax2, "b", x_offset=0.5, y_offset=1.1)


# ============================================================
# MAIN FIGURE LAYOUT
# ============================================================

fig.subplots_adjust(
    left=0.09,
    right=0.98,
    bottom=0.22,
    top=0.96,
    wspace=0.38
)

mystyle.save_pdf_png(
    fig,
    "Figure 4"
)

plt.show()


# ============================================================
# SUPPLEMENTARY FIGURE
#
# (a) Mother centroid error
# (b) Hausdorff discrepancy
# (c) Vesicle quantification deviation
# ============================================================

fig, axes = plt.subplots(
    1, 3,
    figsize=(7.5, 2.3)
)

ax1, ax2, ax3 = axes


# ============================================================
# (a) MOTHER CENTROID ERROR
# ============================================================

# Plot the full centroid-error curve.
ax1.plot(
    noise_centroid,
    mother_centroid_error,
    'o-',
    color=stardist_color,
    markerfacecolor=stardist_color,
    markeredgecolor=stardist_color,
    linewidth=1.6,
    markersize=4.5,
)

# Add error bars only where the CSV actually contains an error value.
if np.any(centroid_error_mask):
    ax1.errorbar(
        noise_centroid[centroid_error_mask],
        mother_centroid_error[centroid_error_mask],
        yerr=mother_centroid_error_err[centroid_error_mask],
        fmt='none',
        ecolor=stardist_color,
        capsize=2.5,
        elinewidth=1.1,
    )


mystyle.format_axis(
    ax1,
    x_label=r"Noise $\sigma$",
    y_label="Mother centroid error"
)

ax1.set_xlim(-0.02, 0.52)
ax1.set_ylim(0, 105)

ax1.set_xticks(np.arange(0, 0.51, 0.1))
ax1.set_yticks(np.arange(0, 101, 20))

mystyle.panel_label(ax1, "a")


# ============================================================
# (b) HAUSDORFF DISCREPANCY
# ============================================================

for method, (noise, values) in hausdorff_data.items():

    ax2.plot(
        noise,
        values,

        marker=method_markers[method],
        linestyle=method_linestyles[method],

        linewidth=(
            2.0 if method == "stardist"
            else 1.5
        ),

        markersize=(
            5.2 if method == "stardist"
            else 4.5
        ),

        color=method_colors[method],

        alpha=(
            1.0 if method == "stardist"
            else 0.75
        ),

        label=method.replace("_", " "),
    )


mystyle.format_axis(
    ax2,
    x_label=r"Noise $\sigma$",
    y_label=r"Hausdorff discrepancy, $H_d$ (pixels)"
)

ax2.set_xlim(-0.05, 1.05)
ax2.set_ylim(5, 42)

ax2.set_xticks([0, 0.5, 1.0])

ax2.legend(
    loc="lower right",
    frameon=False,
    fontsize=6.5,
    handlelength=1.8,
)

mystyle.panel_label(ax2, "b")


# ============================================================
# (c) VESICLE QUANTIFICATION DEVIATION
# ============================================================

for method, (noise, values) in vesicle_quantification_data.items():

    ax3.plot(
        noise,
        values,

        marker=method_markers[method],
        linestyle=method_linestyles[method],

        linewidth=(
            2.0 if method == "stardist"
            else 1.5
        ),

        markersize=(
            5.2 if method == "stardist"
            else 4.5
        ),

        color=method_colors[method],

        alpha=(
            1.0 if method == "stardist"
            else 0.75
        ),

        label=method.replace("_", " "),
    )


mystyle.format_axis(
    ax3,
    x_label=r"Noise $\sigma$",
    y_label=r"Vesicle quantification deviation, $\Delta V$"
)

ax3.set_xlim(-0.05, 1.05)
ax3.set_ylim(1.0, 3.6)

ax3.set_xticks([0, 0.5, 1.0])

mystyle.panel_label(ax3, "c")


# ============================================================
# SUPPLEMENTARY FIGURE LAYOUT
# ============================================================

fig.subplots_adjust(
    left=0.08,
    right=0.99,
    bottom=0.22,
    top=0.96,
    wspace=0.40
)

mystyle.save_pdf_png(
    fig,
    "pipeline_validation_supplementary"
)

plt.show()
