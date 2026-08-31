import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from matplotlib.patches import Patch

try:
    import easygui
except ImportError as exc:
    raise ImportError(
        "This script requires the 'easygui' package, exactly like the rest "
        "of the analysis pipeline. Install it with `pip install easygui`."
    ) from exc

import mystyle

# Apply the shared publication style sheet (mystyle.mplstyle) globally.
# This replaces the old ad-hoc rcParams block so every figure in this
# script matches the rest of the pipeline (fonts, spines, ticks, colors).
mystyle.apply_style()



# Pixel -> micron conversion factor used to convert raw contact radii
# (r_contact, in pixels) into physical units (microns) for display on
# the polar contact plots.
PIXEL_TO_MICRON = 0.227  # um / pixel

# =========================================================
# TIME NORMALIZATION HELPER
# =========================================================
def _normalize_time(time_values):
    """
    Normalize time values from their original range to [0, 1].
    
    Parameters
    ----------
    time_values : array-like
        Time values in minutes (or any original unit).
    
    Returns
    -------
    normalized : ndarray
        Time normalized to [0, 1] range.
    """
    t_min = np.nanmin(time_values)
    t_max = np.nanmax(time_values)
    if t_max == t_min:
        return np.zeros_like(time_values, dtype=float)
    return (time_values - t_min) / (t_max - t_min)

# =========================================================
# TIME PARSING
# =========================================================
def parse_time(tstr):
    return datetime.strptime(tstr, "%H:%M")

# =========================================================
# DIRECTORY SELECTION (EasyGUI)
# =========================================================
def select_results_directory() -> str:
    """
    Ask the user (via EasyGUI) to select the results directory that
    directly contains the Experiment-N folders, exactly like the
    directory-selection style used elsewhere in the pipeline.

    Returns
    -------
    str
        Full path to the selected results directory.

    Raises
    ------
    SystemExit
        If the user cancels the dialog, or the chosen folder contains
        no "Experiment-*" subfolders.
    """
    base_dir = easygui.diropenbox(
        msg="Select the results directory containing Experiment-N folders",
        title="Contact Curvature Analysis",
    )

    if base_dir is None:
        raise SystemExit("[INFO] Directory selection cancelled by user.")

    exp_folders = [
        f for f in os.listdir(base_dir)
        if f.startswith("Experiment-") and os.path.isdir(os.path.join(base_dir, f))
    ]
    if not exp_folders:
        easygui.msgbox(
            f"No 'Experiment-*' folders were found in:\n{base_dir}",
            title="No experiments found",
        )
        raise SystemExit(f"[FATAL] No Experiment-* folders found in {base_dir}")

    print(f"[INFO] Selected results directory: {base_dir}")
    print(f"[INFO] Found {len(exp_folders)} experiment folder(s).")
    return base_dir


def prompt_run_parameters(exp_folders):
    """
    Ask the user (via EasyGUI) for the timing parameters needed to run
    the analysis: time-per-frame (seconds), frame stride, and the
    per-experiment initial delay (minutes).

    Parameters
    ----------
    exp_folders : list[str]
        Experiment folder names (sorted), used to build one delay field
        per experiment.

    Returns
    -------
    tuple[list[float], float, int]
        (delays, time_per_frame_sec, frame_stride)

    Raises
    ------
    SystemExit
        If the user cancels either dialog or enters invalid values.
    """
    global_fields = ["Time per frame (seconds)", "Frame stride"]
    global_defaults = ["1.0", "1"]
    global_values = easygui.multenterbox(
        msg="Enter global timing parameters",
        title="Contact Curvature Analysis",
        fields=global_fields,
        values=global_defaults,
    )
    if global_values is None:
        raise SystemExit("[INFO] Parameter entry cancelled by user.")

    try:
        tpf = float(global_values[0])
        fs = int(global_values[1])
    except ValueError as exc:
        raise SystemExit(f"[FATAL] Invalid global timing parameter: {exc}")

    delay_values = []

    print("\nEnter the initial acquisition delay for each experiment (in minutes).")
    print("Press Enter to use 0.0 min.\n")

    for exp in exp_folders:
        while True:
            value = input(f"Initial delay for {exp} (min) [0.0]: ").strip()

            if value == "":
                value = "0.0"

            try:
                delay_values.append(float(value))
                break
            except ValueError:
                print("  [ERROR] Please enter a valid number.")

    delays = delay_values
    return delays, tpf, fs


# =========================================================
# MOTHER CENTROID LOOKUP (for contact theta)
# =========================================================
def load_mother_centroid(exp_path):
    """
    Load the per-frame mother centroid from
    <Experiment-N>/mother/mother_metrics.csv.

    Parameters
    ----------
    exp_path : str
        Full path to one Experiment-N folder.

    Returns
    -------
    pandas.DataFrame or None
        Columns: frame, mother_centroid_x, mother_centroid_y.
        None if the file is missing (theta cannot be computed for this
        experiment and it is skipped gracefully by the caller).
    """
    metrics_path = os.path.join(exp_path, "mother", "mother_metrics.csv")

    if not os.path.exists(metrics_path):
        print(f"  mother_metrics.csv not found in {exp_path}/mother — "
              f"contact theta cannot be computed for this experiment.")
        return None

    metrics = pd.read_csv(metrics_path)

    required = {"frame", "centroid_x", "centroid_y"}
    missing = required - set(metrics.columns)
    if missing:
        print(f"  mother_metrics.csv in {exp_path} is missing column(s) "
              f"{sorted(missing)} — contact theta cannot be computed.")
        return None

    return metrics[["frame", "centroid_x", "centroid_y"]].rename(
        columns={"centroid_x": "mother_centroid_x", "centroid_y": "mother_centroid_y"}
    )


def v0_compute_contact_theta(daughters, mother_centroid):
    """
    Attach dx, dy, theta (radians, atan2 convention), and r_contact to
    every contact row in ``daughters``, using the mother centroid at
    the matching (raw) frame.

    r_contact is the actual distance from the mother centroid to the
    contact point itself (sqrt(dx**2 + dy**2)) — i.e. each contact's
    OWN radial position, which need not equal the mother's radius at
    that time (initial, final, or otherwise). This is what downstream
    polar/karyograph plots use for the radial coordinate, instead of a
    single fixed reference radius.

    Parameters
    ----------
    daughters : pandas.DataFrame
        Raw per-contact daughter tracking rows for one experiment. Must
        contain 'frame', 'contact_point_x', 'contact_point_y'.
    mother_centroid : pandas.DataFrame or None
        Output of load_mother_centroid(). If None, 'theta' and
        'r_contact' are filled with NaN for every row (handled
        gracefully downstream).

    Returns
    -------
    pandas.DataFrame
        Copy of ``daughters`` with added columns: mother_centroid_x,
        mother_centroid_y, dx, dy, theta, r_contact.
    """
    result = daughters.copy()

    if mother_centroid is None:
        result["mother_centroid_x"] = np.nan
        result["mother_centroid_y"] = np.nan
        result["dx"] = np.nan
        result["dy"] = np.nan
        result["theta"] = np.nan
        result["r_contact"] = np.nan
        return result

    result = result.merge(mother_centroid, on="frame", how="left")

    result["dx"] = result["contact_point_x"] - result["mother_centroid_x"]
    result["dy"] = result["contact_point_y"] - result["mother_centroid_y"]
    result["theta"] = np.arctan2(result["dy"], result["dx"])
    result["r_contact"] = np.sqrt(result["dx"]**2 + result["dy"]**2)

    n_missing = result["theta"].isna().sum()
    if n_missing > 0:
        print(f"  [WARN] {n_missing} contact row(s) have no matching mother "
              f"frame in mother_metrics.csv; theta/r_contact set to NaN for those rows.")

    return result

def compute_contact_theta(daughters, mother_centroid):
    """
    Compute contact angle and radial distance using the SAME
    mother-centered coordinate convention used during daughter tracking.

    The saved contact_point_x/y coordinates are already translated by
        tx = REF_CENTER_x - mother_centroid_x(frame)
        ty = REF_CENTER_y - mother_centroid_y(frame)

    Therefore, the correct mother-centered coordinates are obtained by
    subtracting the fixed reference center, NOT the per-frame mother centroid.
    """

    result = daughters.copy()

    if mother_centroid is None:
        result["mother_centroid_x"] = np.nan
        result["mother_centroid_y"] = np.nan
        result["dx"] = np.nan
        result["dy"] = np.nan
        result["theta"] = np.nan
        result["r_contact"] = np.nan
        return result

    # -----------------------------------------------------
    # Find the same reference center used during tracking.
    #
    # process_daughters_parallel() uses the first valid
    # mother contour centroid as REF_CENTER.
    # -----------------------------------------------------
    valid_mother = mother_centroid.dropna(
        subset=["mother_centroid_x", "mother_centroid_y"]
    ).sort_values("frame")

    if valid_mother.empty:
        print("  [WARN] No valid mother centroid found.")
        result["theta"] = np.nan
        result["r_contact"] = np.nan
        return result

    ref_x = float(valid_mother.iloc[0]["mother_centroid_x"])
    ref_y = float(valid_mother.iloc[0]["mother_centroid_y"])

    print(
        f"  [INFO] Contact geometry reference center: "
        f"({ref_x:.3f}, {ref_y:.3f}) px"
    )

    # Keep the actual mother centroid columns for reference/debugging.
    result = result.merge(
        mother_centroid,
        on="frame",
        how="left"
    )

    # -----------------------------------------------------
    # IMPORTANT:
    #
    # contact_point_x/y are already shifted into the
    # reference-centered coordinate system.
    #
    # Therefore subtract REF_CENTER, not the moving
    # mother centroid.
    # -----------------------------------------------------
    result["dx"] = (
        result["contact_point_x"] - ref_x
    )

    result["dy"] = (
        result["contact_point_y"] - ref_y
    )

    result["theta"] = np.arctan2(
        result["dy"],
        result["dx"]
    )

    result["r_contact"] = np.sqrt(
        result["dx"]**2 +
        result["dy"]**2
    )

    return result

# =========================================================
# PAIRWISE GEOMETRY
# =========================================================
def old_compute_pairwise_geometry(daughters):
    """
    Computes pairwise MSS, effective RS, and propagated errors
    """
    records = []

    for t, g in daughters.groupby("global_time_min"):
        R = g["R_S"].values
        n = len(R)

        if n < 2:
            continue

        mss_vals = []
        rs_eff_vals = []

        for i in range(n):
            for j in range(i + 1, n):
                inv_rs = 0.5 * (1 / R[i] + 1 / R[j])
                mss_vals.append(-inv_rs)
                rs_eff_vals.append(1 / inv_rs)

        records.append({
            "global_time_min": t,
            "MSS": np.mean(mss_vals),
            "dMSS": np.std(mss_vals),
            "R_S_eff": np.mean(rs_eff_vals),
            "dR_S_eff": np.std(rs_eff_vals)
        })

    return pd.DataFrame(records)

def compute_pairwise_geometry(daughters, mothers):
    """
    Computes:
    MSS = -1/2 (1/Ri + 1/Rj)  for each daughter-daughter pair
    MLS =  1/2 (1/RL - 1/Ri)  for each daughter-mother pair
    Using REAL radii (no effective radius)

    In addition to the existing time-averaged MSS/MLS/B_low records,
    this also returns two point-level dataframes (one row per
    individual contact) carrying the contact's angular position
    (theta) AND its own radial position (r_contact — the actual
    distance from the mother centroid to that contact point, not a
    shared reference radius) alongside the SAME MSS/MLS value used
    above, for polar/karyograph visualization. The MSS/MLS math itself
    is unchanged: a pairwise MSS value is simply attributed to both
    contact points that make up the pair, and the MLS value is already
    computed per individual contact.

    NOTE ON CLASS FILTERING: the underlying daughter_tracking_local.csv
    exposes contact type via a single 'contact_class' column with
    values 'MLS' (daughter touching mother) / 'MSS' (daughter touching
    daughter) rather than separate 'touching_daughter'/'touching_mother'
    boolean columns. Those masks are derived here from 'contact_class'.
    """

    records = []
    mss_point_records = []
    mls_point_records = []

    for t, g in daughters.groupby("global_time_min"):

        if t not in mothers["global_time_min"].values:
            continue

        # mother radius at this time
        RL = mothers.loc[
            mothers["global_time_min"] == t, "R_L"
        ].mean()

        dRL = mothers.loc[
            mothers["global_time_min"] == t, "dR_L"
        ].mean()

        g = g.copy()
        g["R_S"] = np.sqrt(g["area"] / np.pi)

        # uncertainty in daughter radius
        g["dR_S"] = g["R_S"].std()

        MSS_vals = []
        MLS_vals = []
        B_vals   = []

        # ================================
        # RESTRICT TO CONTACTING CELLS
        # 'touching_mother'  <=> contact_class == 'MLS'
        # 'touching_daughter' <=> contact_class == 'MSS'
        # (daughter_tracking_local.csv only stores 'contact_class';
        #  the boolean masks are derived from it here)
        # ================================
        g_dd = g[g["contact_class"] == "MSS"]
        g_md = g[g["contact_class"] == "MLS"]

        R_dd = g_dd["R_S"].values
        dR_dd = g_dd["dR_S"].values
        theta_dd = g_dd["theta"].values
        r_contact_dd = g_dd["r_contact"].values

        R_md = g_md["R_S"].values
        dR_md = g_md["dR_S"].values
        theta_md = g_md["theta"].values
        r_contact_md = g_md["r_contact"].values

        n = len(R_dd)

        # =====================================================
        # MSS: ONLY daughters touching daughters
        # =====================================================
        for i in range(n):
            for j in range(i + 1, n):

                MSS = -0.5 * (1.0 / R_dd[i] + 1.0 / R_dd[j])

                dMSS = 0.5 * np.sqrt(
                    (dR_dd[i] / R_dd[i]**2)**2 +
                    (dR_dd[j] / R_dd[j]**2)**2
                )

                MSS_vals.append(MSS)

                # Point-level records for polar/karyograph plotting:
                # both contact points that make up this pair share the
                # same MSS value, each plotted at its OWN angular
                # position (theta) and OWN radial position (r_contact
                # — the true distance of that contact point from the
                # mother centroid, not a shared fixed radius).
                for theta_point, r_point in zip(
                    (theta_dd[i], theta_dd[j]),
                    (r_contact_dd[i], r_contact_dd[j]),
                ):
                    if not np.isnan(theta_point) and not np.isnan(r_point):
                        mss_point_records.append({
                            "global_time_min": t,
                            "theta": theta_point,
                            "r_contact": r_point,
                            "MSS": MSS,
                        })

        # =====================================================
        # MLS: ONLY daughters touching mother
        # =====================================================
        for k in range(len(R_md)):

            RS = R_md[k]

            MLS = 0.5 * (1.0 / RL - 1.0 / RS)

            dMLS = 0.5 * np.sqrt(
                (dRL / RL**2)**2 +
                (dR_md[k] / RS**2)**2
            )

            MLS_vals.append(MLS)

            # B_low (also individual)
            B = -3.0 / (RS * (1.0 + RS / RL))

            den = (RS * RL + RS**2)**2

            dB = np.sqrt(
                (3.0 * RL * (RL + 2.0 * RS) * dR_md[k] / den)**2 +
                (3.0 * RS**2 * dRL / den)**2
            )

            B_vals.append(B)

            if not np.isnan(theta_md[k]) and not np.isnan(r_contact_md[k]):
                mls_point_records.append({
                    "global_time_min": t,
                    "theta": theta_md[k],
                    "r_contact": r_contact_md[k],
                    "MLS": MLS,
                    "B_low": B,
                })

        if len(MSS_vals) == 0 and len(MLS_vals) == 0:
            continue

        records.append({
            "global_time_min": t,
            "MSS": np.mean(MSS_vals) if MSS_vals else np.nan,
            "dMSS": np.std(MSS_vals) if MSS_vals else np.nan,
            "MLS": np.mean(MLS_vals) if MLS_vals else np.nan,
            "dMLS": np.std(MLS_vals) if MLS_vals else np.nan,
            "B_low": np.mean(B_vals) if B_vals else np.nan,
            "dB_low": np.std(B_vals) if B_vals else np.nan
        })

    pair_df = pd.DataFrame(records)
    mss_points_df = pd.DataFrame(mss_point_records)
    mls_points_df = pd.DataFrame(mls_point_records)

    return pair_df, mss_points_df, mls_points_df


def _Fadd_polar_scalebar(ax, r_values_um):
    """
    Remove the default numeric radial (r) tick labels (the "100, 250,
    300 ..." ring labels) and replace them with a single labeled
    reference circle acting as a scale bar (e.g. "50 um"), so the plot
    conveys scale without cluttering every gridline with a number.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Polar axes to style.
    r_values_um : array-like
        The radial values (already converted to microns) being
        plotted, used to pick a sensible round scale-bar length.
    """
    r_max = np.nanmax(r_values_um) if len(r_values_um) else 1.0
    if not np.isfinite(r_max) or r_max <= 0:
        r_max = 1.0

    # Pick a "nice" round scale-bar length (~half the max radius).
    target = r_max / 2
    magnitude = 10 ** np.floor(np.log10(target))
    nice_multiples = np.array([1, 2, 5, 10])
    scale_len = nice_multiples[np.argmin(np.abs(nice_multiples * magnitude - target))] * magnitude

    # Keep only a single reference gridline at the scale-bar radius,
    # with no numeric labels on it.
    # Keep only a single reference gridline at the scale-bar radius
    ax.set_rticks([scale_len])
    ax.set_yticklabels([])

    print(r_max)


def _add_polar_scalebar(ax, r_values_um):
    """
    Remove the default numeric radial (r) tick labels and replace them 
    with a single labeled reference circle acting as a scale bar, 
    positioned at the top right of the polar plot.
    """
    r_max = np.nanmax(r_values_um) if len(r_values_um) else 1.0
    if not np.isfinite(r_max) or r_max <= 0:
        r_max = 1.0

    # Pick a "nice" round scale-bar length (~half the max radius).
    target = r_max / 2
    magnitude = 10 ** np.floor(np.log10(target))
    nice_multiples = np.array([1, 2, 5, 10])
    scale_len = nice_multiples[np.argmin(np.abs(nice_multiples * magnitude - target))] * magnitude

    # Keep only a single reference gridline at the scale-bar radius
    ax.set_rticks([scale_len])
    ax.set_yticklabels([])

    # Place the text at 45 degrees (top-right quadrant) right along the outer rim
    #ax.text(
    ##    np.deg2rad(45), r_max * 1.05, f"Scale: {scale_len:g} \u03bcm",
    #    ha="left", va="bottom", fontsize=8, color="0.2",
    #)
# =========================================================
# POLAR MLS / MSS PLOTS
# =========================================================
def plot_polar_contact_class(points_df, value_col, class_name, base_dir,limit_radial=False, color_by="value"):
    """
    Polar scatter plot of individual contact points, each placed at its
    OWN contact radius (r_contact — the actual distance from the
    mother centroid to that contact point) and angular position theta.

    Two "flavors" are supported via color_by:
      - "value": points colored by their MSS/MLS curvature value
      - "time":  points colored by global_time_min (normalized to [0, 1])

    Call this function twice per class (once per color_by) to get both
    versions as separate figures.

    Parameters
    ----------
    points_df : pandas.DataFrame
        Point-level dataframe from compute_pairwise_geometry(), must
        contain 'theta', 'r_contact', ``value_col``, and
        'global_time_min'.
    value_col : str
        Column to use for point coloring when color_by="value"
        ('MSS' or 'MLS').
    class_name : str
        Label for the contact class, used in the title/filename.
    base_dir : str
        Directory to save the figure into.
    color_by : str
        "value" or "time" — which quantity drives the colorbar.
    """
    if points_df.empty:
        print(f"[WARN] plot_polar_contact_class: no data for class {class_name}; skipping.")
        return

    required_cols = ["theta", "r_contact", value_col]
    if color_by == "time":
        required_cols.append("global_time_min")

    data = points_df.dropna(subset=required_cols)
    if data.empty:
        print(f"[WARN] plot_polar_contact_class: no valid rows for class "
              f"{class_name} (color_by={color_by}); skipping.")
        return

    theta = data["theta"].values
    r = data["r_contact"].values*0.227

    if color_by == "value":
        color_vals = data[value_col].values
        cbar_label = rf"${value_col}\ (\mu m^{{-1}})$"
        suffix = "value"
    elif color_by == "time":
        color_vals = _normalize_time(data["global_time_min"].values)
        cbar_label = "Time (normalized)"
        suffix = "time"
    else:
        raise ValueError(f"color_by must be 'value' or 'time', got {color_by!r}")

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="polar")
    scatter = ax.scatter(
        theta, r, c=color_vals, cmap="viridis", s=25, alpha=0.85, edgecolors="none"
    )
    if limit_radial is not None: 
        ax.set_rlim(0, limit_radial)
    #_add_polar_scalebar(ax, r)
    cbar = fig.colorbar(scatter, ax=ax, pad=0.12, shrink=0.75)
    cbar.set_label(cbar_label, fontsize=15, labelpad=8)
    cbar.ax.tick_params(labelsize=13)
    
    fig.tight_layout()
    fname = f"{base_dir}/fig_polar_{class_name}_{suffix}"
    mystyle.save_pdf_png(fig, f"{fname}")
    plt.show()
    plt.close(fig)
    print(f"[INFO] Saved polar {class_name} ({suffix}) figure to {fname}.png")


# =========================================================
# KARYOGRAPH PLOTS (R vs. Theta, cartesian)
# =========================================================
def plot_karyograph_contact_class(points_df, value_col, class_name, base_dir,
                                   r_initial=None, r_final=None):
    """
    "Karyograph"-style cartesian scatter of contact position: each
    contact's OWN radial distance from the mother centroid (r_contact)
    on the y-axis vs. its angular position (theta, radians) on the
    x-axis. Two figures are produced per contact class:
      - colored by the curvature value (MSS/MLS)
      - colored by acquisition time (normalized to [0, 1])

    Optional horizontal reference lines mark the initial and final
    mother radius (R_L), for visual context of where each contact sits
    relative to the mother's size over the course of the experiment.

    Parameters
    ----------
    points_df : pandas.DataFrame
        Point-level dataframe from compute_pairwise_geometry(), must
        contain 'theta', 'r_contact', ``value_col``, 'global_time_min'.
    value_col : str
        'MSS' or 'MLS'.
    class_name : str
        Label used in title/filename.
    base_dir : str
        Directory to save figures into.
    r_initial, r_final : float, optional
        Initial/final mother radius (um), drawn as reference lines.
    """
    if points_df.empty:
        print(f"[WARN] plot_karyograph_contact_class: no data for class {class_name}; skipping.")
        return

    data = points_df.dropna(subset=["theta", "r_contact", value_col, "global_time_min"])
    if data.empty:
        print(f"[WARN] plot_karyograph_contact_class: no valid rows for class "
              f"{class_name}; skipping.")
        return

    theta = data["theta"].values
    r = data["r_contact"].values
    

    for color_by in ("value", "time"):
        if color_by == "value":
            color_vals = data[value_col].values
            cbar_label = rf"${value_col}\ (\mu m^{{-1}})$"
        else:
            color_vals = _normalize_time(data["global_time_min"].values)
            cbar_label = "Time (normalized)"

        fig, ax = plt.subplots(figsize=(7, 5))
        scatter = ax.scatter(
            theta, r, c=color_vals, cmap="viridis", s=20, alpha=0.85, edgecolors="none"
        )

        has_ref = False
        if r_initial is not None:
            ax.axhline(r_initial, color="0.4", linestyle="--", linewidth=1, label=r"Initial $R_L$")
            has_ref = True
        if r_final is not None:
            ax.axhline(r_final, color="0.7", linestyle=":", linewidth=1, label=r"Final $R_L$")
            has_ref = True

        ax.set_xlabel(r"$\theta$ (rad)")
        ax.set_ylabel(r"$R\ (\mu m)$")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if has_ref:
            ax.legend(fontsize=7, frameon=True, fancybox=True, framealpha=0.75, edgecolor="0.7")

        cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label(cbar_label)

        fig.tight_layout()
        fname = f"{base_dir}/fig_karyograph_{class_name}_{color_by}"
        mystyle.save_pdf_png(fig, f"{fname}")
        plt.close(fig)
        print(f"[INFO] Saved karyograph {class_name} ({color_by}) figure to {fname}.png")


def plot_combined_karyograph(mss_points_df, mls_points_df, base_dir):
    """
    Bonus overlay: MSS and MLS contact points on a single karyograph
    (R vs. theta), colored by time (normalized to [0, 1]) and distinguished 
    by marker shape (circles = MSS, triangles = MLS), for a direct 
    side-by-side view of where each contact class sits on the mother 
    membrane over the course of the experiment.
    """
    mss = mss_points_df.dropna(subset=["theta", "r_contact", "global_time_min"]) \
        if not mss_points_df.empty else mss_points_df
    mls = mls_points_df.dropna(subset=["theta", "r_contact", "global_time_min"]) \
        if not mls_points_df.empty else mls_points_df

    if mss.empty and mls.empty:
        print("[WARN] plot_combined_karyograph: no data available; skipping.")
        return

    # Get min/max across both datasets for consistent normalization
    all_times = []
    if not mss.empty:
        all_times.extend(mss["global_time_min"].values)
    if not mls.empty:
        all_times.extend(mls["global_time_min"].values)
    all_times = np.array(all_times)
    
    t_min = np.nanmin(all_times)
    t_max = np.nanmax(all_times)
    
    # Normalize time for both datasets using shared min/max
    mss_time_norm = (mss["global_time_min"].values - t_min) / (t_max - t_min) if not mss.empty else None
    mls_time_norm = (mls["global_time_min"].values - t_min) / (t_max - t_min) if not mls.empty else None

    fig, ax = plt.subplots(figsize=(8, 5))

    sc = None
    if not mss.empty:
        sc = ax.scatter(
            mss["theta"], mss["r_contact"], c=mss_time_norm,
            cmap="viridis", vmin=0, vmax=1, marker="o", s=20,
            alpha=0.85, edgecolors="none", label="MSS"
        )
    if not mls.empty:
        sc = ax.scatter(
            mls["theta"], mls["r_contact"], c=mls_time_norm,
            cmap="viridis", vmin=0, vmax=1, marker="^", s=20,
            alpha=0.85, edgecolors="none", label="MLS"
        )

    ax.set_xlabel(r"$\theta$ (rad)")
    ax.set_ylabel(r"$R\ (\mu m)$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=7, frameon=True, fancybox=True, framealpha=0.75, edgecolor="0.7")

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Time (normalized)")

    fig.tight_layout()
    mystyle.save_pdf_png(fig, f"{base_dir}/fig_karyograph_combined_time")
    plt.close(fig)
    print(f"[INFO] Saved combined karyograph figure to {base_dir}/fig_karyograph_combined_time.png")


# =========================================================
# MAIN ANALYSIS
# =========================================================
def plot_contact_curvature(base_dir,delays,tpf,fs):

    exp_folders = sorted(
        [f for f in os.listdir(base_dir) if f.startswith("Experiment-")],
        key=lambda x: int(x.split("-")[-1])
    )

    frame_stride = fs
    all_data = []
    daughter_records = []
    global_start_ref = None

    time_per_frame_sec = tpf
    # =====================================================
    # LOAD + TIME ALIGN
    # =====================================================
    initial_delay_min = 0
    for id,exp in enumerate(exp_folders):
        exp_path = os.path.join(base_dir, exp)

        print(f"\nProcessing {exp}")

        # ---- ASK USER FOR TIMING ----
        #initial_delay_min = float(
        #        input(f"Enter INITIAL DELAY for {exp} (minutes): ")
        #    )
        initial_delay_min = delays[id]
        mother_path = os.path.join(exp_path, "mother.csv")#"mother.csv")
        
        

        if not os.path.exists(mother_path):
            print("  mother.csv not found, skipping.")
            continue

        mother = pd.read_csv(mother_path)
        mother["true_frame"] = mother["frame"] * frame_stride
        
        # ---- GLOBAL TIME (mother) ----
        mother["global_time_min"] = (
            initial_delay_min
            + ((mother["true_frame"])
            * time_per_frame_sec) / 60.0
        )

        all_data.append(
            mother[["global_time_min", "area_um2"]]
            .rename(columns={"area_um2": "mother_area"})
        )

        # ---- MOTHER CENTROID (for contact theta) ----
        mother_centroid = load_mother_centroid(exp_path)
        
        
        # ---- DAUGHTERS (OPTIONAL) ----
        '''if os.path.exists(daughter_path):
            daughters = pd.read_csv(daughter_path)
            daughters["true_frame"] = daughters["frame"] * frame_stride

            daughters["global_time_min"] = (
                initial_delay_min
                + ((daughters["true_frame"])* time_per_frame_sec) / 60.0
            )
            bud = daughters.copy(True)
            bud = bud[bud["area"] > np.percentile(bud["area"],20)]
            bud = bud[bud["area"] < np.percentile(bud["area"],60)]
            #daughters = daughters.rename(columns={"area_um2":"area"})

            daughter_records.append(bud)
        
        else:
            print("  daughter_tracking_local.csv not found — skipping daughters")'''
        
        daughter_path1 = os.path.join(
            exp_path, "localDaughter", "daughter_tracking_local.csv"
        )

        daughter_path2 = os.path.join(
            exp_path, "innerlocalDaughter", "daughter_tracking_local.csv"
        )

        if os.path.exists(daughter_path1):
            print(daughter_path1)
            daughters = pd.read_csv(daughter_path1)

        elif os.path.exists(daughter_path2):
            print(daughter_path2)
            daughters = pd.read_csv(daughter_path2)

        else:
            daughters = None

        
        

        if daughters is not None:
            daughters["true_frame"] = daughters["frame"] * frame_stride
            daughters["global_time_min"] = (
                initial_delay_min
                + ((daughters["true_frame"])* time_per_frame_sec) / 60.0
            )

            # Attach contact theta and r_contact (angular position and
            # true radial distance relative to mother centroid at the
            # matching raw frame) before concatenating across
            # experiments, since mother centroid lookup is
            # per-experiment.
            daughters = compute_contact_theta(daughters, mother_centroid)

            daughter_records.append(daughters)
            
        else:
            bud = pd.DataFrame({"global_time_min": mother["global_time_min"], "area": 0})
            
        

    df = pd.concat(all_data, ignore_index=True)
    daughters_all = pd.concat(daughter_records, ignore_index=True)
    # =====================================================
    # DEBUG CONTACT RADIAL DISTANCES
    # =====================================================

    daughters_all["r_contact_um"] = (
        daughters_all["r_contact"] * PIXEL_TO_MICRON
    )

    print("\n========== CONTACT RADIUS DEBUG ==========")

    print(
        daughters_all[
            [
                "frame",
                "contact_class",
                "contact_point_x",
                "contact_point_y",
                "mother_centroid_x",
                "mother_centroid_y",
                "r_contact",
                "r_contact_um",
            ]
        ]
        .sort_values("r_contact_um", ascending=False)
        .head(20)
        .to_string(index=False)
    )

    r_max = daughters_all["r_contact_um"].max()
    r_95 = daughters_all["r_contact_um"].quantile(0.95)
    r_99 = daughters_all["r_contact_um"].quantile(0.99)

    print("\nMaximum contact radius:")
    print(r_max)

    print("\n95th percentile:")
    print(r_95)

    print("\n99th percentile:")
    print(r_99)

    # -----------------------------------------------------
    # POLAR PLOT RADIAL LIMIT
    # -----------------------------------------------------
    limit_radial = r_95
    print(f"\n[INFO] Polar radial limit = {limit_radial:.2f} um")
    # =====================================================
    # RADII
    # =====================================================
    df["R_L"] = np.sqrt(df["mother_area"] / np.pi)
    daughters_all["R_S"] = np.sqrt(daughters_all["area"] / np.pi)

    # =====================================================
    # PAIRWISE CURVATURES
    # =====================================================
    #pair_df = compute_pairwise_geometry(daughters_all)

    # =====================================================
    # MOTHER RADIUS STATISTICS (MEAN + ERROR)
    # =====================================================
    RL_stats = (
        df.groupby("global_time_min")["R_L"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={
            "mean": "R_L",
            "std": "dR_L"
        })
    )

    RL_stats["dR_L"] = RL_stats["dR_L"].fillna(0.0)
    
    pair_df, mss_points_df, mls_points_df = compute_pairwise_geometry(daughters_all, RL_stats)

    # Initial/final mother radius: R_L at the earliest/latest
    # global_time_min in the combined dataset. These are no longer
    # used as a fixed radial coordinate for the polar/karyograph plots
    # (each contact point is now plotted at its OWN r_contact) — they
    # are kept only as optional reference lines/context in the
    # karyograph plots below.
    RL_stats_sorted = RL_stats.sort_values("global_time_min")
    r_initial = float(RL_stats_sorted.iloc[0]["R_L"])
    r_final = float(RL_stats_sorted.iloc[-1]["R_L"])
    print(f"[INFO] Initial mother radius R_L = {r_initial:.4f} um, "
          f"final mother radius R_L = {r_final:.4f} um "
          f"(used only as reference lines in karyograph plots).")

    R_df = pair_df.merge(RL_stats, on="global_time_min", how="inner")

    t_min = R_df["global_time_min"]

    # =====================================================
    # PLOTS
    # =====================================================
    plt.figure(figsize=(6, 6))
    plt.errorbar(t_min, R_df["MSS"], yerr=R_df["dMSS"], fmt="o", ms=2, capsize=2, label="MSS")
    plt.errorbar(t_min, R_df["MLS"], yerr=R_df["dMLS"], fmt="o", ms=2, capsize=2, label="MLS")
    plt.xlabel("Time (min)")
    plt.ylabel(r"$M_{ne}\ (\mu m^{-1})$")
    leg = plt.legend(
        loc='upper right',
        fontsize=7,
        frameon=True,
        fancybox=True,
        framealpha=0.75,
        edgecolor='0.7',
        handlelength=1.2,
        borderpad=0.3,
        labelspacing=0.3,
        markerscale=0.9
    )
    leg.get_frame().set_facecolor('white')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    mystyle.save_pdf_png(plt.gcf(), f"{base_dir}/fig5")
    plt.close() #plt.show()

    #figure 6
    plt.figure(figsize=(6, 6))
    plt.errorbar(t_min, R_df["B_low"], yerr=R_df["dB_low"], fmt="o", ms=2, capsize=2,label=r"$B_{low}$")
    plt.xlabel("Time (min)")
    plt.ylabel(r"$B_{low}\ (\mu m^{-1})$")
    leg = plt.legend(
        loc='upper right',
        fontsize=7,
        frameon=True,
        fancybox=True,
        framealpha=0.75,
        edgecolor='0.7',
        handlelength=1.2,
        borderpad=0.3,
        labelspacing=0.3,
        markerscale=0.9
    )
    leg.get_frame().set_facecolor('white')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    mystyle.save_pdf_png(plt.gcf(), f"{base_dir}/fig6")
    plt.close() #plt.show()


    # =========================================================
    # POLAR MLS / MSS PLOTS (theta vs. each contact's OWN r_contact;
    # two versions per class — colored by value, colored by time)
    # =========================================================
    for color_by in ("value", "time"):
        plot_polar_contact_class(mss_points_df, "MSS", "MSS", base_dir,limit_radial, color_by=color_by)
        plot_polar_contact_class(mls_points_df, "MLS", "MLS", base_dir,limit_radial, color_by=color_by)

    # =========================================================
    # KARYOGRAPH PLOTS (R vs. Theta, cartesian; two colorbars each,
    # plus a combined MSS+MLS overlay colored by time)
    # =========================================================
    plot_karyograph_contact_class(mss_points_df, "MSS", "MSS", base_dir, r_initial, r_final)
    plot_karyograph_contact_class(mls_points_df, "MLS", "MLS", base_dir, r_initial, r_final)
    plot_combined_karyograph(mss_points_df, mls_points_df, base_dir)


    # =========================================================
    # FIG 5b — Global distributions
    # =========================================================

    plt.figure(figsize=(7, 4))

    weights_mss = np.ones_like(R_df["MSS"]) / len(R_df["MSS"])
    weights_mls = np.ones_like(R_df["MLS"]) / len(R_df["MLS"])

    plt.hist(
        R_df["MSS"],
        bins=60,
        weights=weights_mss,
        alpha=0.6,
        label="MSS"
    )

    plt.hist(
        R_df["MLS"],
        bins=60,
        weights=weights_mls,
        alpha=0.6,
        label="MLS"
    )

    plt.xlabel(r"$M_{ne}\ (\mu m^{-1})$")
    plt.ylabel("Probability")
    plt.ylim(0,1)
    leg = plt.legend(
        loc='upper right',
        fontsize=7,
        frameon=True,
        fancybox=True,
        framealpha=0.75,
        edgecolor='0.7',
        handlelength=1.2,
        borderpad=0.3,
        labelspacing=0.3,
        markerscale=0.9
    )
    leg.get_frame().set_facecolor('white')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend()
    plt.tight_layout()
    mystyle.save_pdf_png(plt.gcf(), f"{base_dir}/fig5_b")
    plt.close() #plt.show()
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(R_df["MSS"], alpha=0.6, label="MSS") 
    plt.hist(R_df["MLS"], alpha=0.6, label="MLS") 
    plt.xlabel(r"$M_{ne}\ (\mu m^{-1})$")
    plt.ylabel("Counts")
    leg = plt.legend(
        loc='upper right',
        fontsize=7,
        frameon=True,
        fancybox=True,
        framealpha=0.75,
        edgecolor='0.7',
        handlelength=1.2,
        borderpad=0.3,
        labelspacing=0.3,
        markerscale=0.9
    )
    leg.get_frame().set_facecolor('white')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend()
    plt.tight_layout()
    mystyle.save_pdf_png(plt.gcf(), f"{base_dir}/fig5_b_pdf")
    plt.close() #plt.show()
    plt.close()


    # =========================================================
    # FIG 6b — Global distribution
    # =========================================================

    '''plt.figure(figsize=(7, 4))
    plt.hist(R_df["B_low"], bins=60, density=True, alpha=0.7)
    plt.xlabel(r"$B_{low}\ (\mu m^{-1})$")
    plt.ylabel("Probability density")
    plt.tight_layout()
    plt.savefig(f"{base_dir}/fig6_b.png", dpi=300)
    plt.close() #plt.show()
    plt.close()'''
    
    plt.figure(figsize=(7, 4))

    weights_blow = np.ones_like(R_df["B_low"]) / len(R_df["B_low"])

    
    plt.hist(
        R_df["B_low"],
        bins=60,
        weights=weights_blow,
        alpha=0.6,
        label=r"$B_{low}$"
    )

    plt.xlabel(r"$B_{low}\ (\mu m^{-1})$")
    plt.ylabel("Probability")
    plt.ylim(0,1)
    leg = plt.legend(
        loc='upper right',
        fontsize=7,
        frameon=True,
        fancybox=True,
        framealpha=0.75,
        edgecolor='0.7',
        handlelength=1.2,
        borderpad=0.3,
        labelspacing=0.3,
        markerscale=0.9
    )
    leg.get_frame().set_facecolor('white')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    mystyle.save_pdf_png(plt.gcf(), f"{base_dir}/fig6_b")
    plt.close() #plt.show()
    plt.close()

    
    plt.figure(figsize=(7, 4))
    plt.hist(R_df["B_low"], alpha=0.7,label=r"$B_{low}$")
    plt.ylabel("Counts")
    plt.xlabel(r"$B_{low}\ (\mu m^{-1})$")
    leg = plt.legend(
        loc='upper right',
        fontsize=7,
        frameon=True,
        fancybox=True,
        framealpha=0.75,
        edgecolor='0.7',
        handlelength=1.2,
        borderpad=0.3,
        labelspacing=0.3,
        markerscale=0.9
    )
    leg.get_frame().set_facecolor('white')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend()
    plt.tight_layout()
    mystyle.save_pdf_png(plt.gcf(), f"{base_dir}/fig6_b_hist")
    plt.close() #plt.show()
    plt.close()


    # =========================================================
    # FIG 5c / 6c — COLUMN BOX PLOTS (distribution over time)
    # =========================================================

    bin_width = 1  # minutes
    time_bins = np.arange(t_min.min(), t_min.max() + bin_width, bin_width)
    bin_centers = 0.5 * (time_bins[:-1] + time_bins[1:])

    def binned_values(x, y, bins):
        return [y[(x >= lo) & (x < hi)] for lo, hi in zip(bins[:-1], bins[1:])]

    def binned_stats(x, y, bins):
        means, stds = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            vals = y[(x >= lo) & (x < hi)]
            means.append(vals.mean() if len(vals) > 0 else np.nan)
            stds.append(vals.std() if len(vals) > 1 else np.nan)
        return np.array(means), np.array(stds)

    # -------- FIG 5c --------
    mss_bins = binned_values(t_min.values, R_df["MSS"].values, time_bins)
    mls_bins = binned_values(t_min.values, R_df["MLS"].values, time_bins)

    mss_mean, mss_std = binned_stats(t_min.values, R_df["MSS"].values, time_bins)
    mls_mean, mls_std = binned_stats(t_min.values, R_df["MLS"].values, time_bins)
    
    plt.figure(figsize=(10, 4))

    bp_mss = plt.boxplot(
        mss_bins,
        positions=bin_centers,
        widths=bin_width * 0.35,
        patch_artist=True,
        showfliers=False
    )

    bp_mls = plt.boxplot(
        mls_bins,
        positions=bin_centers + bin_width * 0.35,
        widths=bin_width * 0.35,
        patch_artist=True,
        showfliers=False
    )

    plt.errorbar(bin_centers, mss_mean, yerr=mss_std, fmt="o", capsize=3)
    plt.errorbar(bin_centers + bin_width * 0.35, mls_mean, yerr=mls_std, fmt="o", capsize=3)

    plt.xlabel("Time (min)")
    plt.ylabel(r"$M_{ne}\ (\mu m^{-1})$")

    plt.xlim(time_bins[0], time_bins[-1])
    import matplotlib.ticker as ticker

    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(10))  # tick every 10 min
    plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(1))   # small ticks every 1 min

    plt.gca().xaxis.set_major_formatter(lambda x, pos: f"{int(x)}")

    legend_handles = [
    Patch(facecolor="#1f77b4", edgecolor="black", label="MSS"),
    Patch(facecolor="#ff7f0e", edgecolor="black", label="MLS")
    ]

    leg = plt.legend(
        handles=legend_handles,
        loc='upper right',
        fontsize=7,
        frameon=True,
        fancybox=True,
        framealpha=0.75,
        edgecolor='0.7',
        handlelength=1.2,
        borderpad=0.3,
        labelspacing=0.3,
        markerscale=0.9
    )
    leg.get_frame().set_facecolor('white')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    mystyle.save_pdf_png(plt.gcf(), f"{base_dir}/fig5_c")
    plt.close() #plt.show()
    plt.close()


    # -------- FIG 6c --------
    bl_bins = binned_values(t_min.values, R_df["B_low"].values, time_bins)
    bl_mean, bl_std = binned_stats(t_min.values, R_df["B_low"].values, time_bins)

    plt.figure(figsize=(10, 4))
    plt.boxplot(bl_bins, positions=bin_centers,
                widths=bin_width * 0.5,
                patch_artist=True, showfliers=False)

    plt.errorbar(bin_centers, bl_mean, yerr=bl_std, fmt="o", capsize=3)

    plt.xlabel("Time (min)")
    plt.ylabel(r"$B_{low}\ (\mu m^{-1})$")

    plt.xlim(time_bins[0], time_bins[-1])
    import matplotlib.ticker as ticker

    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(10))  # tick every 10 min
    plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(1))   # small ticks every 1 min
    plt.gca().xaxis.set_major_formatter(lambda x, pos: f"{int(x)}")
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    mystyle.save_pdf_png(plt.gcf(), f"{base_dir}/fig6_c")
    plt.close() #plt.show()
    plt.close()

    # =========================================================
    # FIG 5d — Individual MSS / MLS vs time
    # =========================================================

    plt.figure(figsize=(8, 4))

    plt.scatter(
        R_df["global_time_min"],
        R_df["MSS"],
        s=15,
        alpha=0.6,
        label="MSS"
    )

    plt.scatter(
        R_df["global_time_min"],
        R_df["MLS"],
        s=15,
        alpha=0.6,
        label="MLS"
    )

    plt.xlabel("Time (min)")
    plt.ylabel(r"$M_{ne}\ (\mu m^{-1})$")

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.legend()
    plt.tight_layout()
    mystyle.save_pdf_png(plt.gcf(), f"{base_dir}/fig5_d")
    plt.close() #plt.show()
    plt.close()

    # =========================================================
    # FIG 6d — Individual B_low vs time
    # =========================================================

    plt.figure(figsize=(8, 4))

    plt.scatter(
        R_df["global_time_min"],
        R_df["B_low"],
        s=15,
        alpha=0.6,
        label=r"$B_{low}$"
    )

    plt.xlabel("Time (min)")
    plt.ylabel(r"$B_{low}\ (\mu m^{-1})$")

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.legend()
    plt.tight_layout()
    mystyle.save_pdf_png(plt.gcf(), f"{base_dir}/fig6_d")
    plt.close() #plt.show()
    plt.close()

    # =========================================================
    # SAVE CURVATURE TIME-SERIES CSV
    # =========================================================

    curvature_ts = pd.DataFrame({
        "time_min": R_df["global_time_min"],
        "R_L": R_df["R_L"],
        "dR_L": R_df["dR_L"],
        "MSS": R_df["MSS"],
        "dMSS": R_df["dMSS"],
        "MLS": R_df["MLS"],
        "dMLS": R_df["dMLS"],
        "B_low": R_df["B_low"],
        "dB_low": R_df["dB_low"],
    })

    curvature_ts.to_csv(f"{base_dir}/curvature_time_series.csv", index=False)

    # =========================================================
    # SAVE CURVATURE BINNED CSV (Fig 5c / 6c)
    # =========================================================

    curvature_binned = pd.DataFrame({
        "time_bin_center_min": bin_centers,
        "MSS_mean": mss_mean,
        "MSS_std": mss_std,
        "MLS_mean": mls_mean,
        "MLS_std": mls_std,
        "B_low_mean": bl_mean,
        "B_low_std": bl_std,
    })

    curvature_binned.to_csv(f"{base_dir}/curvature_binned.csv", index=False)

    # =========================================================
    # SAVE POLAR / KARYOGRAPH CONTACT POINT CSVs
    # (theta, r_contact — each contact's own radial position — plus
    # curvature value and time)
    # =========================================================

    if not mss_points_df.empty:
        mss_points_df.to_csv(f"{base_dir}/polar_contacts_MSS.csv", index=False)

    if not mls_points_df.empty:
        mls_points_df.to_csv(f"{base_dir}/polar_contacts_MLS.csv", index=False)

    # =========================================================
    # FIG 5e — Gaussian Fit (MSS & MLS)
    # =========================================================

    from scipy.stats import norm

    plt.figure(figsize=(7, 4))

    # ---------- MSS ----------
    data_mss = R_df["MSS"].dropna().values
    counts_mss, bins_mss, _ = plt.hist(
        data_mss,
        bins=50,
        alpha=0.5,
        label="MSS",
    )

    bin_centers_mss = 0.5 * (bins_mss[:-1] + bins_mss[1:])
    bin_width_mss = bins_mss[1] - bins_mss[0]

    mu_mss, sigma_mss = norm.fit(data_mss)
    pdf_mss = norm.pdf(bin_centers_mss, mu_mss, sigma_mss)
    pdf_mss_scaled = pdf_mss * len(data_mss) * bin_width_mss

    plt.plot(
        bin_centers_mss,
        pdf_mss_scaled,
        linewidth=2,
        linestyle="--",
        label=f"MSS fit (μ={mu_mss:.3f})"
    )

    print("MSS Gaussian mean:", mu_mss)
    print("MSS Gaussian median:", mu_mss)


    # ---------- MLS ----------
    data_mls = R_df["MLS"].dropna().values
    counts_mls, bins_mls, _ = plt.hist(
        data_mls,
        bins=50,
        alpha=0.5,
        label="MLS",
    )

    bin_centers_mls = 0.5 * (bins_mls[:-1] + bins_mls[1:])
    bin_width_mls = bins_mls[1] - bins_mls[0]

    mu_mls, sigma_mls = norm.fit(data_mls)
    pdf_mls = norm.pdf(bin_centers_mls, mu_mls, sigma_mls)
    pdf_mls_scaled = pdf_mls * len(data_mls) * bin_width_mls

    plt.plot(
        bin_centers_mls,
        pdf_mls_scaled,
        linewidth=2,
        linestyle="--",
        label=f"MLS fit (μ={mu_mls:.3f})"
    )

    print("MLS Gaussian mean:", mu_mls)
    print("MLS Gaussian median:", mu_mls)

    plt.xlabel(r"$M_{ne}\ (\mu m^{-1})$")
    plt.ylabel("Counts")
    plt.legend()
    plt.tight_layout()
    mystyle.save_pdf_png(plt.gcf(), f"{base_dir}/fig5_e")
    plt.close() #plt.show()
    plt.close()


    # =========================================================
    # FIG 6e — Gaussian Fit (B_low)
    # =========================================================

    plt.figure(figsize=(7, 4))

    data_b = R_df["B_low"].dropna().values

    counts_b, bins_b, _ = plt.hist(
        data_b,
        bins=50,
        alpha=0.6,
        label=r"$B_{low}$"
    )

    bin_centers_b = 0.5 * (bins_b[:-1] + bins_b[1:])
    bin_width_b = bins_b[1] - bins_b[0]

    mu_b, sigma_b = norm.fit(data_b)
    pdf_b = norm.pdf(bin_centers_b, mu_b, sigma_b)
    pdf_b_scaled = pdf_b * len(data_b) * bin_width_b

    plt.plot(
        bin_centers_b,
        pdf_b_scaled,
        linewidth=2,
        linestyle="--",
        label=f"Gaussian fit (μ={mu_b:.3f})"
    )

    print("B_low Gaussian mean:", mu_b)
    print("B_low Gaussian median:", mu_b)

    plt.xlabel(r"$B_{low}\ (\mu m^{-1})$")
    plt.ylabel("Counts")
    plt.legend()
    plt.tight_layout()
    mystyle.save_pdf_png(plt.gcf(), f"{base_dir}/fig6_e")
    plt.close() #plt.show()
    plt.close()

    from scipy.stats import skewnorm

    # =========================================================
    # FIG 5f — Skew-Normal Fit (MSS & MLS)
    # =========================================================

    plt.figure(figsize=(7, 4))

    # -------- MSS --------
    data_mss = R_df["MSS"].dropna().values

    counts_mss, bins_mss, _ = plt.hist(
        data_mss,
        bins=50,
        alpha=0.5,
        label="MSS"
    )

    bin_centers_mss = 0.5 * (bins_mss[:-1] + bins_mss[1:])
    bin_width_mss = bins_mss[1] - bins_mss[0]

    a_mss, loc_mss, scale_mss = skewnorm.fit(data_mss)

    pdf_mss = skewnorm.pdf(bin_centers_mss, a_mss, loc_mss, scale_mss)
    pdf_mss_scaled = pdf_mss * len(data_mss) * bin_width_mss

    plt.plot(
        bin_centers_mss,
        pdf_mss_scaled,
        linewidth=2,
        linestyle="--",
        label=f"MSS skew-fit (α={a_mss:.2f})"
    )

    print("\nMSS skew-normal parameters:")
    print("alpha:", a_mss)
    print("location:", loc_mss)
    print("scale:", scale_mss)


    # -------- MLS --------
    data_mls = R_df["MLS"].dropna().values

    counts_mls, bins_mls, _ = plt.hist(
        data_mls,
        bins=50,
        alpha=0.5,
        label="MLS"
    )

    bin_centers_mls = 0.5 * (bins_mls[:-1] + bins_mls[1:])
    bin_width_mls = bins_mls[1] - bins_mls[0]

    a_mls, loc_mls, scale_mls = skewnorm.fit(data_mls)

    pdf_mls = skewnorm.pdf(bin_centers_mls, a_mls, loc_mls, scale_mls)
    pdf_mls_scaled = pdf_mls * len(data_mls) * bin_width_mls

    plt.plot(
        bin_centers_mls,
        pdf_mls_scaled,
        linewidth=2,
        linestyle="--",
        label=f"MLS skew-fit (α={a_mls:.2f})"
    )

    print("\nMLS skew-normal parameters:")
    print("alpha:", a_mls)
    print("location:", loc_mls)
    print("scale:", scale_mls)

    plt.xlabel(r"$M_{ne}\ (\mu m^{-1})$")
    plt.ylabel("Counts")
    plt.legend()
    plt.tight_layout()
    mystyle.save_pdf_png(plt.gcf(), f"{base_dir}/fig5_f")
    plt.close() #plt.show()
    plt.close()
    # =========================================================
    # FIG 6f — Skew-Normal Fit (B_low)
    # =========================================================

    plt.figure(figsize=(7, 4))

    data_b = R_df["B_low"].dropna().values

    counts_b, bins_b, _ = plt.hist(
        data_b,
        bins=50,
        alpha=0.6,
        label=r"$B_{low}$"
    )

    bin_centers_b = 0.5 * (bins_b[:-1] + bins_b[1:])
    bin_width_b = bins_b[1] - bins_b[0]

    a_b, loc_b, scale_b = skewnorm.fit(data_b)

    pdf_b = skewnorm.pdf(bin_centers_b, a_b, loc_b, scale_b)
    pdf_b_scaled = pdf_b * len(data_b) * bin_width_b

    plt.plot(
        bin_centers_b,
        pdf_b_scaled,
        linewidth=2,
        linestyle="--",
        label=f"Skew-normal fit (α={a_b:.2f})"
    )

    print("\nB_low skew-normal parameters:")
    print("alpha:", a_b)
    print("location:", loc_b)
    print("scale:", scale_b)

    plt.xlabel(r"$B_{low}\ (\mu m^{-1})$")
    plt.ylabel("Counts")
    plt.legend()
    plt.tight_layout()
    mystyle.save_pdf_png(plt.gcf(), f"{base_dir}/fig6_f")
    plt.close() #plt.show()
    plt.close()

    from sklearn.mixture import GaussianMixture
    from scipy.stats import norm

    plt.figure(figsize=(7, 4))

    # ---------- MSS ----------
    data_mss = R_df["MSS"].dropna().values

    counts_mss, bins_mss, _ = plt.hist(
        data_mss,
        bins=50,
        alpha=0.5,
        label="MSS",
    )

    bin_width_mss = bins_mss[1] - bins_mss[0]

    gmm_mss = GaussianMixture(n_components=2, random_state=0)
    gmm_mss.fit(data_mss.reshape(-1, 1))

    means_mss = gmm_mss.means_.flatten()
    sigmas_mss = np.sqrt(gmm_mss.covariances_).flatten()
    weights_mss = gmm_mss.weights_

    x_mss = np.linspace(bins_mss[0], bins_mss[-1], 1000)
    pdf_total_mss = np.zeros_like(x_mss)

    for w, mu, sigma in zip(weights_mss, means_mss, sigmas_mss):
        pdf_total_mss += w * norm.pdf(x_mss, mu, sigma)

    pdf_total_scaled_mss = pdf_total_mss * len(data_mss) * bin_width_mss

    plt.plot(
        x_mss,
        pdf_total_scaled_mss,
        linestyle="--",
        linewidth=2,
        label="MSS 2-Gaussian fit"
    )

    print("MSS Means:", means_mss)
    print("MSS Sigmas:", sigmas_mss)
    print("MSS Weights:", weights_mss)


    # ---------- MLS ----------
    data_mls = R_df["MLS"].dropna().values

    counts_mls, bins_mls, _ = plt.hist(
        data_mls,
        bins=50,
        alpha=0.5,
        label="MLS",
    )

    bin_width_mls = bins_mls[1] - bins_mls[0]

    gmm_mls = GaussianMixture(n_components=2, random_state=0)
    gmm_mls.fit(data_mls.reshape(-1, 1))

    means_mls = gmm_mls.means_.flatten()
    sigmas_mls = np.sqrt(gmm_mls.covariances_).flatten()
    weights_mls = gmm_mls.weights_

    x_mls = np.linspace(bins_mls[0], bins_mls[-1], 1000)
    pdf_total_mls = np.zeros_like(x_mls)

    for w, mu, sigma in zip(weights_mls, means_mls, sigmas_mls):
        pdf_total_mls += w * norm.pdf(x_mls, mu, sigma)

    pdf_total_scaled_mls = pdf_total_mls * len(data_mls) * bin_width_mls

    plt.plot(
        x_mls,
        pdf_total_scaled_mls,
        linestyle="--",
        linewidth=2,
        label="MLS 2-Gaussian fit"
    )

    print("MLS Means:", means_mls)
    print("MLS Sigmas:", sigmas_mls)
    print("MLS Weights:", weights_mls)

    plt.xlabel(r"$M_{ne}\ (\mu m^{-1})$")
    plt.ylabel("Counts")
    plt.legend()
    plt.tight_layout()
    mystyle.save_pdf_png(plt.gcf(), f"{base_dir}/fig5_g")
    plt.close() #plt.show()
    plt.close()

    plt.figure(figsize=(7, 4))

    data_b = R_df["B_low"].dropna().values

    counts_b, bins_b, _ = plt.hist(
        data_b,
        bins=50,
        alpha=0.6,
        label=r"$B_{low}$"
    )

    bin_width_b = bins_b[1] - bins_b[0]

    gmm_b = GaussianMixture(n_components=2, random_state=0)
    gmm_b.fit(data_b.reshape(-1, 1))

    means_b = gmm_b.means_.flatten()
    sigmas_b = np.sqrt(gmm_b.covariances_).flatten()
    weights_b = gmm_b.weights_

    x_b = np.linspace(bins_b[0], bins_b[-1], 1000)
    pdf_total_b = np.zeros_like(x_b)

    for w, mu, sigma in zip(weights_b, means_b, sigmas_b):
        pdf_total_b += w * norm.pdf(x_b, mu, sigma)

    pdf_total_scaled_b = pdf_total_b * len(data_b) * bin_width_b

    plt.plot(
        x_b,
        pdf_total_scaled_b,
        linestyle="--",
        linewidth=2,
        label="2-Gaussian fit"
    )

    print("B_low Means:", means_b)
    print("B_low Sigmas:", sigmas_b)
    print("B_low Weights:", weights_b)

    plt.xlabel(r"$B_{low}\ (\mu m^{-1})$")
    plt.ylabel("Counts")
    plt.legend()
    plt.tight_layout()
    mystyle.save_pdf_png(plt.gcf(), f"{base_dir}/fig6_g")
    plt.close() #plt.show()
    plt.close()


# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":
    selected_base_dir = select_results_directory()

    selected_exp_folders = sorted(
        [f for f in os.listdir(selected_base_dir) if f.startswith("Experiment-")],
        key=lambda x: int(x.split("-")[-1])
    )

    selected_delays, selected_tpf, selected_fs = prompt_run_parameters(selected_exp_folders)

    plot_contact_curvature(selected_base_dir, selected_delays, selected_tpf, selected_fs)