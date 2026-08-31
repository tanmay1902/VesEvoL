#!/usr/bin/env python3
"""
arrange_polar_plots.py

Assembles existing polar-plot PNGs (fig_polar_MLS_value.png, fig_polar_MLS_time.png,
fig_polar_MSS_value.png, fig_polar_MSS_time.png) from a "Complete Data" directory
into three publication-ready composite figures:

    combined_polar_MLS.png / .pdf          (4x4)
    combined_polar_MSS.png / .pdf          (4x4)
    combined_polar_MSS_MLS.png / .pdf      (4x8 with demarcation)

Layout for individual 4x4 figures (rows x columns):

                     1 uM   1.5 uM   2 uM   2.5 uM
    POPC - value       o      o       o       o
    POPC - time        o      o       o       o
    POPC:Chol - value  o      o       o       o
    POPC:Chol - time   o      o       o       o

Layout for combined 4x8 figure:

                MSS             |             MLS
                1uM  1.5uM  2uM  2.5uM | 1uM  1.5uM  2uM  2.5uM
    POPC - value  o    o     o    o    |  o    o     o    o
    POPC - time   o    o     o    o    |  o    o     o    o
    POPC:Chol - val o   o     o    o   |  o    o     o    o
    POPC:Chol - time o   o    o    o   |  o    o     o    o

For each (lipid, concentration) pair, the script looks at the immediate
subfolders of that concentration folder (GUV/experiment folders):
    - 0 found  -> cell marked Missing
    - 1 found  -> used automatically
    - >1 found -> EasyGUI choicebox lets you pick which GUV folder to use
    - choicebox cancelled -> cell marked Missing

Once a GUV folder is chosen, the script searches recursively inside it for
a folder named "results_v2" (which is expected to contain the PNGs).

Requires: easygui, matplotlib, Pillow (PIL), mystyle.py + mystyle.mplstyle
          (must sit next to this script, or edit MYSTYLE_DIR below).
"""

import os
import sys
import importlib.util
from collections import deque

import easygui
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

# ---------------------------------------------------------------------------
# Load mystyle.py explicitly (works regardless of current working directory)
# ---------------------------------------------------------------------------
MYSTYLE_DIR = os.path.dirname(os.path.abspath(__file__))
_mystyle_path = os.path.join(MYSTYLE_DIR, "mystyle.py")
_spec = importlib.util.spec_from_file_location("mystyle", _mystyle_path)
mystyle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mystyle)

mystyle.apply_style()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LIPID_FOLDERS = [
    ("POPC", "POPC"),
    ("POPC-Chol_Starch", "POPC:Chol"),
]

# Concentration "tokens" to match against folder names. Matching is
# case-insensitive, ignores spaces, and matches on a PREFIX -- so a token
# of "1uM" matches folders named "1uM", "1uM AA", "1uM_AA", etc.
CONC_TOKENS = [
    ("1uM AA", "1 \u00b5M"),
    ("1.5uM AA", "1.5 \u00b5M"),
    ("2uM AA", "2 \u00b5M"),
    ("2.5uM AA", "2.5 \u00b5M"),
]

METRICS = ["value", "time"]  # used to build fig_polar_<TYPE>_<metric>.png
FIG_TYPES = ["MLS", "MSS"]

RESULTS_DIRNAME = "results_v2"


# ---------------------------------------------------------------------------
# Step 1: ask the user to select the "Complete Data" directory
# ---------------------------------------------------------------------------
def select_root_directory() -> str:
    root_dir = easygui.diropenbox(
        title="Select the 'Complete Data' directory",
        msg="Select the 'Complete Data' directory containing POPC and "
            "POPC-Chol_Starch subfolders.",
    )
    if not root_dir:
        print("No directory selected. Exiting.")
        sys.exit(0)
    return root_dir


# ---------------------------------------------------------------------------
# Step 2: locate the concentration folder (name may vary, e.g. "1uM" or
#          "1uM AA"), then the GUV/experiment folders inside it, then
#          results_v2 inside the chosen GUV folder
# ---------------------------------------------------------------------------
def find_conc_folder(lipid_path: str, token: str):
    """Finds a subfolder of lipid_path whose name matches the concentration
    token as a prefix, ignoring case and spaces (e.g. token "1uM" matches
    folders named "1uM", "1uM AA", "1uM_AA", ...). Returns the full path,
    or None if no match. If several folders match, the first (alphabetical)
    is used and a warning is printed."""
    if not os.path.isdir(lipid_path):
        return None

    norm_token = token.lower().replace(" ", "").replace("_", "")
    candidates = []
    for entry in sorted(os.listdir(lipid_path)):
        full = os.path.join(lipid_path, entry)
        if not os.path.isdir(full):
            continue
        norm_entry = entry.lower().replace(" ", "").replace("_", "")
        if norm_entry.startswith(norm_token):
            candidates.append(full)

    if not candidates:
        return None
    if len(candidates) > 1:
        names = [os.path.basename(c) for c in candidates]
        print(f"  [warning] multiple folders match concentration '{token}' "
              f"in '{lipid_path}': {names}. Using '{names[0]}'.")
    return candidates[0]


def find_guv_folders(conc_path: str) -> list:
    """Immediate subdirectories of conc_path (each is a GUV/experiment
    folder)."""
    if not os.path.isdir(conc_path):
        return []
    entries = sorted(os.listdir(conc_path))
    return [os.path.join(conc_path, e) for e in entries
            if os.path.isdir(os.path.join(conc_path, e))]


def find_results_dir_in(guv_path: str, max_depth: int = 6, skip_dirnames=None):
    """Breadth-first search inside guv_path for a directory named
    results_v2, stopping the instant one is found.

    This is much faster than a full recursive glob when the GUV/experiment
    folder also contains large sibling trees (raw TIFF stacks, per-frame
    images, etc.) that would otherwise be scanned in their entirety before
    any match is returned. Folders are checked level-by-level, so a
    results_v2 that sits near the top of the tree is found almost
    instantly regardless of how much data lives elsewhere.
    """
    if skip_dirnames is None:
        # Common "heavy" folder names that are unlikely to contain
        # results_v2 and are safe to skip descending into. Extend this
        # set if your raw-data folders have other names.
        skip_dirnames = {"raw", "raw data", "raw_data"}

    queue = deque([(guv_path, 0)])
    while queue:
        current_dir, depth = queue.popleft()
        try:
            with os.scandir(current_dir) as it:
                subdirs = [e for e in it if e.is_dir(follow_symlinks=False)]
        except (PermissionError, FileNotFoundError, NotADirectoryError):
            continue

        for entry in subdirs:
            if entry.name == RESULTS_DIRNAME:
                return entry.path

        if depth < max_depth:
            for entry in subdirs:
                if entry.name.strip().lower() not in skip_dirnames:
                    queue.append((entry.path, depth + 1))

    return None


def select_guv_folder_for_cell(lipid_folder: str, conc_folder: str, conc_path: str):
    """Returns the selected GUV/experiment folder path, or None if
    missing/cancelled."""
    candidates = find_guv_folders(conc_path)

    if len(candidates) == 0:
        return None

    if len(candidates) == 1:
        return candidates[0]

    # Multiple GUV/experiment folders found -> ask the user
    labels = [os.path.basename(c) for c in candidates]
    choice = easygui.choicebox(
        msg=f"Multiple GUV/experiment folders found for "
            f"{lipid_folder} / {conc_folder}.\nSelect which one to use:",
        title="Select GUV folder",
        choices=labels,
    )
    if choice is None:
        return None
    idx = labels.index(choice)
    return candidates[idx]


# ---------------------------------------------------------------------------
# Step 3: build the selection table for all lipid x concentration cells
# ---------------------------------------------------------------------------
def build_selections(root_dir: str) -> dict:
    selections = {}
    print("\n=== Experiment selection summary ===")
    for lipid_folder, lipid_label in LIPID_FOLDERS:
        lipid_path = os.path.join(root_dir, lipid_folder)
        for conc_token, conc_label in CONC_TOKENS:
            conc_path = find_conc_folder(lipid_path, conc_token)

            guv_folder = None
            if conc_path is not None:
                guv_folder = select_guv_folder_for_cell(lipid_folder, conc_label, conc_path)

            results_dir = None
            if guv_folder is not None:
                print(f"  Searching for '{RESULTS_DIRNAME}' in "
                      f"'{os.path.basename(guv_folder)}' ...", end=" ", flush=True)
                results_dir = find_results_dir_in(guv_folder)
                print("found" if results_dir else "not found")
                if results_dir is None:
                    print(f"  [warning] '{RESULTS_DIRNAME}' not found inside "
                          f"'{guv_folder}' -> treating as Missing")

            selections[(lipid_folder, conc_token)] = results_dir

            if results_dir is None:
                print(f"  {lipid_label:12s} | {conc_label:8s} -> Missing")
            else:
                print(f"  {lipid_label:12s} | {conc_label:8s} -> "
                      f"{os.path.basename(guv_folder)}")
    print("=====================================\n")
    return selections


# ---------------------------------------------------------------------------
# Step 4: draw one cell (image or "Missing" placeholder)
# ---------------------------------------------------------------------------
def draw_image_cell(ax, image_path: str) -> bool:
    """Loads image_path into ax, preserving its aspect ratio. Returns True
    on success, False if the file is missing/unreadable."""
    if not image_path or not os.path.isfile(image_path):
        return False
    try:
        img = Image.open(image_path)
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        return True
    except Exception as exc:
        print(f"  [warning] Could not load image '{image_path}': {exc}")
        return False


def draw_missing_cell(ax):
    ax.set_facecolor("#f5f5f5")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.5, 0.5, "Missing", ha="center", va="center",
             fontsize=11, color="#999999", style="italic",
             transform=ax.transAxes)


# ---------------------------------------------------------------------------
# Step 5a: build one 4x4 composite figure (MLS or MSS)
# ---------------------------------------------------------------------------
def build_composite_figure(fig_type: str, selections: dict, root_dir: str):
    rows = [
        (LIPID_FOLDERS[0][0], LIPID_FOLDERS[0][1], "value"),
        (LIPID_FOLDERS[0][0], LIPID_FOLDERS[0][1], "time"),
        (LIPID_FOLDERS[1][0], LIPID_FOLDERS[1][1], "value"),
        (LIPID_FOLDERS[1][0], LIPID_FOLDERS[1][1], "time"),
    ]

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))

    print(f"--- Building {fig_type} composite ---")
    for i, (lipid_folder, lipid_label, metric) in enumerate(rows):
        for j, (conc_token, conc_label) in enumerate(CONC_TOKENS):
            ax = axes[i, j]
            results_dir = selections.get((lipid_folder, conc_token))

            image_ok = False
            if results_dir is not None:
                image_name = f"fig_polar_{fig_type}_{metric}.png"
                image_path = os.path.join(results_dir, image_name)
                image_ok = draw_image_cell(ax, image_path)
                if not image_ok:
                    print(f"  [missing file] {lipid_label} / {conc_label} "
                          f"({metric}): {image_name} not found in {results_dir}")

            if not image_ok:
                draw_missing_cell(ax)

            # Column headers on the top row
            if i == 0:
                ax.set_title(conc_label, fontsize=13, fontweight="bold", pad=10)

            # Row labels on the left column
            if j == 0:
                ax.set_ylabel(f"{lipid_label} \u2014 {metric}",
                               fontsize=12, fontweight="bold")

    fig.subplots_adjust(wspace=0.06, hspace=0.10, top=0.94)

    out_base = os.path.join(root_dir, f"combined_polar_{fig_type}")
    mystyle.save_pdf_png(fig, out_base)
    plt.close(fig)
    print(f"Saved {out_base}.png (and .pdf if enabled in mystyle.py)\n")


# ---------------------------------------------------------------------------
# Step 5b: build a combined 4x8 figure with MSS on left, MLS on right
# ---------------------------------------------------------------------------
def build_combined_mss_mls_figure(selections: dict, root_dir: str):
    rows = [
        (LIPID_FOLDERS[0][0], LIPID_FOLDERS[0][1], "value"),
        (LIPID_FOLDERS[0][0], LIPID_FOLDERS[0][1], "time"),
        (LIPID_FOLDERS[1][0], LIPID_FOLDERS[1][1], "value"),
        (LIPID_FOLDERS[1][0], LIPID_FOLDERS[1][1], "time"),
    ]

    fig, axes = plt.subplots(4, 8, figsize=(14, 8))

    print(f"--- Building MSS+MLS combined composite ---")
    
    # Process both figure types (MSS and MLS)
    for fig_type_idx, fig_type in enumerate(["MSS", "MLS"]):
        col_offset = fig_type_idx * 4  # MSS: 0-3, MLS: 4-7
        
        for i, (lipid_folder, lipid_label, metric) in enumerate(rows):
            for j, (conc_token, conc_label) in enumerate(CONC_TOKENS):
                ax = axes[i, col_offset + j]
                results_dir = selections.get((lipid_folder, conc_token))

                image_ok = False
                if results_dir is not None:
                    image_name = f"fig_polar_{fig_type}_{metric}.png"
                    image_path = os.path.join(results_dir, image_name)
                    image_ok = draw_image_cell(ax, image_path)
                    if not image_ok:
                        print(f"  [missing file] {lipid_label} / {conc_label} "
                              f"({metric}, {fig_type}): {image_name} not found in {results_dir}")

                if not image_ok:
                    draw_missing_cell(ax)

                # Column headers on the top row (show concentration labels)
                if i == 0:
                    ax.set_title(conc_label, fontsize=12, fontweight="bold", pad=8)

                # Row labels on the left column (MSS side only)
                if j == 0 and fig_type_idx == 0:
                    ax.set_ylabel(f"{lipid_label} \u2014 {metric}",
                                   fontsize=11, fontweight="bold")

    # Add section headers above the concentration labels
    fig.text(0.15, 0.985, "(a)", ha="center", fontsize=16, fontweight="bold",
             transform=fig.transFigure, color="#000000",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="none", alpha=0.8))
    fig.text(0.55, 0.985, "(b)", ha="center", fontsize=16, fontweight="bold",
             transform=fig.transFigure, color="#000000",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="none", alpha=0.8))

    fig.subplots_adjust(wspace=0.04, hspace=0.12, top=0.92)

    out_base = os.path.join(root_dir, "Figure 8")
    mystyle.save_pdf_png(fig, out_base)
    plt.close(fig)
    print(f"Saved {out_base}.png (and .pdf if enabled in mystyle.py)\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    root_dir = select_root_directory()
    selections = build_selections(root_dir)

    # Build individual 4x4 figures
    for fig_type in FIG_TYPES:
        build_composite_figure(fig_type, selections, root_dir)

    # Build combined 4x8 figure (MSS + MLS)
    build_combined_mss_mls_figure(selections, root_dir)

    print("Done. Composite figures saved in:", root_dir)


if __name__ == "__main__":
    main()