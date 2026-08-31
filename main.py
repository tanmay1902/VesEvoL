"""
Top-Level Orchestration Script: Batch Mother/Daughter Vesicle Detection & Tracking Pipeline.

Walks the user through selecting an input directory of microscopy stacks (.czi/.tif),
then for each file: loads and crops the stack, detects the "mother" vesicle boundary
per frame, optionally deconvolves the frames, detects and locally tracks "daughter"
vesicles, and writes per-file mother/daughter CSVs. After all files are processed it
reassembles a combined summary dataset from the per-file CSVs on disk.

See README.md and docs/PIPELINE.md for a full description of the pipeline stages,
expected inputs/outputs, and directory layout.

Contact Details:
Email(s): ms22113@iisermohali.ac.in , caritra@iitg.ac.in
"""
SETTING_DECONVULATION = False

# Set to True only if you need the raw per-frame image/mask cache written to disk
# (e.g. for offline debugging). It is NOT needed for local tracking or for the
# combined summary CSVs below, and it is the single largest thing you can hold
# in memory/disk per file, so it defaults to off.
SAVE_RAW_NPZ = False

import gc
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import easygui as eg
from mother_detection import MotherTuner, detect_mother_stack_parallelized
import numpy as np
import pandas as pd

from detect_and_track import process_daughters_parallel
from modules.deconvulation import deconvolve_frames, gaussian_psf
from modules.global_rect_roi import select_roi_per_frame_ranges
from modules.read_czi_tiff import load_stack_efficient, read_directory, read_voxel_size
from modules.results import compile_experiment_data
from modules.savenpzraw import save_npz_raw
from modules.slicer import select_frame_range


def main():
    files, directory_path = read_directory()
    if not files:
        return

    tot_dir = len(files)
    set_pixel = False
    main_out = Path(directory_path) / 'results_v2'

    roi_size = None
    if eg.ynbox("Want to use prev roi size defined from last run? ", "SMLB"):
        askroi_h = int(eg.enterbox("Enter height of ROI if there: ", "SMBL"))
        askroi_w = int(eg.enterbox("Enter width of ROI if there: ", "SMBL"))
        roi_size = (askroi_h, askroi_w)

    # Execution mode is asked once per run (not per file). Parallel spreads work across
    # CPU worker processes and is faster but shows no progress bar (a live progress bar
    # can't be shared across separate processes). Sequential runs one frame at a time in
    # this process and shows a progress bar.
    run_parallel = eg.ynbox(
        "Choose execution mode for mother/daughter detection:\n\n"
        "Parallel  -> faster, uses multiple CPU cores, no progress bar\n"
        "Sequential -> single core, shows a progress bar",
        "Execution Mode",
        choices=("Parallel", "Sequential")
    )

    for fid, file in enumerate(files):
        # Clear stdout terminal depending on operating system type
        if sys.platform in ('linux', 'darwin'):
            os.system('clear')
        else:
            os.system('cls')

        print(f"***** {directory_path} | Current : {fid+1}/{tot_dir} ({fid*100/tot_dir:.1f}%) ****")
        for path in files:
            size_gb = os.path.getsize(path) / (1024 ** 3)
            if os.path.abspath(path) == os.path.abspath(file):
                print(f">> {os.path.basename(path)} | Size: {size_gb:.2f} GB <<")
            else:
                print(f"{os.path.basename(path)} | Size: {size_gb:.2f} GB")

        if not set_pixel:
            px = float(read_voxel_size(file))
            print(px)
        fname = os.path.splitext(os.path.basename(file))[0]
        out_dir = main_out / fname
        out_dir.mkdir(parents=True, exist_ok=True)

        npz_path = out_dir / "detections.npz"
        mother_csv = out_dir / "mother.csv"
        daughter_csv = out_dir / "daughter.csv"

        if daughter_csv.exists() and (npz_path.exists() or not SAVE_RAW_NPZ):
            print(f"⏩ Skipping {fname} (already processed)")
            continue

        # 1. Load & Crop Stack Elements
        try:
            stack = load_stack_efficient(file)
            stack = np.asarray(stack)
            if stack.ndim == 3:
                frames = [stack[i] for i in range(stack.shape[0])]
            else:
                frames = [stack]

            # Free the raw stack once split into per-frame views; frames is what we use from here.
            del stack

            start_frame, end_frame = select_frame_range(frames)
            frames = frames[start_frame:end_frame + 1]
            frames = select_roi_per_frame_ranges(frames, roi_size)

            if roi_size is None:
                h, w = frames[0].shape
                roi_size = (h, w)
                print(f"Using ROI: {roi_size}")

        except Exception as e:
            print(f"Error loading {fname}: {e}")
            continue

        # Only keep a pre-deconvolution copy of the frames if deconvolution is
        # actually going to run - this avoids silently doubling frame memory
        # for every file when the flag is off (the common case).
        og_frames = list(frames) if SETTING_DECONVULATION else None

        # 2. Mother Boundary Detection and Polar Spatial Curvature Matrix Extraction
        print("--- Step 1: Mother Detection ---")
        m_params = MotherTuner.tune(frames, px)

        m_data = detect_mother_stack_parallelized(
            frames, px, m_params,
            save_data=True,
            output_folder=str(out_dir / "mother"),
            save_training=False,
            save_curvature=True,
            parallel=run_parallel
        )

        try:
            rows = []
            n = len(m_data["area_um2"])

            for i in range(n):
                centroid = m_data["centroid"][i]
                rows.append({
                    "frame": i,
                    "area_um2": m_data["area_um2"][i],
                    "perimeter_um": m_data["perimeter_um"][i],
                    "radius_um": m_data["radius_um"][i],
                    "centroid_x": centroid[0] if centroid is not None else np.nan,
                    "centroid_y": centroid[1] if centroid is not None else np.nan,
                    "score": m_data["score"][i],
                    "perimeter_px": m_data["perimeter_px"][i],
                    "area_px": m_data["area_px"][i],
                    "circularity": m_data["circularity"][i]
                })

            df_mother = pd.DataFrame(rows)
            csv_path = out_dir / "mother" / "mother_metrics.csv"
            df_mother.to_csv(csv_path, index=False)
            print(f"[OK] Mother metrics saved → {csv_path}")
            del df_mother, rows

        except Exception as e:
            print(f"[WARN] Failed to save mother CSV file: {e}")

        if "area_um2" in m_data:
            areas = np.array(m_data["area_um2"])
            with np.errstate(invalid='ignore'):
                radii_px = np.sqrt((areas / (px**2)) / np.pi)
            m_data["radius_px"] = radii_px.tolist()
            del areas, radii_px

        # 3. Microstructural Image Deconvolution Cascade
        deconv_params = {
            'sigma_px': 2.0,
            'size': 25,
            'iterations': 3,
            'background': 0.02,
            'method': "RL",
            'balance': 1e-4
        }

        if SETTING_DECONVULATION:
            psf = gaussian_psf(deconv_params["sigma_px"], deconv_params["size"])
            frames = deconvolve_frames(
                og_frames,
                psf,
                iterations=deconv_params["iterations"],
                method=deconv_params["method"],
                balance=deconv_params.get("balance", 1e-4),
                background=deconv_params["background"]
            )

        # og_frames is only ever needed as deconvolution input; drop it now.
        del og_frames

        # 4. Daughter Analysis (local tracking runs internally, per file)
        print("--- Step 2: Daughter Analysis ---")
        # "Parallel" means CPU multiprocessing here - a live GPU context can't be shared
        # across worker processes, so choosing Parallel also runs daughter detection on
        # CPU. Choosing Sequential keeps the existing GPU + progress-bar path.
        frame_cache, all_detections = process_daughters_parallel(
            frames, m_data, px,
            output_folder=str(out_dir),
            save_training=False,
            supraname=fname,
            iflocalTracking=True,
            parallel=run_parallel,
            use_gpu=not run_parallel
        )

        if SAVE_RAW_NPZ:
            save_npz_raw(
                str(out_dir),
                frame_cache=frame_cache,
                detections=all_detections,
                px=px
            )

        m_df, d_df = compile_experiment_data(
            m_data,
            {"stats": [], "regions": all_detections},
            px
        )

        m_df.to_csv(mother_csv, index=False)
        d_df.to_csv(daughter_csv, index=False)

        # ===== PER-FILE MEMORY RESET =====
        # Everything above is scoped to this file only - drop every large
        # reference before moving on, so peak memory never exceeds what a
        # single file needs, regardless of how many files are in the folder.
        del frames, frame_cache, all_detections, m_data, m_df, d_df
        gc.collect()
        # ==================================

    # ================= COMBINED SUMMARY (built from disk, not memory) =================
    # Per-file mother.csv / daughter.csv are small (no raw images/masks), so we can
    # safely reassemble a combined view of everything processed so far by reading
    # them back from disk, instead of holding every file's data in RAM during the run.
    print("\n=== Building combined summary from per-file results ===")
    mother_frames = []
    daughter_frames = []

    for file in files:
        fname = os.path.splitext(os.path.basename(file))[0]
        out_dir = main_out / fname
        mother_csv = out_dir / "mother.csv"
        daughter_csv = out_dir / "daughter.csv"

        if not (mother_csv.exists() and daughter_csv.exists()):
            continue

        m_df = pd.read_csv(mother_csv)
        d_df = pd.read_csv(daughter_csv)
        m_df["video"] = fname
        d_df["video"] = fname

        mother_frames.append(m_df)
        daughter_frames.append(d_df)

    if mother_frames:
        mother_all = pd.concat(mother_frames, ignore_index=True)
        daughter_all = pd.concat(daughter_frames, ignore_index=True)

        user_input_dir = eg.enterbox("Enter a name for CSV result directory:")
        final_out = main_out / (user_input_dir if user_input_dir else "combined_results")
        final_out.mkdir(parents=True, exist_ok=True)

        mother_all.to_csv(final_out / "mother_all.csv", index=False)
        daughter_all.to_csv(final_out / "inner_daughter_all.csv", index=False)
        print(f"\n✅ Combined per-file dataset sheets saved in: {final_out}")
    else:
        print("⚠ No valid operational logs found in target directory structure.")


if __name__ == "__main__":
    main()