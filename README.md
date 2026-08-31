# Mother–Daughter Vesicle Detection & Tracking Pipeline

Supplementary code for **[Paper Title — add citation once available]**.

This repository contains the image-analysis pipeline used to detect a large "mother"
vesicle/cell boundary in each frame of a time-lapse microscopy stack, detect and
locally track smaller "daughter" vesicles inside/around it, and quantify their
geometric contacts (mother–daughter and daughter–daughter) over time.

> **Status: pre-release.** This repo currently contains the three core driver
> scripts (`main.py`, `mother_detection.py`, `detect_and_track.py`). The
> `modules/` helper package and the plotting / raw-data-processing scripts are
> being added — see [Repository status](#repository-status) below.

---

## Pipeline overview

```
main.py
 │
 ├─ 1. read_directory()                 select a folder of .czi/.tif stacks
 │
 ├─ For each file:
 │   ├─ load_stack_efficient()          load stack, split into per-frame arrays
 │   ├─ select_frame_range()            crop to a frame range (interactive)
 │   ├─ select_roi_per_frame_ranges()   crop to a spatial ROI (interactive / reused)
 │   │
 │   ├─ Step 1 — Mother detection            (mother_detection.py)
 │   │   ├─ MotherTuner.tune()               interactive slider GUI to pick
 │   │   │                                   Frangi/wavelet/threshold parameters
 │   │   └─ detect_mother_stack_parallelized()
 │   │       ├─ analyze_mother_frame_gpu_fast() / _cpu()
 │   │       │     wavelet or Sobel gradient → multi-scale Frangi vesselness
 │   │       │     → skeletonize → contour → shape/temporal scoring
 │   │       ├─ per-frame polar curvature profile (theta_grid / kappa_grid)
 │   │       └─ writes mother/mother_metrics.csv, overlay PNGs, curvature .npy
 │   │
 │   ├─ (optional) Deconvolution              (modules/deconvulation.py)
 │   │
 │   ├─ Step 2 — Daughter detection & local tracking (detect_and_track.py)
 │   │   ├─ process_daughters_parallel()
 │   │   │     per-frame StarDist / Cellpose instance segmentation, restricted
 │   │   │     to candidates near/inside the mother boundary
 │   │   ├─ tracking()
 │   │   │     trackpy linking across frames + KD-tree contact detection
 │   │   │     (mother↔daughter "MLS" and daughter↔daughter "MSS" contacts)
 │   │   └─ writes daughter_tracking_local.csv, segmented/tracked overlay PNGs
 │   │
 │   └─ compile_experiment_data()             (modules/results.py)
 │       writes <file>/mother.csv and <file>/daughter.csv
 │
 └─ 2. Combined summary
     reads every per-file mother.csv / daughter.csv back from disk and
     concatenates them into mother_all.csv / inner_daughter_all.csv
```

Execution mode (Parallel vs. Sequential) is chosen once per run:
- **Parallel** — spreads frames across CPU worker processes (faster, no live
  progress bar; GPU segmentation is disabled in this mode, see below).
- **Sequential** — single process, one frame at a time, with a live progress bar.

Mother detection prefers GPU (CuPy) automatically if available and falls back to
CPU (scikit-image `frangi`) otherwise. Daughter segmentation can use a GPU-backed
StarDist/Cellpose model, but only in Sequential mode — a live CUDA/TensorFlow
context cannot be shared across the multiple worker processes that Parallel mode
spawns, so `parallel=True` + `use_gpu=True` automatically falls back to
sequential GPU execution (this is logged to the console when it happens).

## Repository status

| Component | Status |
|---|---|
| `main.py` | ✅ included |
| `mother_detection.py` | ✅ included |
| `detect_and_track.py` | ✅ included |
| `modules/` (`read_czi_tiff`, `global_rect_roi`, `slicer`, `results`, `savenpzraw`, `deconvulation`, `pre_process`, `image_buildFFT`, `image_wavelet2d_gauss`) | ⏳ to be added |
| `curvature_plot_metric.py` | ⏳ to be added |
| Plotting / downstream analysis scripts | ⏳ to be added |
| Example / test data | ⏳ to be added |

The three scripts here are not independently runnable yet because they import
from the `modules/` package and `curvature_plot_metric.py` above, which are not
in this commit. This is expected during incremental upload — once those are
added the pipeline can be run end-to-end as described below.

## Installation

The pipeline was built around a mix of CPU and (optional) GPU-accelerated
libraries. We recommend a dedicated conda/mamba environment, since two of the
GPU dependencies (`tensorflow`, `cupy`) are much easier to get working via
conda-forge than pip on most systems.

```bash
conda create -n vesicle-pipeline python=3.10
conda activate vesicle-pipeline
pip install -r requirements.txt
```

See [`requirements.txt`](requirements.txt) for the full dependency list and
notes on optional GPU packages (`cupy`, `tensorflow`-GPU, `torch`-CUDA,
`cellpose`, `stardist`). The pipeline runs on CPU only if none of the GPU
extras are installed — it detects this automatically and prints which mode it
is using.

## Usage

```bash
python main.py
```

The script is interactive (via `easygui` dialog boxes and, for mother-detection
parameter tuning, a Matplotlib slider GUI) and will prompt you to:

1. Choose the folder containing the microscopy stacks to process.
2. Optionally reuse a previously-defined ROI size.
3. Choose **Parallel** or **Sequential** execution mode for the whole run.
4. For the first file, interactively tune the mother-detection parameters
   (Gaussian sigma, wavelet scale, threshold) using the slider GUI, and select
   the frame range / spatial ROI.

Each input file `foo.czi` produces `results_v2/foo/` containing:

```
results_v2/foo/
├── mother.csv                    per-frame mother-boundary geometry (final)
├── daughter.csv                  per-frame daughter geometry + contacts (final)
├── detections.npz                (only if SAVE_RAW_NPZ = True)
├── mother/
│   ├── mother_metrics.csv
│   ├── raw/, overlay/            per-frame diagnostic images
│   └── theta_grid_radians.npy, kappa_normalized_vs_theta.npy
└── (segmentation/tracking working folders, overlay PNGs, etc.)
```

After all files in the folder are processed, you'll be prompted for a name for
the combined-results folder, which will contain:

```
results_v2/<combined_results>/
├── mother_all.csv            mother.csv for every file, with a `video` column
└── inner_daughter_all.csv    daughter.csv for every file, with a `video` column
```

### Configuration flags (top of `main.py`)

| Flag | Default | Effect |
|---|---|---|
| `SETTING_DECONVULATION` | `False` | Run Richardson-Lucy deconvolution on frames before daughter detection. |
| `SAVE_RAW_NPZ` | `False` | Also cache the raw per-frame image/mask stack to `detections.npz`. This is the single largest artifact per file (memory and disk); leave off unless you specifically need it for offline debugging. |

## Outputs — column reference

**`mother.csv` / `mother_metrics.csv`**: `frame`, `area_um2`, `perimeter_um`,
`radius_um`, `centroid_x`, `centroid_y`, `score` (detection confidence),
`perimeter_px`, `area_px`, `circularity`.

**`daughter.csv` / `daughter_tracking_local.csv`**: `id` (track id), `frame`,
`area`, `centroid_x`, `centroid_y`, `daughter_contour`, `contact_class`
(`MLS` = touching the mother, `MSS` = touching another daughter, `FREE` =
touching neither), `contact_point_x/y`, `interacting_with_id` (for `MSS` rows).

## Hardware notes

- Mother detection: GPU path requires a working CUDA toolkit + `cupy` matching
  your CUDA version. The script auto-detects `CUDA_PATH` on Windows and probes
  that a CuPy kernel actually compiles (not just that a device is visible)
  before enabling GPU mode, and prints a clear reason if it falls back to CPU.
- Daughter segmentation: StarDist (TensorFlow) and/or Cellpose (PyTorch) —
  whichever is installed and set as `SEGMENTATION_BACKEND` in
  `detect_and_track.py` (default: `"stardist"`).
- Multiprocessing (`Parallel` mode) is capped to a small worker count by
  default to avoid OpenBLAS/BLAS thread-oversubscription crashes on shared
  workstations; adjust `n_workers` in the relevant function calls if you have
  a dedicated machine with more cores.

## Citation

If you use this code, please cite the associated paper:

> [Author list]. *[Paper title]*. [Journal], [Year]. DOI: [pending]

See [`CITATION.cff`](CITATION.cff).

## Contact

- ms22113@iisermohali.ac.in
- itstanmaypandey@gmail.com
- caritra@iitg.ac.in

## License

Released under the [MIT License](LICENSE) unless your journal/funder requires
otherwise — see that file for details, and update it before publishing if you
need a different license.
