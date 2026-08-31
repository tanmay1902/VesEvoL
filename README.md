# VesEvoL — Vesicle Evolution over Long time

**VesEvoL** is an image-analysis framework developed to study the evolution of vesicle shape parameters over extended time periods using microscopy image sequences.

The software accompanies the paper:

> **Tanmay Pandey, Rajni Kudawlaa, Jeyanth Varshan B S, Chetana Devarakonda, Aritra Chatterjee, and Tripta Bhatia.**  
> *Evolution of Vesicle Shape Parameters Based on A Deep-Learning Based Image Analysis Framework Over Extended Time Periods.*  
> [Journal], 2026. DOI: [pending]

---

## Overview

VesEvoL (**Vesicle Evolution over Long time**) provides an image-analysis workflow for processing time-lapse microscopy data of vesicles.

The pipeline is designed to:

- Detect the large **mother vesicle** boundary.
- Detect smaller **daughter vesicles** around/inside the mother vesicle.
- Track daughter vesicles over consecutive frames.
- Quantify vesicle geometry and shape parameters.
- Identify mother–daughter and daughter–daughter contacts.
- Calculate curvature-related parameters.
- Analyze vesicle area evolution over time.
- Generate polar plots and karyographs.
- Validate segmentation performance.
- Produce downstream figures for quantitative analysis.

The repository contains both the **core image-analysis pipeline** and the **downstream plotting/analysis scripts** used for processing the resulting datasets.

---

## Repository structure

```text
VesEvoL/
│
├── main.py
├── mother_detection.py
├── detect_and_track.py
├── curvature_plot_metric.py       # required but missing from supplied archive
│
├── modules/
│   ├── read_czi_tiff.py
│   ├── global_rect_roi.py
│   ├── slicer.py
│   ├── deconvulation.py
│   ├── results.py
│   ├── savenpzraw.py
│   ├── pre_process.py
│   ├── image_buildFFT.py
│   ├── image_wavelet2d_gauss.py
│   └── image_edgedet.py
│
├── plot_curvature/
│   ├── divergence_MSS_MLS_Binned.py
│   ├── heatmap.py
│   ├── plot_curvature_population.py
│   ├── plot_histogram.py
│   └── utils.py
│
├── plot_normalized_area/
│   ├── plot_binned_area.py
│   ├── plot_single_exp.py
│   └── utils.py
│
├── plot_polar/
│   ├── Figure-preparation.py
│   ├── PolarPlot.py
│   ├── mystyle.py
│   └── mystyle.mplstyle
│
├── plot_segmentation/
│   ├── pipeline_validation_from_csv.py
│   ├── pipeline_validation_data.csv
│   └── mystyle.py
│
├── requirements.txt
├── CITATION.cff
├── LICENSE
├── index.html
├── README.md
│
└── docs/
    └── VesEvoL_Documentation.docx
```

---

## Repository status

| Component | Status |
|---|---|
| `main.py` | ✅ Included |
| `mother_detection.py` | ✅ Included |
| `detect_and_track.py` | ✅ Included |
| `modules/` helper package | ✅ Included |
| `plot_curvature/` | ✅ Included |
| `plot_normalized_area/` | ✅ Included |
| `plot_polar/` | ✅ Included |
| `plot_segmentation/` | ✅ Included |
| `requirements.txt` | ✅ Included |
| `CITATION.cff` | ✅ Included |
| `LICENSE` | ✅ Included |
| `index.html` | ✅ Included |
| DOCX documentation | ✅ Included |
| `curvature_plot_metric.py` | ⚠️ Missing from supplied archive |
| Example microscopy stacks | Not included |

### Important

`mother_detection.py` imports:

```python
from curvature_plot_metric import compute_curvature
```

However, `curvature_plot_metric.py` was **not present in the supplied `VF.zip`**.

Therefore, the repository as supplied is not completely self-contained/runnable until this file is added or the corresponding functionality is restored.

---

# Installation

A dedicated Conda environment is recommended because the pipeline uses a combination of scientific Python, image-processing, tracking, and deep-learning libraries.

## Create the environment

```bash
conda create -n vesevol python=3.10
conda activate vesevol
```

## Install dependencies

```bash
pip install -r requirements.txt
```

The pipeline uses packages including:

- NumPy
- Pandas
- SciPy
- scikit-image
- OpenCV
- Matplotlib
- EasyGUI
- Trackpy
- alive-progress
- TensorFlow
- StarDist
- CSBDeep
- PyTorch
- Cellpose
- CuPy

Not every dependency is necessarily required for every execution path.

---

# Core pipeline

The main workflow is controlled through:

```text
main.py
```

Run it with:

```bash
python main.py
```

The pipeline interactively guides the user through the analysis.

The general workflow is:

```text
Microscopy stack
       │
       ▼
Load image stack
       │
       ▼
Select frame range
       │
       ▼
Select spatial ROI
       │
       ▼
Mother vesicle detection
       │
       ▼
Mother geometry + curvature
       │
       ▼
Optional deconvolution
       │
       ▼
Daughter vesicle segmentation
       │
       ▼
Daughter filtering
       │
       ▼
Daughter tracking
       │
       ▼
Contact classification
       │
       ▼
Per-file CSV results
       │
       ▼
Combined datasets
       │
       ▼
Downstream analysis
       │
       ├── Curvature
       ├── Normalized area
       ├── Polar/karyograph
       └── Segmentation validation
```

---

# 1. Mother vesicle detection

`mother_detection.py` detects the large mother vesicle boundary in each frame.

The workflow includes:

1. Image normalization.
2. Gaussian filtering.
3. Gradient/wavelet processing.
4. Multi-scale Frangi vesselness filtering.
5. Thresholding.
6. Skeletonization.
7. Contour extraction.
8. Candidate contour scoring.
9. Temporal consistency scoring.
10. Mother-vesicle geometry calculation.
11. Polar curvature calculation.

The mother contour is selected using a combination of:

- Circularity.
- Ellipse similarity.
- Area.
- Aspect ratio.
- Centroid displacement.
- Radius change between consecutive frames.

The pipeline can use a GPU-accelerated CuPy implementation when a suitable CUDA environment is available.

If GPU support is unavailable, the code falls back to CPU processing.

---

# 2. Daughter vesicle detection

`detect_and_track.py` performs daughter-vesicle detection.

Two segmentation backends are supported by the supplied code:

### StarDist

```text
StarDist2D
2D_versatile_fluo
```

### Cellpose

```text
CellposeModel
cpsam
```

Detected objects are filtered according to their:

- Position relative to the mother vesicle.
- Area.
- Circularity.

The supplied workflow restricts candidate daughters to regions near or inside the mother-vesicle boundary.

---

# 3. Daughter tracking

After detection, daughter vesicles can be linked between frames using:

```text
trackpy.link_df
```

The supplied tracking configuration uses:

```text
search_range = 10
memory       = 5
```

Short tracks can subsequently be removed using trackpy's filtering functionality.

Each tracked daughter receives a track ID.

---

# 4. Contact classification

The pipeline identifies interactions between vesicles.

Two principal contact classes are used:

| Class | Meaning |
|---|---|
| `MLS` | Mother–daughter contact |
| `MSS` | Daughter–daughter contact |
| `FREE` | Daughter touching neither |

Nearest-point calculations use KD-tree based distance queries.

For mother contacts, the analysis also records the contact location and local curvature information.

For daughter–daughter contacts, the interacting daughter track ID is recorded.

---

# 5. Output files

For an input file such as:

```text
vesicle_001.czi
```

the pipeline creates a result directory similar to:

```text
results_v2/
└── vesicle_001/
    ├── mother.csv
    ├── daughter.csv
    ├── detections.npz
    │
    └── mother/
        ├── mother_metrics.csv
        ├── raw/
        ├── overlay/
        ├── theta_grid_radians.npy
        └── kappa_normalized_vs_theta.npy
```

`detections.npz` is generated only when:

```python
SAVE_RAW_NPZ = True
```

---

# Combined results

After processing multiple input files, the pipeline combines the per-file results.

The combined directory contains:

```text
results_v2/
└── <combined_results>/
    ├── mother_all.csv
    └── inner_daughter_all.csv
```

The combined datasets include a `video` column identifying the source experiment/file.

---

# Output columns

## Mother data

Important columns include:

| Column | Description |
|---|---|
| `frame` | Frame index |
| `area_um2` | Mother area in µm² |
| `area_px` | Mother area in pixels |
| `perimeter_um` | Perimeter in µm |
| `perimeter_px` | Perimeter in pixels |
| `radius_um` | Equivalent radius |
| `centroid_x` | X-coordinate of centroid |
| `centroid_y` | Y-coordinate of centroid |
| `score` | Detection confidence score |
| `circularity` | Circularity metric |

Circularity is calculated as:

```text
4πA / P²
```

and capped at 1.0.

---

## Daughter data

Important columns include:

| Column | Description |
|---|---|
| `id` | Daughter track ID |
| `frame` | Frame index |
| `area` | Daughter area |
| `centroid_x` | X-coordinate |
| `centroid_y` | Y-coordinate |
| `daughter_contour` | Daughter boundary contour |
| `contact_class` | MLS, MSS, or FREE |
| `contact_point_x` | Contact-point X-coordinate |
| `contact_point_y` | Contact-point Y-coordinate |
| `interacting_with_id` | Other daughter ID for MSS events |

---

# Downstream analysis

The repository includes several dedicated analysis directories.

---

## `plot_curvature/`

This directory contains scripts for curvature-based analysis.

### `plot_curvature_population.py`

Used for population-level curvature analysis and figure generation.

### `heatmap.py`

Generates curvature-related heatmaps.

### `plot_histogram.py`

Generates histogram-based curvature analysis, including the fitted distributions implemented in the script.

### `divergence_MSS_MLS_Binned.py`

Performs binned comparison/divergence analysis between:

```text
MSS
```

and

```text
MLS
```

events.

### `utils.py`

Contains utilities for:

- Reading experiment/result directories.
- Loading trajectories.
- Processing vesicle trajectories.
- Time binning.

---

# `plot_normalized_area/`

This directory contains analysis scripts for vesicle area evolution.

### `plot_binned_area.py`

Collects and processes normalized-area measurements and generates binned figures.

### `plot_single_exp.py`

Performs:

- Vesicle-level normalization.
- Time-series processing.
- Single-exponential fitting.
- Parameter extraction.
- Parameter visualization.
- Vesicle-level scatter plots.

### `utils.py`

Provides utilities for:

- CSV loading.
- Vesicle normalization.
- Time binning.
- Condition aggregation.

---

# `plot_polar/`

This directory contains polar and karyograph-related analysis.

### `PolarPlot.py`

Performs geometry calculations and generates:

- Polar plots.
- Karyographs.
- Combined karyographs.
- Contact-curvature plots.
- Contact-angle analysis.

### `Figure-preparation.py`

Provides functionality for preparing composite figures from experiment/result directories.

### Plot style

```text
mystyle.py
mystyle.mplstyle
```

contain plotting-style definitions used by the analysis scripts.

---

# `plot_segmentation/`

This directory contains segmentation-validation code.

### `pipeline_validation_from_csv.py`

Reads segmentation validation measurements from CSV and generates comparative analysis/plots.

### `pipeline_validation_data.csv`

Contains the supplied validation dataset used by the validation script.

---

# Configuration

Several parameters are configured directly in the Python scripts.

Important options include:

### Deconvolution

```python
SETTING_DECONVULATION = False
```

When enabled, Richardson–Lucy deconvolution is applied before daughter analysis.

### Raw data saving

```python
SAVE_RAW_NPZ = False
```

When enabled, raw image/mask information is stored in:

```text
detections.npz
```

Because this can substantially increase disk usage, it is disabled by default.

### Segmentation backend

The supplied daughter-processing code uses:

```python
SEGMENTATION_BACKEND = "stardist"
```

The exact backend configuration should be checked in `detect_and_track.py` before running an analysis.

---

# Parallel vs sequential execution

The core pipeline supports parallel and sequential execution.

### Parallel mode

Parallel processing distributes frames across worker processes.

Advantages:

- Faster CPU processing.
- Useful for large datasets.

Limitations:

- GPU segmentation cannot be shared between worker processes.
- No live progress bar in the same way as sequential execution.

### Sequential mode

Sequential execution processes frames in the main process.

Advantages:

- Supports GPU-based segmentation.
- Provides live progress information.
- Maintains temporal state directly between frames.

---

# GPU support

The mother-detection code contains a CuPy-based GPU implementation.

GPU operation requires a compatible:

- NVIDIA GPU
- CUDA installation
- CuPy build

The code checks whether the CUDA environment is actually usable rather than relying only on device visibility.

If GPU initialization fails, CPU processing can be used instead.

---

# Input data

The supplied code is designed to work with microscopy image stacks, including:

```text
CZI
TIFF
```

The repository does **not** contain example microscopy stacks.

Users should therefore provide their own compatible experimental data.

---

# Important path configuration for plotting

Some downstream plotting scripts contain machine-specific data paths.

Before running these scripts, inspect variables such as:

```python
DATA_ROOT
```

and update them to the location of the processed results on your system.

For example:

```python
DATA_ROOT = r"/path/to/results"
```

The exact path variable differs between scripts.

---

# Documentation

A more detailed documentation file is included at:

```text
docs/VesEvoL_Documentation.docx
```

An interactive HTML version of the documentation is also provided:

```text
index.html
```

The HTML documentation describes:

- Installation.
- Pipeline stages.
- Configuration.
- Core functions.
- Output datasets.
- Plotting modules.
- Hardware considerations.
- Citation and release information.

---

# Citation

If you use VesEvoL in your work, please cite the associated paper:

> **Pandey, T.; Kudawlaa, R.; Varshan B S, J.; Devarakonda, C.; Chatterjee, A.; Bhatia, T.**  
> *Evolution of Vesicle Shape Parameters Based on A Deep-Learning Based Image Analysis Framework Over Extended Time Periods.*  
> [Journal], 2026. DOI: [pending]

A machine-readable citation is provided in:

```text
CITATION.cff
```

The repository URL, journal, paper DOI, and Zenodo DOI should be filled in once the corresponding identifiers are available.

---

# License

VesEvoL is distributed under the:

**MIT License**

See:

```text
LICENSE
```

for the complete license text.

---

# Repository release notes

The supplied repository has been audited for the major source components.

### Included

- Core detection/tracking pipeline.
- Supporting modules.
- Curvature analysis.
- Normalized-area analysis.
- Polar/karyograph analysis.
- Segmentation validation.
- Citation metadata.
- License.
- HTML documentation.
- DOCX documentation.

### Remaining action

The file:

```text
curvature_plot_metric.py
```

is imported by `mother_detection.py` but was not present in the supplied `VF.zip`.

This file should be added before claiming that the repository is completely self-contained and runnable.

The downstream plotting scripts should also be checked for machine-specific data paths before distribution.

---

## Authors

**Tanmay Pandey**  
**Rajni Kudawlaa**  
**Jeyanth Varshan B S**  
**Chetana Devarakonda**  
**Aritra Chatterjee**  
**Tripta Bhatia**

---

## Project

**VesEvoL — Vesicle Evolution over Long time**

Developed as supplementary software for:

> *Evolution of Vesicle Shape Parameters Based on A Deep-Learning Based Image Analysis Framework Over Extended Time Periods*