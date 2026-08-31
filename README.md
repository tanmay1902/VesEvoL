# VesEvoL — Vesicle Evolution over Long time

**VesEvoL** (**Vesicle Evolution over Long time**) is an image-analysis framework for quantitative analysis of vesicle morphology and interactions over extended time periods using time-lapse microscopy data.

The software accompanies the manuscript:

> **Tanmay Pandey, Rajni Kudawlaa, Jeyanth Varshan B S, Chetana Devarakonda, Aritra Chatterjee, and Tripta Bhatia.**
> *Evolution of Vesicle Shape Parameters Based on A Deep-Learning Based Image Analysis Framework Over Extended Time Periods.*
> [Journal], 2026. DOI: [pending]

---

## Overview

VesEvoL provides an end-to-end workflow for analysing time-lapse microscopy datasets containing large mother vesicles and smaller daughter vesicles.

The framework performs:

* Mother-vesicle boundary detection.
* Temporal tracking of the mother vesicle.
* Mother-vesicle shape and geometric analysis.
* Curvature calculation and polar projection.
* Daughter-vesicle segmentation.
* Daughter-vesicle tracking across frames.
* Mother–daughter and daughter–daughter contact detection.
* Contact classification as `MLS`, `MSS`, or `FREE`.
* Vesicle area and shape evolution analysis.
* Curvature population analysis.
* Normalized-area analysis.
* Polar plots and karyographs.
* Segmentation-pipeline validation.
* Generation of downstream analysis figures.

The repository contains the **core image-analysis pipeline**, supporting modules, downstream analysis scripts, validation data, documentation, and release metadata.

---

# Repository structure

```text
VesEvoL/
│
├── main.py
├── mother_detection.py
├── detect_and_track.py
├── curvature_plot_metric.py
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
│   ├── plot_curvature_population.py
│   ├── heatmap.py
│   ├── plot_histogram.py
│   ├── divergence_MSS_MLS_Binned.py
│   ├── utils.py
│   ├── mystyle.py
│   └── mystyle.mplstyle
│
├── plot_normalized_area/
│   ├── plot_binned_area.py
│   ├── plot_single_exp.py
│   ├── utils.py
│   ├── mystyle.py
│   └── mystyle.mplstyle
│
├── plot_polar/
│   ├── PolarPlot.py
│   ├── Figure-preparation.py
│   ├── mystyle.py
│   └── mystyle.mplstyle
│
├── plot_segmentation/
│   ├── pipeline_validation_from_csv.py
│   ├── pipeline_validation_data.csv
│   ├── mystyle.py
│   └── mystyle.mplstyle
│
├── requirements.txt
├── CITATION.cff
├── LICENSE
├── README.md
├── index.html
│
└── docs/
    └── VesEvoL_Documentation.docx
```

---

# Repository status

The repository contains the complete software components required by the supplied codebase.

| Component                         | Status       |
| --------------------------------- | ------------ |
| `main.py`                         | ✅ Included   |
| `mother_detection.py`             | ✅ Included   |
| `detect_and_track.py`             | ✅ Included   |
| `curvature_plot_metric.py`        | ✅ Included   |
| `modules/`                        | ✅ Included   |
| `plot_curvature/`                 | ✅ Included   |
| `plot_normalized_area/`           | ✅ Included   |
| `plot_polar/`                     | ✅ Included   |
| `plot_segmentation/`              | ✅ Included   |
| `requirements.txt`                | ✅ Included   |
| `CITATION.cff`                    | ✅ Included   |
| `LICENSE`                         | ✅ Included   |
| `README.md`                       | ✅ Included   |
| `index.html`                      | ✅ Included   |
| `docs/VesEvoL_Documentation.docx` | ✅ Included   |
| Example microscopy datasets       | Not included |

Raw microscopy datasets are not distributed with the repository. Users should provide their own compatible CZI/TIFF time-lapse microscopy data.

---

# Quick Start

## 1. Create the environment

A dedicated Conda environment is recommended.

```bash
conda create -n vesevol python=3.10
conda activate vesevol
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Run the main pipeline

```bash
python main.py
```

The pipeline interactively guides the user through the processing workflow.

The main stages are:

1. Select the microscopy-data directory.
2. Select the frame range.
3. Select or reuse the spatial ROI.
4. Tune mother-vesicle detection parameters.
5. Select Parallel or Sequential execution.
6. Detect and analyse the mother vesicle.
7. Detect and segment daughter vesicles.
8. Track daughter vesicles.
9. Identify vesicle contacts.
10. Generate per-file results.
11. Generate combined datasets for downstream analysis.

---

# Analysis workflow

The overall VesEvoL workflow is:

```text
                 Microscopy stack
                        │
                        ▼
                Load image stack
                        │
                        ▼
              Frame range + ROI
                        │
                        ▼
             Mother-vesicle detection
                        │
                        ├── Geometry
                        ├── Shape parameters
                        └── Curvature
                        │
                        ▼
                Optional deconvolution
                        │
                        ▼
              Daughter segmentation
                        │
                        ├── StarDist
                        └── Cellpose
                        │
                        ▼
                 Daughter filtering
                        │
                        ▼
                  Track daughters
                        │
                        ▼
                Contact classification
                        │
                ┌───────┼────────┐
                ▼       ▼        ▼
               MLS     MSS      FREE
                │       │        │
                └───────┼────────┘
                        ▼
                 Result compilation
                        │
                        ▼
                 Combined datasets
                        │
                        ▼
              Downstream analysis
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
      Curvature    Normalized area   Polar/
      analysis        analysis      karyograph
                        │
                        ▼
                 Segmentation
                   validation
```

---

# Core pipeline

## `main.py`

`main.py` is the main orchestration script.

It coordinates:

* Input selection.
* Image-stack loading.
* Frame selection.
* ROI selection.
* Mother-vesicle analysis.
* Optional deconvolution.
* Daughter-vesicle detection.
* Daughter tracking.
* Result compilation.
* Combined-result generation.

The script is intended to be executed directly rather than imported as a library.

```bash
python main.py
```

---

# Mother-vesicle detection

## `mother_detection.py`

The mother-detection workflow identifies the large vesicle boundary in each frame.

The analysis includes:

1. Image normalization.
2. Gaussian filtering.
3. Gradient/wavelet processing.
4. Multi-scale Frangi vesselness.
5. Thresholding.
6. Skeletonization.
7. Contour extraction.
8. Candidate-contour scoring.
9. Temporal consistency analysis.
10. Shape and area measurements.
11. Curvature calculation.
12. Polar projection of curvature.

Candidate contours are evaluated using multiple properties, including:

* Circularity.
* Ellipse similarity.
* Aspect ratio.
* Area.
* Centroid displacement.
* Radius changes.
* Temporal consistency.

The selected mother contour is then used for subsequent daughter-vesicle analysis.

---

# Curvature calculation

## `curvature_plot_metric.py`

`curvature_plot_metric.py` provides the curvature calculations used by the mother-vesicle analysis.

The resulting curvature data can be projected onto a uniform angular coordinate system around the vesicle.

The pipeline can therefore produce:

```text
theta_grid_radians.npy
kappa_normalized_vs_theta.npy
```

These data are subsequently used by the curvature and polar-analysis scripts.

---

# Daughter-vesicle detection

## `detect_and_track.py`

Daughter vesicles are detected using deep-learning-based instance segmentation.

The supplied code supports:

### StarDist

```text
2D_versatile_fluo
```

### Cellpose

```text
cpsam
```

Detected objects are filtered according to their:

* Location relative to the mother vesicle.
* Area.
* Circularity.
* Spatial relationship with the mother boundary.

Only suitable daughter-vesicle candidates are retained for subsequent tracking.

---

# Daughter-vesicle tracking

Daughter detections are linked across consecutive frames using `trackpy`.

The supplied tracking workflow uses:

```text
search_range = 10
memory = 5
```

Short tracks can be filtered before the final results are generated.

Each daughter trajectory is assigned a track ID.

---

# Contact classification

Vesicle interactions are classified into three categories.

| Class  | Description                                      |
| ------ | ------------------------------------------------ |
| `MLS`  | Mother–daughter contact                          |
| `MSS`  | Daughter–daughter contact                        |
| `FREE` | Daughter not contacting another analysed vesicle |

Nearest-point calculations are performed using KD-tree-based distance measurements.

For `MLS` contacts, the analysis can additionally record:

* Contact position.
* Local mother-vesicle curvature.
* Local contour information.

For `MSS` contacts, the interacting daughter track ID is recorded.

---

# Output structure

For an input microscopy file such as:

```text
vesicle_001.czi
```

the pipeline produces a result directory similar to:

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

The `detections.npz` file is optional and is generated when:

```python
SAVE_RAW_NPZ = True
```

---

# Combined datasets

After processing all files in an input directory, VesEvoL generates combined datasets.

```text
results_v2/
└── <combined_results>/
    ├── mother_all.csv
    └── inner_daughter_all.csv
```

### `mother_all.csv`

Contains the mother-vesicle measurements from all processed files, together with a `video` column identifying the source file.

### `inner_daughter_all.csv`

Contains daughter-vesicle measurements from all processed files, together with a `video` column.

---

# Output columns

## Mother data

Important columns include:

| Column         | Description                         |
| -------------- | ----------------------------------- |
| `frame`        | Frame index                         |
| `area_um2`     | Mother area in µm²                  |
| `area_px`      | Mother area in pixels               |
| `perimeter_um` | Perimeter in µm                     |
| `perimeter_px` | Perimeter in pixels                 |
| `radius_um`    | Equivalent radius                   |
| `centroid_x`   | X-coordinate of the centroid        |
| `centroid_y`   | Y-coordinate of the centroid        |
| `score`        | Combined detection-confidence score |
| `circularity`  | Isoperimetric circularity           |

Circularity is calculated as:

```text
4πA / P²
```

with the resulting value capped at 1.0.

---

## Daughter data

Important columns include:

| Column                | Description                            |
| --------------------- | -------------------------------------- |
| `id`                  | Daughter track ID                      |
| `frame`               | Frame index                            |
| `area`                | Daughter area                          |
| `centroid_x`          | X-coordinate                           |
| `centroid_y`          | Y-coordinate                           |
| `daughter_contour`    | Daughter boundary contour              |
| `contact_class`       | `MLS`, `MSS`, or `FREE`                |
| `contact_point_x`     | Contact-point X-coordinate             |
| `contact_point_y`     | Contact-point Y-coordinate             |
| `interacting_with_id` | Other daughter track ID for MSS events |

---

# Downstream analysis

VesEvoL contains four major downstream analysis areas.

---

## `plot_curvature/`

This directory contains scripts for curvature-based analysis.

### `plot_curvature_population.py`

Used for population-level curvature analysis and figure generation.

### `heatmap.py`

Generates heatmaps of curvature-related measurements.

### `plot_histogram.py`

Generates histogram-based curvature analysis and distribution fits.

### `divergence_MSS_MLS_Binned.py`

Performs binned comparison/divergence analysis between MSS and MLS measurements.

### `utils.py`

Provides utilities for:

* Reading result directories.
* Loading trajectories.
* Processing trajectory data.
* Time binning.

---

## `plot_normalized_area/`

This directory contains scripts for analysing vesicle-area evolution.

### `plot_binned_area.py`

Processes normalized-area measurements and generates binned area-evolution figures.

### `plot_single_exp.py`

Provides analysis including:

* Vesicle-level normalization.
* Time-series processing.
* Single-exponential fitting.
* Parameter extraction.
* Parameter visualization.
* Vesicle-level scatter plots.

### `utils.py`

Provides utilities for:

* CSV loading.
* Vesicle normalization.
* Time binning.
* Condition aggregation.

---

## `plot_polar/`

This directory contains polar and karyograph analysis.

### `PolarPlot.py`

Performs geometric analysis and generates:

* Polar plots.
* Karyographs.
* Combined karyographs.
* Contact-curvature figures.
* Contact-angle analysis.

### `Figure-preparation.py`

Provides tools for preparing composite figures from processed experimental results.

### Plot styles

```text
mystyle.py
mystyle.mplstyle
```

provide plotting-style definitions used by the polar-analysis scripts.

---

## `plot_segmentation/`

This directory contains segmentation-validation code.

### `pipeline_validation_from_csv.py`

Reads segmentation-validation measurements and generates comparative analysis.

### `pipeline_validation_data.csv`

Contains the supplied validation dataset.

---

# Parallel and sequential processing

VesEvoL supports both parallel and sequential execution.

## Parallel mode

Parallel processing distributes work between CPU worker processes.

Advantages:

* Faster CPU processing.
* Useful for processing large image sequences.

Because CUDA/TensorFlow contexts cannot safely be shared between multiprocessing workers, GPU-based segmentation is not used in the parallel workflow.

---

## Sequential mode

Sequential processing handles frames one at a time in the main process.

Advantages:

* Supports GPU-based segmentation.
* Provides live progress information.
* Allows temporal information to be passed directly between frames.

---

# GPU acceleration

Mother-vesicle detection includes an optional CuPy/CUDA implementation.

GPU acceleration requires a compatible:

* NVIDIA GPU.
* CUDA installation.
* CuPy installation.

The code checks CUDA availability and attempts to verify that the GPU processing environment is actually functional.

If GPU processing is unavailable, the mother-detection workflow can use the CPU implementation.

---

# Configuration

Several parameters can be configured in the Python scripts.

## Richardson–Lucy deconvolution

```python
SETTING_DECONVULATION = False
```

When enabled, Richardson–Lucy deconvolution is applied before daughter-vesicle detection.

---

## Raw-data caching

```python
SAVE_RAW_NPZ = False
```

When enabled, raw frame and mask information is stored in:

```text
detections.npz
```

This can substantially increase storage requirements and is therefore disabled by default.

---

## Segmentation backend

The daughter-processing code supports StarDist and Cellpose.

The backend configuration should be checked in:

```text
detect_and_track.py
```

before running a new analysis.

---

# Input data

The pipeline is designed to process microscopy image stacks, including:

```text
CZI
TIFF
```

The repository does not contain raw experimental microscopy datasets.

Users should provide their own microscopy data in a format supported by the supplied image-loading functions.

---

# Plotting path configuration

Some downstream analysis scripts contain local or machine-specific data paths.

Before running the plotting scripts, inspect variables such as:

```python
DATA_ROOT
```

and update them to the location of the processed experimental results.

For example:

```python
DATA_ROOT = r"/path/to/results"
```

The exact path configuration can differ between scripts.

---

# Documentation

Additional documentation is provided in two formats.

### Interactive HTML documentation

```text
index.html
```

The HTML documentation provides an interactive overview of:

* Pipeline stages.
* Installation.
* Configuration.
* Function references.
* Output files.
* Plotting modules.
* Hardware requirements.
* Citation information.

### DOCX documentation

```text
docs/VesEvoL_Documentation.docx
```

The DOCX provides a printable/document-style description of the repository and analysis workflow.

---

# Citation

If you use VesEvoL in your research, please cite the associated paper:

> **Pandey, T.; Kudawlaa, R.; Varshan B S, J.; Devarakonda, C.; Chatterjee, A.; Bhatia, T.**
> *Evolution of Vesicle Shape Parameters Based on A Deep-Learning Based Image Analysis Framework Over Extended Time Periods.*
> [Journal], 2026. DOI: [pending]

A machine-readable citation is provided in:

```text
CITATION.cff
```

The repository URL, journal information, paper DOI, and Zenodo DOI should be updated once the corresponding identifiers are available.

---

# License

VesEvoL is distributed under the **MIT License**.

See:

```text
LICENSE
```

for the complete license text.

---

# Reproducibility

For reproducible analysis:

1. Use the supplied `requirements.txt`.
2. Use a dedicated Python environment.
3. Preserve the original microscopy data and acquisition metadata.
4. Record the frame range and ROI used for each experiment.
5. Record the mother-detection parameters selected during interactive tuning.
6. Record the segmentation backend used for daughter detection.
7. Preserve the generated CSV and NumPy output files.
8. Update the downstream plotting scripts' data paths before generating figures.
9. Retain the VesEvoL version used for the analysis.

---

# Release information

**Software:** VesEvoL
**Full name:** Vesicle Evolution over Long time
**Version:** 0.1.0
**Year:** 2026

VesEvoL is provided as supplementary software accompanying:

> *Evolution of Vesicle Shape Parameters Based on A Deep-Learning Based Image Analysis Framework Over Extended Time Periods*

The repository contains the core analysis pipeline and downstream analysis scripts required to process the resulting vesicle measurements.

---

# Authors

**Tanmay Pandey**
**Rajni Kudawlaa**
**Jeyanth Varshan B S**
**Chetana Devarakonda**
**Aritra Chatterjee**
**Tripta Bhatia**

---

## VesEvoL

**Vesicle Evolution over Long time**

Supplementary software for quantitative analysis of vesicle evolution over extended time periods.
