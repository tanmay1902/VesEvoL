"""
Mother Cell/Vesicle Segmentation and Shape Tracking Pipeline.

Contact Details:
Email(s): itstanmaypandey@gmail.com , caritra@iitg.ac.in
"""

from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, cpu_count
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

from alive_progress import alive_bar
import cv2
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.widgets import PolygonSelector, Slider
import numpy as np
from skimage.filters import frangi
from skimage.morphology import skeletonize

# --- Auto-detect CUDA_PATH before importing cupy -----------------------------------
# cupy's NVRTC JIT step needs CUDA_PATH to find the toolkit (headers etc.). On Windows
# this env var is very easy to lose - it's set by the CUDA Toolkit installer, but a new
# conda env, a reinstall, or a driver update can leave it stale or unset even though the
# toolkit itself is still on disk. If it's already set we leave it alone; otherwise we
# scan the default NVIDIA install location and, if found, set it for this process only
# (does not touch your permanent system/user environment variables).
def _autodetect_cuda_path() -> Optional[str]:
    if os.environ.get("CUDA_PATH"):
        return os.environ["CUDA_PATH"]

    if sys.platform == "win32":
        base = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA")
    else:
        base = Path("/usr/local")

    if not base.exists():
        return None

    # Prefer the highest installed version (v12.4 > v11.7 > ...) if several are present.
    candidates = sorted(
        [d for d in base.glob("v*") if d.is_dir()] if sys.platform == "win32"
        else [d for d in base.glob("cuda-*") if d.is_dir()],
        reverse=True
    )
    if not candidates:
        return None

    chosen = str(candidates[0])
    os.environ["CUDA_PATH"] = chosen
    bin_dir = str(candidates[0] / "bin")
    if bin_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    return chosen


# --- Also register any pip-installed `nvidia-*-cu12` package DLL folders -----------
# Unlike Linux wheels, Windows doesn't put a package's own DLL folder on the system
# DLL search path just because it's importable - `pip install nvidia-cuda-nvrtc-cu12`
# etc. lands the actual .dll files under site-packages\nvidia\<component>\bin, but
# nothing tells Windows (or cupy) to look there unless we add it explicitly via
# os.add_dll_directory(). This is the actual reason `pip install nvidia-cuda-nvrtc-cu12`
# alone doesn't fix the "nvrtc64_120_0.dll not found" error on Windows even though the
# package installed successfully - the file exists on disk, it's just not discoverable.
def _register_pip_nvidia_dll_dirs() -> List[str]:
    registered = []
    try:
        import importlib.util
        spec = importlib.util.find_spec("nvidia")
        if spec is None or not spec.submodule_search_locations:
            return registered
        nvidia_root = Path(list(spec.submodule_search_locations)[0])
    except Exception:
        return registered

    subdir_name = "bin" if sys.platform == "win32" else "lib"
    for component_dir in sorted(nvidia_root.glob(f"*/{subdir_name}")):
        if not component_dir.is_dir():
            continue
        has_dll = any(component_dir.glob("*.dll")) or any(component_dir.glob("*.so*"))
        if not has_dll:
            continue
        d = str(component_dir)
        if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(d)
            except (OSError, FileNotFoundError):
                pass
        if d not in os.environ.get("PATH", ""):
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
        registered.append(d)
    return registered


_detected_cuda_path = _autodetect_cuda_path()
if _detected_cuda_path:
    print(f"ℹ CUDA_PATH = {_detected_cuda_path}")
else:
    print("⚠ Could not auto-detect a CUDA Toolkit install (CUDA_PATH not set and no "
          "default NVIDIA install folder found). If you have the toolkit installed "
          "somewhere non-standard, set CUDA_PATH yourself before running this script.")

_pip_nvidia_dirs = _register_pip_nvidia_dll_dirs()
if _pip_nvidia_dirs:
    print(f"ℹ Registered {len(_pip_nvidia_dirs)} pip-installed nvidia-*-cu12 DLL folder(s) "
          f"for loading: {', '.join(Path(d).parent.name for d in _pip_nvidia_dirs)}")
else:
    print("ℹ No pip-installed `nvidia-*-cu12` package DLL folders found "
          "(only relevant if you installed those instead of / in addition to a full CUDA Toolkit).")

# GPU is the default execution path for mother detection. We verify not just that
# cupy imports and a device is visible, but that a kernel can actually be JIT-compiled
# (via NVRTC) and run - device visibility alone doesn't catch a broken/missing
# CUDA_PATH, which only surfaces the first time a real kernel (e.g. gaussian_filter)
# is compiled. Without this probe that failure happens deep inside MotherTuner or a
# worker process and crashes the whole run instead of falling back to CPU.
try:
    import cupy as cp
    import cupyx.scipy.ndimage as ndimg
    _n_devices = cp.cuda.runtime.getDeviceCount()
    if _n_devices > 0:
        _probe = cp.asarray(np.zeros((4, 4), dtype=np.float32))
        ndimg.gaussian_filter(_probe, 1.0)  # forces an actual NVRTC kernel compile
        del _probe
        HAS_GPU = True
        print(f"✅ GPU mode active for mother detection ({_n_devices} CUDA device(s), kernel compile OK).")
    else:
        HAS_GPU = False
        print("⚠ cupy imported but no CUDA device is visible — falling back to CPU for mother detection. "
              "Check `nvidia-smi` and your CUDA driver/toolkit install.")
except ImportError:
    HAS_GPU = False
    print("⚠ cupy is not installed — falling back to CPU for mother detection. "
          "Install a cupy build matching your CUDA version to enable GPU mode.")
except Exception as e:
    HAS_GPU = False
    print(f"⚠ GPU kernel compile check failed ({e}) — falling back to CPU for mother detection. "
          "This is almost always a missing/mismatched CUDA_PATH environment variable "
          "rather than a missing GPU - see the GPU setup notes.")

# Try optional spatial utilities
try:
    from scipy.spatial.distance import cdist
except ImportError:
    cdist = None

# Custom Project Modules
from curvature_plot_metric import compute_curvature
from modules.pre_process import preprocess_phase_contrast

try:
    from modules.image_buildFFT import BuildFFT
    from modules.image_wavelet2d_gauss import Wavelet2D_gauss
except ImportError:
    BuildFFT, Wavelet2D_gauss = None, None


# ==============================================================================
# CONFIGURATION & PARAMETERS
# ==============================================================================

DEFAULT_MOTHER_PARAMS = {
    # Filtering & Scale Space
    "gauss_ksize": 7,
    "gauss_sigma": 3.0,
    "wavelet_scale": 1.2,
    "frangi_sigma_min": 1.0,
    "frangi_sigma_max": 15.0,
    "frangi_scale_step": 2.0,
    "threshold_value": 5,
    # Geometric Thresholds
    "min_area_um2": 50.0,
    "ellipse_aspect_ratio_max": 2.0,
    "small_mother_diam_um": 5.0,
    # Scoring Optimization Weights
    "circularity_weight": 0.2,
    "area_weight": 0.6,
    "temporal_weight": 0.2,
    # Tracking Constraints
    "max_centroid_shift_um": 20.0,
    "max_radius_change": 0.15,
}


# ==============================================================================
# MATHEMATICAL, MORPHOLOGICAL & POLAR MAPPING UTILITIES
# ==============================================================================

def normalize(img: np.ndarray) -> np.ndarray:
    """Normalize image intensities linearly to a [0.0, 1.0] range."""
    img_float = img.astype(np.float32)
    img_min, img_max = img_float.min(), img_float.max()
    if img_max > img_min:
        return (img_float - img_min) / (img_max - img_min)
    return img_float


def contour_to_mask(contour: Optional[np.ndarray], shape: Tuple[int, ...]) -> np.ndarray:
    """Generate a binary mask canvas filled completely inside a contour array."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if contour is not None:
        cv2.drawContours(mask, [contour], -1, 255, -1)
    return mask


def compute_circularity(contour: np.ndarray) -> float:
    """Calculate the isoperimetric quotient (circularity metric score)."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, closed=True)
    if perimeter == 0:
        return 0.0
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    return min(circularity, 1.0)


def fit_ellipse_score(contour: np.ndarray) -> Tuple[float, float]:
    """Fit an algebraic ellipse boundary to compute aspect ratio and matching index."""
    if len(contour) < 5 or cdist is None:
        return 0.0, 0.0

    try:
        ellipse = cv2.fitEllipse(contour)
        (_, _), (w, h), angle = ellipse

        aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
        ellipse_contour = cv2.ellipse2Poly(
            (int(ellipse[0][0]), int(ellipse[0][1])),
            (int(w / 2), int(h / 2)),
            int(angle), 0, 360, 5
        )

        cnt_pts = contour.reshape(-1, 2).astype(np.float32)
        ell_pts = ellipse_contour.astype(np.float32)

        dist_matrix = cdist(cnt_pts, ell_pts)
        match_score = 1.0 / (1.0 + np.mean(dist_matrix.min(axis=1)))

        return aspect_ratio, match_score
    except Exception:
        return 0.0, 0.0


def project_curvature_to_theta(
    kappa: np.ndarray,
    kappa_clean: np.ndarray,
    x_s: np.ndarray,
    y_s: np.ndarray,
    num_theta_bins: int = 360
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transforms continuous spatial curvature vectors onto a uniform polar angular grid.
    
    Returns:
        theta_grid: 1D grid array of values between [-pi, pi] in radians.
        kappa_norm_grid: 1D array of normalized curvature mapped directly to theta_grid.
    """
    # 1. Uniform peak normalization matching standard analysis routines
    kappa_max = np.max(np.abs(kappa_clean))
    if kappa_max < 1e-8:
        kappa_max = np.max(np.abs(kappa))
    kappa_max = max(kappa_max, 1e-10)
    kappa_normalized = kappa_clean / kappa_max

    # 2. Extract reference coordinate center of mass
    cx = np.mean(x_s)
    cy = np.mean(y_s)

    # 3. Project to polar angles relative to centroid location coordinates
    x_rel = x_s - cx
    y_rel = y_s - cy
    angles = np.arctan2(y_rel, x_rel)

    # 4. Sort angular domain array elements monotonically
    sort_idx = np.argsort(angles)
    sorted_angles = angles[sort_idx]
    sorted_kappa_norm = kappa_normalized[sort_idx]

    # 5. Handle periodic boundary continuity unwrapping across the wrap seam (-pi/pi)
    sorted_angles = np.concatenate(([sorted_angles[-1] - 2 * np.pi], sorted_angles, [sorted_angles[0] + 2 * np.pi]))
    sorted_kappa_norm = np.concatenate(([sorted_kappa_norm[-1]], sorted_kappa_norm, [sorted_kappa_norm[0]]))

    # 6. Sample precisely onto uniform step locations
    theta_grid = np.linspace(-np.pi, np.pi, num_theta_bins)
    kappa_norm_grid = np.interp(theta_grid, sorted_angles, sorted_kappa_norm)

    return theta_grid, kappa_norm_grid


# ==============================================================================
# CORE GPU COMPUTE RUNTIMES
# ==============================================================================

def cupy_frangi(image_gpu: Any, p: Dict[str, Any]) -> Any:
    """GPU accelerated Multi-Scale Frangi Vesselness Filter using CuPy arrays."""
    if not HAS_GPU:
        raise RuntimeError("CUDA toolkit environments are inaccessible.")

    sigmas = cp.arange(p["frangi_sigma_min"], p["frangi_sigma_max"], p["frangi_scale_step"])
    beta, c = 0.5, 15.0
    max_vesselness = cp.zeros_like(image_gpu)

    kernel_dxy = cp.array([
        [1, 0, -1],
        [0, 0, 0],
        [-1, 0, 1]
    ], dtype=cp.float32) / 4.0

    for sigma in sigmas:
        img_s = ndimg.gaussian_filter(image_gpu, sigma)
        Dxx = ndimg.convolve(img_s, cp.array([[1, -2, 1]], dtype=cp.float32))
        Dyy = ndimg.convolve(img_s, cp.array([[1, -2, 1]], dtype=cp.float32).T)
        Dxy = ndimg.convolve(img_s, kernel_dxy)

        diff = Dxx - Dyy
        sqrt_disc = cp.sqrt(diff**2 + 4 * Dxy**2)

        l1 = (Dxx + Dyy + sqrt_disc) / 2.0
        l2 = (Dxx + Dyy - sqrt_disc) / 2.0

        mask = cp.abs(l1) > cp.abs(l2)
        l1_s = cp.where(mask, l2, l1)
        l2_s = cp.where(mask, l1, l2)

        Rb = (l1_s / (l2_s + 1e-9))**2
        S2 = l1_s**2 + l2_s**2

        vesselness = cp.exp(-Rb / (2 * beta**2)) * (1 - cp.exp(-S2 / (2 * c**2)))
        vesselness = cp.where(l2_s < 0, vesselness, 0)
        max_vesselness = cp.maximum(max_vesselness, vesselness)

    return max_vesselness


# ==============================================================================
# CANDIDATE TRACKING & SEGMENTATION ENGINES
# ==============================================================================

def score_contour(
    contour: Optional[np.ndarray], 
    prev_state: Optional[Dict[str, Any]], 
    micron_per_pixel: float, 
    params: Dict[str, Any]
) -> float:
    """Evaluate a metric score evaluating morphological and tracking consistency."""
    if contour is None or len(contour) < 5:
        return 0.0

    circularity = compute_circularity(contour)
    aspect_ratio, ellipse_match = fit_ellipse_score(contour)

    if aspect_ratio > params["ellipse_aspect_ratio_max"]:
        shape_score = 0.1
    else:
        shape_score = 0.5 * circularity + 0.5 * ellipse_match

    area_px = cv2.contourArea(contour)
    area_um2 = area_px * (micron_per_pixel ** 2)

    if area_um2 < params["min_area_um2"]:
        area_score = 0.0
    elif area_um2 < 500:
        area_score = area_um2 / 500.0
    elif area_um2 < 5000:
        area_score = 1.0
    else:
        area_score = max(0.5, 1.0 - (area_um2 - 5000) / 10000)

    temporal_score = 1.0
    if prev_state is not None and "contour" in prev_state:
        prev_cnt = prev_state["contour"]
        M = cv2.moments(contour)
        M_prev = cv2.moments(prev_cnt)

        if M['m00'] != 0 and M_prev['m00'] != 0:
            cx, cy = M['m10'] / M['m00'], M['m01'] / M['m00']
            cx_prev, cy_prev = M_prev['m10'] / M_prev['m00'], M_prev['m01'] / M_prev['m00']

            centroid_dist_um = np.sqrt((cx - cx_prev)**2 + (cy - cy_prev)**2) * micron_per_pixel
            if centroid_dist_um > params["max_centroid_shift_um"]:
                temporal_score *= 0.3
            else:
                temporal_score *= (1.0 - centroid_dist_um / params["max_centroid_shift_um"])

            curr_radius_px = np.sqrt(area_px / np.pi)
            prev_radius_px = prev_state.get("radius_px", np.sqrt(cv2.contourArea(prev_cnt) / np.pi))

            radius_change = abs(curr_radius_px - prev_radius_px) / (prev_radius_px + 1e-6)
            if radius_change > params["max_radius_change"]:
                temporal_score *= 0.2
            else:
                temporal_score *= (1.0 - radius_change / params["max_radius_change"])

    return float(
        params["circularity_weight"] * shape_score +
        params["area_weight"] * area_score +
        params["temporal_weight"] * temporal_score
    )


_gpu_runtime_broken = False  # set True the first time a GPU call fails mid-run; per-process


def analyze_mother_frame_gpu_fast(
    img: np.ndarray, 
    micron_per_pixel: float, 
    p: Dict[str, Any], 
    prev_state: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray, Dict[str, Any]]:
    """Segment a single tracking frame utilizing optimal GPU acceleration cascades."""
    global HAS_GPU, _gpu_runtime_broken
    if not HAS_GPU or _gpu_runtime_broken:
        return analyze_mother_frame_cpu(img, micron_per_pixel, p, prev_state)

    try:
        img_gpu = cp.asarray(img, dtype=cp.float32)
        img_blur = ndimg.gaussian_filter(img_gpu, p["gauss_sigma"])

        gx = ndimg.sobel(img_blur, axis=1)
        gy = ndimg.sobel(img_blur, axis=0)
        g_mag = cp.sqrt(gx * gx + gy * gy)
        g_mag /= (cp.max(g_mag) + 1e-9)

        frangi_gpu = cupy_frangi(g_mag, p)
        frangi_gpu -= frangi_gpu.min()
        frangi_gpu /= (frangi_gpu.max() + 1e-9)

        frangi_u8 = cp.asnumpy((frangi_gpu * 255).astype(cp.uint8))
    except Exception as e:
        # A GPU call failed mid-run even though the startup probe passed (e.g. driver
        # hiccup, out-of-memory, or contention between multiple worker processes each
        # holding a CUDA context). Disable GPU for the rest of this process and
        # transparently continue on CPU rather than losing the whole run.
        _gpu_runtime_broken = True
        HAS_GPU = False
        print(f"⚠ GPU kernel execution failed at runtime ({e}) — disabling GPU for the "
              "rest of this process and falling back to CPU.")
        return analyze_mother_frame_cpu(img, micron_per_pixel, p, prev_state)

    _, frangi_bin = cv2.threshold(frangi_u8, p["threshold_value"], 255, cv2.THRESH_BINARY)

    try:
        skel_u8 = cv2.ximgproc.thinning(frangi_bin, thinningType=cv2.ximgproc.THINNING_GUOHALL)
    except AttributeError:
        skel_u8 = (skeletonize(frangi_bin > 0) * 255).astype(np.uint8)

    contours, _ = cv2.findContours(skel_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    mother_contour = None
    best_score = 0.0

    if contours:
        scores = np.array([score_contour(cnt, prev_state, micron_per_pixel, p) for cnt in contours])
        if np.any(scores > 0):
            best_idx = np.argmax(scores)
            mother_contour = contours[best_idx]
            best_score = scores[best_idx]

    intermediates = {"frangi": frangi_u8, "skeleton": skel_u8, "score": best_score}
    return mother_contour, frangi_u8, skel_u8, intermediates


def analyze_mother_frame_cpu(
    img: np.ndarray, 
    micron_per_pixel: float, 
    p: Dict[str, Any], 
    prev_state: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray, Dict[str, Any]]:
    """CPU runtime alternative path for systems lacking discrete parallel architecture nodes."""
    k = int(p["gauss_ksize"]) // 2 * 2 + 1
    img_blur = cv2.GaussianBlur(img, (k, k), sigmaX=p["gauss_sigma"])

    if BuildFFT is not None and Wavelet2D_gauss is not None:
        try:
            xg, yg, fft_img = BuildFFT.img2D(img_blur)
            WT_mod, _ = Wavelet2D_gauss.firstder(p["wavelet_scale"], xg, yg, fft_img)
            gradient_norm = WT_mod / (np.max(WT_mod) + 1e-9)
        except Exception:
            gx = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3)
            gradient_norm = np.sqrt(gx**2 + gy**2)
            gradient_norm /= (gradient_norm.max() + 1e-9)
    else:
        gx = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3)
        gradient_norm = np.sqrt(gx**2 + gy**2)
        gradient_norm /= (gradient_norm.max() + 1e-9)

    frangi_img = frangi(
        gradient_norm,
        sigmas=range(int(p["frangi_sigma_min"]), int(p["frangi_sigma_max"]), int(p["frangi_scale_step"])),
        black_ridges=False
    )

    frangi_u8 = cv2.normalize(frangi_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, frangi_bin = cv2.threshold(frangi_u8, p["threshold_value"], 255, cv2.THRESH_BINARY)
    skel_u8 = (skeletonize(frangi_bin > 0) * 255).astype(np.uint8)

    contours, _ = cv2.findContours(skel_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mother_contour = None
    best_score = 0.0

    if contours:
        scored = [(cnt, score_contour(cnt, prev_state, micron_per_pixel, p)) for cnt in contours]
        scored = [item for item in scored if item[1] > 0]
        if scored:
            scored.sort(key=lambda x: x[1], reverse=True)
            mother_contour, best_score = scored[0]

    intermediates = {"frangi": frangi_u8, "skeleton": skel_u8, "score": best_score}
    return mother_contour, frangi_u8, skel_u8, intermediates


# ==============================================================================
# HUMAN-IN-THE-LOOP INTERACTION METHODS
# ==============================================================================

class ManualCorrector:
    """Polygonal interface utility context for direct operator manual override parsing."""
    
    def __init__(self, img: np.ndarray, title: str = "Draw Mother Contour"):
        self.img = img
        self.contour: Optional[np.ndarray] = None
        self.mask: Optional[np.ndarray] = None
        self.verts: List[Tuple[float, float]] = []

        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.imshow(img, cmap='gray')
        self.ax.set_title(f"{title}\nClick vertices -> Complete with ENTER.")

        self.selector = PolygonSelector(self.ax, self.on_select)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        plt.show(block=True)

    def on_select(self, verts: List[Tuple[float, float]]):
        self.verts = verts

    def on_key(self, event: Any):
        if event.key == 'enter' and len(self.verts) > 2:
            pts = np.array(self.verts, dtype=np.int32).reshape((-1, 1, 2))
            self.contour = pts
            self.mask = contour_to_mask(pts, self.img.shape)
            plt.close(self.fig)

    @staticmethod
    def get_correction(img: np.ndarray, frame_idx: int) -> Optional[np.ndarray]:
        mc = ManualCorrector(img, title=f"Manual ROI Intervention | Frame {frame_idx}")
        return mc.contour


class MotherTuner:
    """Interactive GUI module using Matplotlib Sliders to test hyperparameter responses."""
    
    def __init__(self, frames: List[np.ndarray], micron_per_pixel: float):
        self.frames = frames
        self.um_px = micron_per_pixel
        self.n_frames = len(frames)
        self.p = DEFAULT_MOTHER_PARAMS.copy()

        self.fig = plt.figure(figsize=(16, 8))
        self.ax_orig = self.fig.add_axes([0.05, 0.4, 0.3, 0.5])
        self.ax_frangi = self.fig.add_axes([0.38, 0.4, 0.3, 0.5])
        self.ax_res = self.fig.add_axes([0.71, 0.4, 0.25, 0.5])

        self.sliders: Dict[str, Slider] = {}
        slider_configs = [
            ('frame', 0, 0, self.n_frames - 1, 1),
            ('gauss_sigma', 3.0, 0.1, 10.0, 0.1),
            ('wavelet_scale', 1.2, 0.5, 5.0, 0.1),
            ('threshold_value', 5, 0, 100, 1)
        ]

        y_position = 0.25
        for key, val, vmin, vmax, step in slider_configs:
            ax_slide = self.fig.add_axes([0.2, y_position, 0.6, 0.03])
            self.sliders[key] = Slider(ax_slide, key, vmin, vmax, valinit=val, valstep=step)
            self.sliders[key].on_changed(self.update)
            y_position -= 0.05

        self.update(None)

    def update(self, _: Any):
        for k in self.sliders:
            if k != 'frame':
                self.p[k] = self.sliders[k].val

        idx = int(self.sliders['frame'].val)
        img = self.frames[idx]

        cnt, frangi_vis, _, _ = analyze_mother_frame_gpu_fast(img, self.um_px, self.p)

        self.ax_orig.imshow(img, cmap='gray')
        self.ax_frangi.imshow(frangi_vis, cmap='inferno')

        res = cv2.cvtColor(normalize(img), cv2.COLOR_GRAY2RGB)
        if cnt is not None:
            cv2.drawContours(res, [cnt], -1, (0, 1, 0), 2)

        self.ax_res.imshow(res)
        self.fig.canvas.draw_idle()

    @classmethod
    def tune(cls, frames: List[np.ndarray], px: float) -> Dict[str, Any]:
        tuner = cls(frames, px)
        plt.show()
        return tuner.p


# ==============================================================================
# PIPELINE DATA EXTRACTION AND I/O BATCH EXECUTION RUNTIMES
# ==============================================================================

def _process_chunk(
    args: Tuple[List[np.ndarray], int, float, Dict[str, Any]],
    bar: Optional[Any] = None
) -> Tuple[int, Dict[str, List[Any]]]:
    """Worker process subroutine designed to run parallel vector chunks across discrete CPUs.

    `bar` is an optional alive_bar callable that gets ticked once per frame. It is only
    ever passed when this function is called directly in-process (sequential mode) - Pool
    workers always call it with the default (None), since a live progress-bar object can't
    be pickled/shared across separate processes.
    """
    frames_chunk, start_idx, micron_per_pixel, params = args

    chunk_results: Dict[str, List[Any]] = {
        "contour": [], "area_um2": [], "area_px": [], "perimeter_um": [],
        "perimeter_px": [], "radius_um": [], "centroid": [], "score": [], "circularity": [],
        "theta_grid": [], "kappa_normalized_grid": []
    }

    prev_state = None
    for frame in frames_chunk:
        cnt, _, _, intermediates = analyze_mother_frame_gpu_fast(frame, micron_per_pixel, params, prev_state)
        curr_score = intermediates.get("score", 0.0)

        if cnt is not None:
            area_px = cv2.contourArea(cnt)
            perimeter_px = cv2.arcLength(cnt, True)
            radius_px = np.sqrt(area_px / np.pi)

            M = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00']) if M['m00'] != 0 else 0
            cy = int(M['m01'] / M['m00']) if M['m00'] != 0 else 0

            circ = 4 * np.pi * (area_px * (micron_per_pixel**2)) / ((perimeter_px * micron_per_pixel) ** 2)
            prev_state = {"contour": cnt, "radius_px": radius_px, "centroid": (cx, cy)}

            # Pure Numeric Spatial Curvature Analysis Mapping Execution
            try:
                kappa, kappa_clean, s, x_s, y_s = compute_curvature(
                    cnt, remove_outliers=True, interpolate=True, num_points=200
                )
                t_grid, k_grid = project_curvature_to_theta(kappa, kappa_clean, x_s, y_s, num_theta_bins=360)
            except Exception:
                t_grid = np.linspace(-np.pi, np.pi, 360)
                k_grid = np.full(360, np.nan)

            chunk_results["contour"].append(cnt)
            chunk_results["radius_um"].append(radius_px * micron_per_pixel)
            chunk_results["area_um2"].append(area_px * (micron_per_pixel**2))
            chunk_results["perimeter_um"].append(perimeter_px * micron_per_pixel)
            chunk_results["centroid"].append((cx, cy))
            chunk_results["score"].append(curr_score)
            chunk_results["perimeter_px"].append(perimeter_px)
            chunk_results["area_px"].append(area_px)
            chunk_results["circularity"].append(circ)
            chunk_results["theta_grid"].append(t_grid)
            chunk_results["kappa_normalized_grid"].append(k_grid)
        else:
            for k in chunk_results:
                if k == "score":
                    chunk_results[k].append(0.0)
                elif k in ["theta_grid", "kappa_normalized_grid"]:
                    chunk_results[k].append(np.full(360, np.nan) if k == "kappa_normalized_grid" else np.linspace(-np.pi, np.pi, 360))
                else:
                    chunk_results[k].append(None if k in ["contour", "centroid"] else np.nan)

        if bar is not None:
            bar()

    return start_idx, chunk_results


def _save_curvature_worker(args: Tuple[int, np.ndarray, Dict[str, Path]]) -> Tuple[int, np.ndarray, np.ndarray]:
    """Pure mathematical transformation thread worker mapping curvature vectors to a theta space grid."""
    i, cnt, _ = args
    if cnt is None:
        return i, np.linspace(-np.pi, np.pi, 360), np.full(360, np.nan)
    try:
        kappa, kappa_clean, s, x_s, y_s = compute_curvature(
            cnt, remove_outliers=True, interpolate=True, num_points=200
        )
        theta_grid, kappa_norm_grid = project_curvature_to_theta(
            kappa, kappa_clean, x_s, y_s, num_theta_bins=360
        )
        return i, theta_grid, kappa_norm_grid
    except Exception as e:
        print(f"[ERROR] Mathematical angular transformation failed at Frame {i}: {e}")
        return i, np.linspace(-np.pi, np.pi, 360), np.full(360, np.nan)


def save_curvature_parallel(results: Dict[str, List[Any]], output_folder: Path, n_workers: int = 2) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Execute mathematical polar mapping pipelines across a dedicated parallel worker pool."""
    tasks = [
        (i, cnt, {}) for i, cnt in enumerate(results["contour"])
    ]
    workers = min(n_workers, cpu_count())
    
    print("📈 Extracting unified polar curvature metric distributions...")
    with Pool(workers) as pool:
        mapped_outputs = pool.map(_save_curvature_worker, tasks)
        
    return {item[0]: (item[1], item[2]) for item in mapped_outputs}


def save_mother_results(
    frames: List[np.ndarray],
    results: Dict[str, List[Any]],
    micron_per_pixel: float,
    output_folder: Union[str, Path] = "mother_data",
    save_data: bool = True,
    save_training: bool = False,
    save_curvature: bool = False,
    save_every: int = 1,
    n_save_workers: int = 4
):
    """Export frame segmentations asynchronously to optimize system disk I/O bottlenecks."""
    out_path = Path(output_folder)
    dirs = {k: out_path / k for k in ["raw", "overlay"]}
    tr_dirs = {k: out_path / "training_data" / k for k in ["images", "masks"]}

    if save_data:
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        if save_training:
            for d in tr_dirs.values():
                d.mkdir(parents=True, exist_ok=True)

    print("💾 Committing analysis frame arrays to storage array elements...")
    frames_u8 = [(normalize(f) * 255).astype(np.uint8) for f in frames]

    def save_single_frame(i: int):
        if i % save_every != 0:
            return
        frame = frames_u8[i]
        cnt = results["contour"][i]
        score = results["score"][i]
        fname = f"frame_{i:04d}.png"

        cv2.imwrite(str(dirs["raw"] / fname), frame)

        res = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if cnt is not None:
            cv2.drawContours(res, [cnt], -1, (0, 255, 0), 2)
            cv2.putText(res, f"{score:.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imwrite(str(dirs["overlay"] / fname), res)

        if save_training:
            cv2.imwrite(str(tr_dirs["images"] / fname), frame)
            mask = contour_to_mask(cnt, frame.shape)
            cv2.imwrite(str(tr_dirs["masks"] / fname), mask)

    if n_save_workers > 0:
        with ThreadPoolExecutor(max_workers=n_save_workers) as executor:
            list(executor.map(save_single_frame, range(len(frames))))
    else:
        for i in range(len(frames)):
            save_single_frame(i)

    if save_curvature:
        # Executes polar mappings and writes structural numerical metrics to storage arrays directly
        curvature_profiles = save_curvature_parallel(results, out_path, n_workers=2)
        
        # Save structural multi-dimensional files directly to workspace root locations
        theta_stack = np.array([curvature_profiles[i][0] for i in range(len(frames))])
        kappa_stack = np.array([curvature_profiles[i][1] for i in range(len(frames))])
        
        np.save(str(out_path / "theta_grid_radians.npy"), theta_stack)
        np.save(str(out_path / "kappa_normalized_vs_theta.npy"), kappa_stack)

    print("✅ Frame export finalized.")


def detect_mother_stack_parallelized(
    frames: List[np.ndarray],
    micron_per_pixel: float,
    params: Optional[Dict[str, Any]] = None,
    save_data: bool = False,
    output_folder: str = "mother_data",
    save_training: bool = False,
    save_curvature: bool = False,
    n_workers: Optional[int] = None,
    chunk_size: int = 20,
    parallel: bool = True
) -> Dict[str, List[Any]]:
    """
    Execute high-throughput biological boundary extraction processing.

    parallel=True  -> chunks are spread across a multiprocessing Pool. Fastest, but no
                       progress bar (worker processes can't share one live object).
    parallel=False -> frames are processed one at a time in this process, with an
                       alive_bar progress indicator.

    Returns structured parameters and spatial angular distribution tracking data arrays.
    """
    p_dict = params if params is not None else DEFAULT_MOTHER_PARAMS

    final: Dict[str, List[Any]] = {
        "contour": [], "area_um2": [], "area_px": [], "perimeter_um": [],
        "perimeter_px": [], "radius_um": [], "centroid": [], "score": [], "circularity": [],
        "theta_grid": [], "kappa_normalized_grid": []
    }

    if parallel:
        workers = n_workers if n_workers is not None else min(2, cpu_count())
        print(f"🚀 Spawning tracking threads across ({workers}) multi-core workers (parallel, no progress bar)...")

        chunks = [
            (frames[i:i + chunk_size], i, micron_per_pixel, p_dict)
            for i in range(0, len(frames), chunk_size)
        ]

        with Pool(workers) as pool:
            raw_chunks = pool.map(_process_chunk, chunks)

        raw_chunks.sort(key=lambda x: x[0])

        for _, chunk_res in raw_chunks:
            for key in final:
                final[key].extend(chunk_res[key])
    else:
        print("▶ Running mother detection sequentially on CPU (single process, progress bar below)...")
        with alive_bar(len(frames), title="Mother detection") as bar:
            _, chunk_res = _process_chunk((frames, 0, micron_per_pixel, p_dict), bar=bar)
            for key in final:
                final[key].extend(chunk_res[key])

    if save_data:
        save_mother_results(
            frames, final, micron_per_pixel,
            output_folder=output_folder,
            save_data=save_data,
            save_training=save_training,
            save_curvature=save_curvature
        )

    return final