"""
Daughter Vesicle Segmentation, Trajectory Linking, and Geometrical Contact Pipeline.

Contact Details:
Email(s): itstanmaypandey@gmail.com , caritra@iitg.ac.in
"""

from multiprocessing import Pool, cpu_count
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import tensorflow as tf
import cv2
from alive_progress import alive_bar
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from skimage.color import label2rgb
from skimage.measure import regionprops
import torch
import trackpy as tp

# Optional Machine Learning Engine Backend Support Initialisation
try:
    from cellpose import models
    CELLPOSE_AVAILABLE = True
except Exception as e:
    CELLPOSE_AVAILABLE = False
    CELLPOSE_IMPORT_ERROR = e

try:
    from stardist.models import StarDist2D
    from csbdeep.utils import normalize as csbdeep_normalize
    STARDIST_AVAILABLE = True
except Exception as e:
    STARDIST_AVAILABLE = False
    STARDIST_IMPORT_ERROR = e


# ==============================================================================
# DATA FORMATTING & MATHEMATICAL UTILITIES
# ==============================================================================

def save_local_tracking_csv(final_results: Dict[str, List[Any]], output_folder: Union[str, Path]):
    """Flatten tracking list outputs into structured long-format Pandas DataFrames."""
    rows = []
    for frame_stats in final_results.get('stats', []):
        for s in frame_stats:
            row_copy = s.copy()
            if isinstance(row_copy.get('daughter_contour'), list):
                pass  # Already a list
            rows.append(row_copy)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(Path(output_folder) / "daughter_tracking_local.csv", index=False)


def kdist(m_pts: np.ndarray, d_cnt: np.ndarray) -> Tuple[float, int, int]:
    """Execute high-speed single nearest-neighbor queries via a spatial KD-Tree."""
    tree = cKDTree(m_pts)
    dists, idxs = tree.query(d_cnt, k=1)
    min_i = np.argmin(dists)
    return float(dists[min_i]), int(min_i), int(idxs[min_i])


def translate_mask(mask: np.ndarray, tx: float, ty: float) -> np.ndarray:
    """Apply discrete affine vector translation matrices cleanly to geometric arrays."""
    h, w = mask.shape
    matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(
        mask.astype(np.uint16), matrix, (w, h),
        flags=cv2.INTER_NEAREST, borderValue=0
    )


def compute_circularity(mask: np.ndarray) -> float:
    """Calculate the isoperimetric quotient score directly from a binary component mask."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    cnt = cnts[0]
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        return 0.0
    return float(4 * np.pi * area / (perimeter * perimeter))


def curvature_radius(contour: np.ndarray, idx: int, k: int = 5) -> Optional[float]:
    """Estimate local boundary radius of curvature via Menger curvature approximations."""
    n = len(contour)
    if n < 2 * k + 1:
        return None

    p1 = contour[(idx - k) % n]
    p2 = contour[idx]
    p3 = contour[(idx + k) % n]

    a = np.linalg.norm(p2 - p1)
    b = np.linalg.norm(p3 - p2)
    c = np.linalg.norm(p3 - p1)

    if a * b * c == 0:
        return None

    s = (a + b + c) / 2
    area = s * (s - a) * (s - b) * (s - c)
    if area <= 0:
        return None

    return float((a * b * c) / (4 * np.sqrt(area)))


def get_contour_points(mask: np.ndarray) -> np.ndarray:
    """Extract ordered contour coordinates from a targeted binary frame slice."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        return cnts[0].reshape(-1, 2)
    return np.empty((0, 2), dtype=np.float32)


# ==============================================================================
# BIFURCATED SEGMENTATION INDEPENDENT CORE FUNCTIONS
# ==============================================================================

def segment_with_stardist(img_input: np.ndarray) -> np.ndarray:
    """
    Execute instance segmentation using a pretrained StarDist2D fluo model.
    
    Args:
        img_input: Normalized float32 image array in range [0.0, 1.0].
    Returns:
        labels: 16-bit unsigned integer instance segmentation label mask.
    """
    if not STARDIST_AVAILABLE:
        raise ImportError(f"StarDist runtime library is inaccessible: {STARDIST_IMPORT_ERROR}")
    
    # Load model context inside execution environment thread
    model = StarDist2D.from_pretrained("2D_versatile_fluo")
    img_norm = csbdeep_normalize(img_input, 1, 99.8)
    labels, _ = model.predict_instances(img_norm, prob_thresh=0.5, nms_thresh=0.4)
    return labels.astype(np.uint16)


def segment_with_cellpose(img_input: np.ndarray, use_gpu: bool = True) -> np.ndarray:
    """
    Execute instance segmentation using a custom trained Cellpose pipeline.
    
    Args:
        img_input: Normalized float32 image array in range [0.0, 1.0].
        use_gpu: Toggles hardware CUDA acceleration.
    Returns:
        masks: 16-bit unsigned integer instance segmentation label mask.
    """
    if not CELLPOSE_AVAILABLE:
        raise ImportError(f"Cellpose runtime library is inaccessible: {CELLPOSE_IMPORT_ERROR}")
    
    model = models.CellposeModel(gpu=use_gpu, nchan=1, pretrained_model="cpsam")
    masks, _, _ = model.eval(
        img_input,
        diameter=None,
        flow_threshold=0.4,
        cellprob_threshold=0.0
    )
    return masks.astype(np.uint16)


# ==============================================================================
# DATA PRESENTATION & FILE EXPORT VISUAL METHODS
# ==============================================================================

def save_visuals_robust(
    output_dir: str, fname: str, raw: np.ndarray, masks: np.ndarray, 
    mother_cnt: Optional[np.ndarray], mother_radius: float, daughters_data: List[Dict[str, Any]]
):
    """Render color-coded overlay diagnostics detailing membrane boundary metrics."""
    base_path = Path(output_dir)
    for sub in ["1_raw", "2_overlay_clean", "3_overlay_status"]:
        (base_path / sub).mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(base_path / "1_raw" / fname), raw)
    vis_base = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    
    if mother_cnt is not None:
        cv2.drawContours(vis_base, [mother_cnt], -1, (255, 0, 0), 2)
        if mother_radius > 0:
            M = cv2.moments(mother_cnt)
            if M["m00"] != 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                cv2.circle(vis_base, (cx, cy), int(mother_radius), (0, 255, 255), 1)

    vis_clean = vis_base.copy()
    overlay = np.zeros_like(vis_clean)
    for d in daughters_data:
        mask = (masks == d['label'])
        overlay[mask] = d['color']
    vis_clean = cv2.addWeighted(vis_clean, 1.0, overlay, 0.5, 0)
    
    for d in daughters_data:
        mask = (masks == d['label']).astype(np.uint8)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_clean, cnts, -1, d['color'], 1)
    cv2.imwrite(str(base_path / "2_overlay_clean" / fname), vis_clean)

    vis_status = vis_base.copy()
    for d in daughters_data:
        mask = (masks == d['label']).astype(np.uint8)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_status, cnts, -1, d['color'], 2)
        if cnts:
            M = cv2.moments(cnts[0])
            if M["m00"] != 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                cv2.putText(vis_status, str(d['id']), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                cv2.putText(vis_status, str(d['id']), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, d['color'], 1)
    cv2.imwrite(str(base_path / "3_overlay_status" / fname), vis_status)


def save_binary_daughter_sample(base_dir: str, raw_img: np.ndarray, daughter_mask: np.ndarray, frame_idx: int, daughter_id: int):
    """Commit raw frames and binary regions directly to training folders."""
    p = Path(base_dir)
    img_dir, mask_dir = p / "images", p / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    fname = f"frame_{frame_idx:04d}_d{daughter_id:03d}.png"
    raw_u8 = cv2.normalize(raw_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mask_u8 = (daughter_mask.astype(np.uint8) * 255)

    cv2.imwrite(str(img_dir / fname), raw_u8)
    cv2.imwrite(str(mask_dir / fname), mask_u8)


# ==============================================================================
# PIPELINE MULTIPROCESSING COMPUTE LAYERS
# ==============================================================================

GLOBAL_BACKEND_NAME: str = "stardist"

def init_worker(backend: str):
    """Assign target configuration names down to local sub-worker contexts."""
    global GLOBAL_BACKEND_NAME
    GLOBAL_BACKEND_NAME = backend.lower()


def process_frame_worker(args: Tuple[int, np.ndarray, Dict[str, Any], float, Optional[np.ndarray], str, int]) -> Optional[Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
    """Execute segmented morphometry math on an isolated video slice."""
    i, img, mother_data, micron_per_pixel, REF_CENTER, output_folder, RADIUS_BUFFER_PX = args
    backend = GLOBAL_BACKEND_NAME

    try:
        m_cnt = None
        m_radius_eff = 0.0
        tx, ty = 0.0, 0.0

        if "contour" in mother_data and i < len(mother_data["contour"]) and mother_data["contour"][i] is not None:
            m_cnt = mother_data["contour"][i]
            M = cv2.moments(m_cnt)
            if M["m00"] != 0:
                cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
                if REF_CENTER is not None:
                    tx, ty = float(REF_CENTER[0] - cx), float(REF_CENTER[1] - cy)
            m_radius_eff = np.sqrt(cv2.contourArea(m_cnt) / np.pi)

        fname = f"frame_{i:04d}.png"
        img_f = img.astype(np.float32)
        p1, p99 = np.percentile(img_f, (1, 99))
        img_input = np.clip((img_f - p1) / (p99 - p1 + 1e-9), 0.0, 1.0)

        # Route dynamically to the requested isolated segmentation function
        if backend == "cellpose":
            masks = segment_with_cellpose(img_input, use_gpu=False)
        else:
            masks = segment_with_stardist(img_input)

        regions = regionprops(masks)
        detections = []
        daughters_viz = []
        mother_area = cv2.contourArea(m_cnt) if m_cnt is not None else 0.0

        if m_cnt is not None and mother_area > 0:
            M = cv2.moments(m_cnt)
            mcx, mcy = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) if M["m00"] != 0 else (0, 0)

            for k, reg in enumerate(regions):
                y, x = reg.centroid
                dist = np.sqrt((x - mcx)**2 + (y - mcy)**2)
                if dist < (m_radius_eff + RADIUS_BUFFER_PX) and reg.area < (0.5 * mother_area):
                    daughters_viz.append({'label': reg.label, 'color': (255, 255, 0), 'id': k})

                    binary = (masks == reg.label).astype(np.uint8)
                    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not cnts:
                        continue
                    contour = max(cnts, key=cv2.contourArea).squeeze()
                    if contour.ndim != 2 or contour.shape[1] != 2:
                        continue

                    if compute_circularity(binary) >= 0.6:
                        detections.append({
                            'frame': i, 'y': float(y + ty), 'x': float(x + tx),
                            'abs_y': float(y), 'abs_x': float(x),
                            'area': float(reg.area * (micron_per_pixel**2)),
                            'label_orig': int(reg.label), 'contour': contour
                        })

        save_visuals_robust(
            os.path.join(output_folder, "segmented_output_inner"), fname,
            cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            masks, m_cnt, m_radius_eff, daughters_viz
        )

        frame_cache_entry = {
            'img': img, 'masks': masks, 'm_cnt': m_cnt, 'm_rad': m_radius_eff,
            'fname': fname, 'orig_frame': i, 'translation': (tx, ty)
        }
        return frame_cache_entry, detections

    except Exception as e:
        print(f"[ERROR] Frame {i} runtime segmentation failure: {e}")
        return None


def process_daughters_parallel(
    frames: List[np.ndarray], mother_data: Dict[str, Any], micron_per_pixel: float,
    output_folder: str = "results", save_training: bool = False, supraname: Optional[str] = None,
    iflocalTracking: bool = True, n_workers: Optional[int] = None, use_gpu: bool = True,
    parallel: bool = True
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Execute tracking pipelines across movie frame stacks safely.

    use_gpu   - which device the segmentation backend runs on.
    parallel  - execution topology, independent of device:
                  True  -> multiprocessing Pool across CPU workers, no progress bar.
                  False -> single process, frame-by-frame, with an alive_bar progress bar.

    GPU execution can't safely be spread across multiple processes (each worker would need
    its own CUDA/TensorFlow context), so if use_gpu=True and parallel=True are both
    requested, we fall back to sequential GPU processing and say so.
    """
    print("\n--- Segmentation Initialisation ---")
    SEGMENTATION_BACKEND = "stardist"  # Set to "cellpose" or "stardist" dynamically

    gpu_devices = tf.config.list_physical_devices('GPU')
    if use_gpu and not gpu_devices:
        print("⚠ use_gpu=True but TensorFlow reports no visible GPU device — "
              "proceeding on GPU path anyway; if this fails, check your CUDA/driver setup "
              "or call with use_gpu=False.")

    if use_gpu and parallel:
        print("⚠ parallel=True is incompatible with use_gpu=True (GPU context can't be "
              "shared across worker processes) — falling back to sequential GPU processing.")
        parallel = False

    print(f"Segmentation backend = {SEGMENTATION_BACKEND}")
    print(f"use_gpu = {use_gpu} (TensorFlow-visible GPU devices: {len(gpu_devices)})")
    print(f"execution mode = {'parallel (CPU, no progress bar)' if parallel else 'sequential (progress bar)'}")
    print("--------------------------------\n")

    REF_CENTER = None
    for cnt in mother_data.get("contour", []):
        if cnt is not None:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                REF_CENTER = np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]])
                break

    RADIUS_BUFFER_PX = 5
    all_detections = []
    frame_cache = []
#    use_gpu=False
    # Safe Multiprocessing Allocation Capping Limit Parameter to completely halt OpenBLAS RAM crashes
    workers = n_workers if n_workers is not None else (1 if use_gpu else min(2, cpu_count()))
    print(f"Phase 1: Detection Runtime active across ({workers}) target workers...")

    if parallel:
        # parallel=True implies use_gpu=False at this point (see fallback above).
        print(f"🚀 Running PARALLEL workflow topology processing on CPU via ({workers}) tasks.")
        args = [
            (i, frames[i], mother_data, micron_per_pixel, REF_CENTER, output_folder, RADIUS_BUFFER_PX)
            for i in range(len(frames))
        ]
        with Pool(workers, initializer=init_worker, initargs=(SEGMENTATION_BACKEND,)) as pool:
            results = pool.map(process_frame_worker, args)
            
        for res in results:
            if res is not None:
                fc, det = res
                frame_cache.append(fc)
                all_detections.extend(det)
    elif not use_gpu:
        print("▶ Running SEQUENTIAL processing workflow on CPU (single process, progress bar below).")
        init_worker(SEGMENTATION_BACKEND)  # sets GLOBAL_BACKEND_NAME for this process
        args = [
            (i, frames[i], mother_data, micron_per_pixel, REF_CENTER, output_folder, RADIUS_BUFFER_PX)
            for i in range(len(frames))
        ]
        with alive_bar(len(frames), title="Daughter detection (CPU)") as bar:
            for a in args:
                res = process_frame_worker(a)
                if res is not None:
                    fc, det = res
                    frame_cache.append(fc)
                    all_detections.extend(det)
                bar()
    else:
        print("⚠ Running SEQUENTIAL processing workflow loops on direct GPU layers.")
        global GLOBAL_BACKEND_NAME
        GLOBAL_BACKEND_NAME = SEGMENTATION_BACKEND

        with alive_bar(len(frames)) as bar:
            for i in range(len(frames)):
                # Handle single-stream direct evaluations straight to the GPU card
                try:
                    m_cnt = None
                    m_radius_eff = 0.0
                    tx, ty = 0.0, 0.0

                    if "contour" in mother_data and i < len(mother_data["contour"]) and mother_data["contour"][i] is not None:
                        m_cnt = mother_data["contour"][i]
                        M = cv2.moments(m_cnt)
                        if M["m00"] != 0:
                            cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
                            if REF_CENTER is not None:
                                tx, ty = float(REF_CENTER[0] - cx), float(REF_CENTER[1] - cy)
                        m_radius_eff = np.sqrt(cv2.contourArea(m_cnt) / np.pi)

                    fname = f"frame_{i:04d}.png"
                    img_f = frames[i].astype(np.float32)
                    p1, p99 = np.percentile(img_f, (1, 99))
                    img_input = np.clip((img_f - p1) / (p99 - p1 + 1e-9), 0.0, 1.0)

                    if SEGMENTATION_BACKEND == "cellpose":
                        masks = segment_with_cellpose(img_input, use_gpu=True)
                    else:
                        masks = segment_with_stardist(img_input)

                    regions = regionprops(masks)
                    daughters_viz = []
                    mother_area = cv2.contourArea(m_cnt) if m_cnt is not None else 0.0

                    if m_cnt is not None and mother_area > 0:
                        M = cv2.moments(m_cnt)
                        mcx, mcy = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) if M["m00"] != 0 else (0, 0)

                        for k, reg in enumerate(regions):
                            y, x = reg.centroid
                            dist = np.sqrt((x - mcx)**2 + (y - mcy)**2)
                            if dist < (m_radius_eff + RADIUS_BUFFER_PX) and reg.area < (0.5 * mother_area):
                                daughters_viz.append({'label': reg.label, 'color': (255, 255, 0), 'id': k})

                                binary = (masks == reg.label).astype(np.uint8)
                                cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                if not cnts:
                                    continue
                                contour = max(cnts, key=cv2.contourArea).squeeze()
                                if contour.ndim != 2 or contour.shape[1] != 2:
                                    continue

                                if compute_circularity(binary) >= 0.6:
                                    all_detections.append({
                                        'frame': i, 'y': float(y + ty), 'x': float(x + tx),
                                        'abs_y': float(y), 'abs_x': float(x),
                                        'area': float(reg.area * (micron_per_pixel**2)),
                                        'label_orig': int(reg.label), 'contour': contour
                                    })

                    save_visuals_robust(
                        os.path.join(output_folder, "segmented_output_inner"), fname,
                        cv2.normalize(frames[i], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                        masks, m_cnt, m_radius_eff, daughters_viz
                    )

                    frame_cache.append({
                        'img': frames[i], 'masks': masks, 'm_cnt': m_cnt, 'm_rad': m_radius_eff,
                        'fname': fname, 'orig_frame': i, 'translation': (tx, ty)
                    })
                except Exception as gpu_err:
                    print(f"[GPU ERROR] Processing failed on Frame {i}: {gpu_err}")
                bar()

    if iflocalTracking and all_detections:
        print("Phase 2: Linking Local Trajectories...")
        local_dir = Path(output_folder) / "innerlocalDaughter"
        local_dir.mkdir(parents=True, exist_ok=True)

        local_tracking = tracking(frame_cache, str(local_dir), all_detections, save_training=save_training, save_images=False)
        save_local_tracking_csv(local_tracking, str(local_dir))

    print("✅ Done.")
    return frame_cache, all_detections


# ==============================================================================
# SPATIAL LINKING AND GEOMETRICAL PROFILING
# ==============================================================================

def tracking(frame_cache: List[Dict[str, Any]], output_folder: str, all_detections: List[Dict[str, Any]], save_training: bool = False, save_images: bool = True) -> Dict[str, List[Any]]:
    """Link spatial components across time coordinates and isolate flat interaction coordinates."""
    CONTACT_PX = 10.0
    print("Phase 2: Linking Trajectories...")
    if not all_detections:
        return {'stats': [], 'regions': []}

    df = pd.DataFrame(all_detections)
    t = tp.link_df(df, search_range=10, memory=5, pos_columns=['y', 'x'])
    if t.empty:
        return {'stats': [], 'regions': []}
    t = tp.filter_stubs(t, threshold=5)
    t = t.reset_index(drop=True)

    print("Phase 3: Processing Local Profiles and Visual Outputs...")
    final_results = {'regions': [], 'stats': []}
    n_frames = len(frame_cache)

    tracks_by_frame = {int(f): grp for f, grp in t.groupby('frame')}

    with alive_bar(n_frames) as bar:
        for i in range(n_frames):
            cache = frame_cache[i]
            tx, ty = cache.get('translation', (0.0, 0.0))
            masks = cache['masks']
            m_cnt = cache['m_cnt']
            m_rad = cache['m_rad']
            orig_i = cache['orig_frame']

            frame_tracks = tracks_by_frame.get(orig_i, pd.DataFrame())
            daughters_viz = []
            frame_stats_list = []
            frame_masks_list = []

            if not frame_tracks.empty and m_cnt is not None:
                m_cnt_t = m_cnt.reshape(-1, 2).astype(float)
                m_cnt_t[:, 0] += tx
                m_cnt_t[:, 1] += ty
                m_pts = m_cnt_t

                d_contours = {}
                valid_tracks = []
                
                for _, row in frame_tracks.iterrows():
                    lbl = int(row['label_orig'])
                    cnt = get_contour_points(masks == lbl)
                    if len(cnt) > 0:
                        cnt = cnt.astype(float)
                        cnt[:, 0] += tx
                        cnt[:, 1] += ty
                        d_contours[lbl] = cnt
                        valid_tracks.append(row)

                if not valid_tracks:
                    final_results['regions'].append([])
                    final_results['stats'].append([])
                    bar()
                    continue

                status_map = {
                    int(row['particle']): {
                        'touch_mother': False, 'touch_daughter': False,
                        'mother_contact_roc': None, 'mother_contact_contour': None,
                        'daughter_contacts': {}, 'daughter_contact_contours': {},
                        'label': int(row['label_orig']), 'row': row
                    }
                    for row in valid_tracks
                }

                # A. Mother Contact Vector Extraction Pass (Isolating the local arc layout)
                for tid, s in status_map.items():
                    d_cnt = d_contours[s['label']]
                    min_dist, d_idx, _ = kdist(m_pts, d_cnt)
                    if min_dist < CONTACT_PX:
                        s['touch_mother'] = True
                        s['mother_contact_roc'] = curvature_radius(d_cnt, d_idx)
                        
                        w_start = max(0, d_idx - 5)
                        w_end = min(len(d_cnt), d_idx + 6)
                        s['mother_contact_contour'] = d_cnt[w_start:w_end]

                # B. Pairwise Daughter Contact Vector Extraction Pass via Dual KD-Tree Neighborhood Checking
                tids = list(status_map.keys())
                if len(tids) > 1:
                    kd_trees = {tid: cKDTree(d_contours[status_map[tid]['label']]) for tid in tids}
                    for k1 in range(len(tids)):
                        for k2 in range(k1 + 1, len(tids)):
                            id1, id2 = tids[k1], tids[k2]
                            cnt1 = d_contours[status_map[id1]['label']]
                            cnt2 = d_contours[status_map[id2]['label']]

                            dists, idxs = kd_trees[id2].query(cnt1, k=1)
                            min_i = np.argmin(dists)
                            if dists[min_i] < CONTACT_PX:
                                status_map[id1]['touch_daughter'] = True
                                status_map[id2]['touch_daughter'] = True
                                status_map[id1]['daughter_contacts'][id2] = curvature_radius(cnt1, min_i)
                                status_map[id2]['daughter_contacts'][id1] = curvature_radius(cnt2, int(idxs[min_i]))
                                
                                w1_start, w1_end = max(0, min_i - 5), min(len(cnt1), min_i + 6)
                                w2_start, w2_end = max(0, int(idxs[min_i]) - 5), min(len(cnt2), int(idxs[min_i]) + 6)
                                
                                status_map[id1]['daughter_contact_contours'][id2] = cnt1[w1_start:w1_end]
                                status_map[id2]['daughter_contact_contours'][id1] = cnt2[w2_start:w2_end]

                # C. Final Aggregation and Export Flattening Loop
                translated_mask = translate_mask(masks, tx, ty)
                for tid, s in status_map.items():
                    row = s['row']
                    d_cnt = d_contours[s['label']]
                    
                    if s['touch_mother'] and s['touch_daughter']:
                        color = (0, 255, 255)
                    elif s['touch_daughter']:
                        color = (147, 20, 255)
                    elif s['touch_mother']:
                        color = (0, 255, 0)
                    else:
                        color = (0, 0, 255)

                    daughters_viz.append({'label': s['label'], 'color': color, 'id': tid})

                    base_stats = {
                        "id": tid,
                        "frame": orig_i,

                        "area": float(row["area"]),

                        # Daughter centroid
                        "centroid_x": float(row["x"]),
                        "centroid_y": float(row["y"]),

                        # Full contour
                        "daughter_contour": d_cnt.tolist(),
                    }

                    # Handle MLS (Mother-Daughter Local Contact Space)
                    if s['touch_mother'] and s['mother_contact_contour'] is not None:
                        mid_idx = len(s['mother_contact_contour']) // 2
                        cp_x, cp_y = s['mother_contact_contour'][mid_idx]
                        
                        mls_stats = base_stats.copy()
                        mls_stats.update({

                            "contact_class": "MLS",

                            "contact_point_x": float(cp_x),
                            "contact_point_y": float(cp_y),

                            "interacting_with_id": np.nan
                        })
                        frame_stats_list.append(mls_stats)

                    # Handle MSS (Daughter-Daughter Local Contact Space)
                    if s['touch_daughter'] and s['daughter_contact_contours']:
                        for other_tid, c_seg in s['daughter_contact_contours'].items():
                            if c_seg is not None and len(c_seg) > 0:
                                mid_idx = len(c_seg) // 2
                                cp_x, cp_y = c_seg[mid_idx]
                                
                                mss_stats = base_stats.copy()
                                mss_stats.update({
                                    "contact_class": "MSS",
                                    "contact_point_x": float(cp_x),
                                    "contact_point_y": float(cp_y),
                                    "interacting_with_id": other_tid
                                })
                                frame_stats_list.append(mss_stats)

                    # Fallback entry line for free components
                    if not s['touch_mother'] and not s['touch_daughter']:
                        free_stats = base_stats.copy()
                        free_stats.update({
                            'contact_class': 'FREE',
                            'contact_point_x': np.nan, 'contact_point_y': np.nan
                        })
                        frame_stats_list.append(free_stats)

                    daughter_binary = (translated_mask == s['label'])
                    frame_masks_list.append(daughter_binary)

                    if save_training:
                        save_binary_daughter_sample(
                            os.path.join(output_folder, "unet_training_daughters_inner"),
                            cache['img'], daughter_binary, i, int(tid)
                        )

            if save_images:
                save_visuals_robust(
                    os.path.join(output_folder, "tracked_output_inner"), cache['fname'],
                    cv2.normalize(cache['img'], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                    masks, m_cnt, m_rad, daughters_viz
                )

            final_results['regions'].append(frame_masks_list)
            final_results['stats'].append(frame_stats_list)
            bar()

    return final_results