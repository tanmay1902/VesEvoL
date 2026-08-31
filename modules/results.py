import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from skimage.measure import perimeter as calc_perimeter
import os

def compile_experiment_data(mother_results, daughter_results, micron_per_pixel, dt_seconds=1.0):
    # --- 1. MOTHER DATA ---
    m_rad_px = np.array(mother_results.get("radius_px", []))
    if len(m_rad_px) == 0: return pd.DataFrame(), pd.DataFrame()

    m_rad_um = m_rad_px * micron_per_pixel
    
    mother_df = pd.DataFrame({
        "frame": range(len(m_rad_px)),
        "time_sec": np.arange(len(m_rad_px)) * dt_seconds,
        "radius_um": m_rad_um,
        "area_um2": np.pi * (m_rad_um ** 2),
        "surface_area_um2": 4 * np.pi * (m_rad_um ** 2)
    })

    # --- 2. DAUGHTER DATA ---
    daughter_records = []
    
    for frame_idx, (frame_stats, frame_regions) in enumerate(zip(daughter_results["stats"], daughter_results["regions"])):
        time_point = frame_idx * dt_seconds
        
        for i, d in enumerate(frame_stats):
            # Calculate Perimeter (if mask exists)
            perim_px = np.nan
            if not d.get("is_ghost", False) and i < len(frame_regions):
                perim_px = calc_perimeter(frame_regions[i])
            
            area_um2 = d["area"]
            perim_um = perim_px * micron_per_pixel if not np.isnan(perim_px) else np.nan
            radius_um = np.sqrt(area_um2 / np.pi)

            daughter_records.append({
                "frame": frame_idx,
                "time_sec": time_point,
                "id": d["id"],
                "radius_um": radius_um,
                "area_um2": area_um2,
                "surface_area_um2": 4 * area_um2, # Sphere approx
                "perimeter_um": perim_um,
                "touching_mother": d["touching_mother"],
                "touching_daughter": d.get("touching_daughter", False)
            })
            
    daughter_df = pd.DataFrame(daughter_records)
    return mother_df, daughter_df

def filter_tracks(daughter_df, min_frames=15, min_radius_um=0.2):
    if daughter_df.empty: return daughter_df
    
    # 1. Lifetime Filter
    lifetimes = daughter_df.groupby("id")["frame"].count()
    valid_ids = lifetimes[lifetimes >= min_frames].index
    clean = daughter_df[daughter_df["id"].isin(valid_ids)]
    
    # 2. Size Filter
    clean = clean[clean["radius_um"] >= min_radius_um]
    return clean

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from skimage.measure import perimeter as calc_perimeter
import os

def compile_experiment_data(mother_results, daughter_results, micron_per_pixel, dt_seconds=1.0):
    # [Same Mother Data Logic as before...]
    m_rad_px = np.array(mother_results.get("radius_px", []))
    if len(m_rad_px) == 0: return pd.DataFrame(), pd.DataFrame()
    m_rad_um = m_rad_px * micron_per_pixel
    mother_df = pd.DataFrame({
        "frame": range(len(m_rad_px)),
        "time_sec": np.arange(len(m_rad_px)) * dt_seconds,
        "radius_um": m_rad_um,
        "area_um2": np.pi * (m_rad_um ** 2),
        "surface_area_um2": 4 * np.pi * (m_rad_um ** 2)
    })

    # --- DAUGHTER DATA ---
    daughter_records = []
    zip_iter = zip(daughter_results["stats"], daughter_results["regions"])
    
    for frame_idx, (frame_stats, frame_regions) in enumerate(zip_iter):
        time_point = frame_idx * dt_seconds
        
        for i, d in enumerate(frame_stats):
            perim_px = np.nan
            if not d.get("is_ghost", False) and i < len(frame_regions):
                perim_px = calc_perimeter(frame_regions[i])
            
            area_um2 = d["area"]
            perim_um = perim_px * micron_per_pixel if not np.isnan(perim_px) else np.nan
            radius_um = np.sqrt(area_um2 / np.pi)

            daughter_records.append({
                "frame": frame_idx,
                "time_sec": time_point,
                "id": d["id"],
                "radius_um": radius_um,
                "area_um2": area_um2,
                "surface_area_um2": 4 * area_um2,
                "perimeter_um": perim_um,
                # NEW FIELDS
                "touching_mother": d["touching_mother"],
                "touching_daughter": d.get("touching_daughter", False),
                "contact_curvature": d.get("contact_curvature", np.nan)
            })
            
    daughter_df = pd.DataFrame(daughter_records)
    return mother_df, daughter_df

# [Keep filter_tracks, save_experiment_data, plot_experiment_dashboard as they were]
def filter_tracks(daughter_df, min_frames=15, min_radius_um=0.2):
    if daughter_df.empty: return daughter_df
    lifetimes = daughter_df.groupby("id")["frame"].count()
    valid_ids = lifetimes[lifetimes >= min_frames].index
    clean = daughter_df[daughter_df["id"].isin(valid_ids)]
    return clean[clean["radius_um"] >= min_radius_um]

def save_experiment_data(mother_df, daughter_df, output_folder="results"):
    os.makedirs(output_folder, exist_ok=True)
    mother_df.to_csv(os.path.join(output_folder, "mother_stats.csv"), index=False)
    daughter_df.to_csv(os.path.join(output_folder, "daughter_stats.csv"), index=False)
    if not daughter_df.empty:
        daughter_df.pivot(index="time_sec", columns="id", values="radius_um").to_csv(os.path.join(output_folder, "daughter_radius_matrix.csv"))

def plot_experiment_dashboard(mother_df, daughter_df, save_path=None):
    sns.set_style("whitegrid")
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    axs[0,0].plot(mother_df["time_sec"], mother_df["radius_um"], color="navy"); axs[0,0].set_title("Mother Radius")
    
    if not daughter_df.empty:
        counts = daughter_df.groupby("time_sec")["id"].nunique().reindex(mother_df["time_sec"], fill_value=0)
        axs[0,1].step(counts.index, counts.values, color="green"); axs[0,1].set_title("Daughter Count")
                #sns.lineplot(data=daughter_df, x="time_sec", y="radius_um", hue="id", legend=False, ax=axs[1,0]); axs[1,0].set_title("Growth")
        # --- Panel 3: Total daughter area over time ---
        total_area = (
            daughter_df
            .groupby("time_sec")["area_um2"]
            .sum()
            .reindex(mother_df["time_sec"], fill_value=0)
        )

        axs[1,0].plot(
            total_area.index,
            total_area.values,
            color="purple",
            linewidth=2
        )
        axs[1,0].set_title("Total Daughter Area")
        axs[1,0].set_xlabel("Time (s)")
        axs[1,0].set_ylabel("Σ Area (µm²)")


        # Panel 4: Curvature over time (Mean contact curvature)
        if "contact_curvature" in daughter_df.columns:
            sns.scatterplot(data=daughter_df[daughter_df["contact_curvature"] > 0], 
                            x="time_sec", y="contact_curvature", hue="id", ax=axs[1,1])
            axs[1,1].set_title("Contact Curvature (1/um)")
            
    if save_path: plt.savefig(save_path)
