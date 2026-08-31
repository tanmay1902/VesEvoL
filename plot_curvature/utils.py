import os
import glob
import numpy as np
import pandas as pd

TARGET_METRICS = ['MSS', 'MLS', 'B_low']

def parse_dataset_tree(root_dir):
    conditions = ["POPC", "POPC_Chol"]
    conc_mapping = {"1uM": "1uM", "1.5uM": "1.5uM", "2uM": "2uM", "2.5uM": "2.5uM"}
    
    data_tree = {}
    print(f"Scanning data root directory: {root_dir}")
    
    for cond in conditions:
        cond_folder = cond
        if not os.path.exists(os.path.join(root_dir, cond_folder)):
            alt_folder = cond.replace("_", "-")
            if os.path.exists(os.path.join(root_dir, alt_folder)):
                cond_folder = alt_folder
            else:
                alt_folder = cond.replace("_", ":")
                if os.path.exists(os.path.join(root_dir, alt_folder)):
                    cond_folder = alt_folder

        full_cond_path = os.path.join(root_dir, cond_folder)
        if not os.path.exists(full_cond_path):
            continue
            
        clean_cond_key = cond.replace("_", ":")
        data_tree[clean_cond_key] = {}
        
        for folder_conc, clean_conc in conc_mapping.items():
            search_pattern = os.path.join(full_cond_path, folder_conc, "V*", "curvature_time_series.csv")
            matching_files = glob.glob(search_pattern)
            
            print(f"Processing [ {clean_cond_key} | {clean_conc} ] - Found {len(matching_files)} vesicles.")
            if matching_files:
                data_tree[clean_cond_key][clean_conc] = matching_files
                
    return data_tree

def load_vesicle_trajectories(files):
    trajectories = []
    for f_idx, csv_path in enumerate(files):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if 'time_min' not in df.columns or len(df) == 0:
            continue
            
        df = df.sort_values(by='time_min')
        t_raw = df['time_min'].values
        t_max = t_raw[-1]
        if t_max <= 0:
            continue
            
        df_clean = df[['time_min'] + TARGET_METRICS + [f'd{m}' for m in TARGET_METRICS]].copy()
        df_clean['time_norm'] = df_clean['time_min'] / t_max
        df_clean['vesicle_name'] = f"V{f_idx + 1}"
        trajectories.append(df_clean)
        
    return trajectories

def bin_individual_vesicle_2min(df_v, metric, use_normalized=False):
    """
    Bins a single vesicle's time series into absolute 2-minute chunks.
    Calculates the internal average value and instrument error propagation 
    for that specific vesicle within each time block.
    """
    v_t = df_v['time_min'].values
    v_y = df_v[metric].values
    v_dy = df_v[f"d{metric}"].values if f"d{metric}" in df_v.columns else np.zeros_like(v_y)
    
    t_max = v_t[-1]
    bin_size = 2.0
    bins = np.arange(v_t.min(), v_t.max() + bin_size, bin_size)
    
    binned_t = []
    binned_y = []
    binned_dy = []
    
    for i in range(len(bins) - 1):
        mask = (v_t >= bins[i]) & (v_t < bins[i+1])
        valid_mask = mask & (~np.isnan(v_y))
        n_points = np.sum(valid_mask)
        
        if n_points > 0:
            center_t = 0.5 * (bins[i] + bins[i+1])
            if use_normalized:
                center_t = center_t / t_max
                
            binned_t.append(center_t)
            binned_y.append(np.mean(v_y[valid_mask]))
            
            # Root-mean-square propagation of structural instrumentation uncertainty inside the bin
            binned_dy.append(np.sqrt(np.sum(v_dy[valid_mask]**2)) / n_points)
            
    return np.array(binned_t), np.array(binned_y), np.array(binned_dy)