import numpy as np
import pandas as pd
from typing import List, Dict, Any

def load_csv(file_path: str) -> pd.DataFrame:
    """Loads raw CSV target profiles extracted from optical tracking environments."""
    return pd.read_csv(file_path)

def normalize_vesicle_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies initial surface area conversions and scales absolute timeline 
    ranges down to fractional increments.
    """
    df_norm = df.copy()
    
    # Extract initial mother area benchmark safely
    init_mother = df_norm['mother_area'].max()
    
    df_norm['mother_norm'] = df_norm['mother_area'] / init_mother
    df_norm['bud_norm'] = df_norm['bud_area'] / init_mother
    
    # Scale total tracking intervals down to 0-1 values
    max_t = df_norm['time_min'].max()
    df_norm['time_norm'] = df_norm['time_min'] / max_t if max_t > 0 else 0.0
    
    return df_norm

def bin_vesicle_data(df_norm: pd.DataFrame) -> pd.DataFrame:
    """
    Groups experimental timelines into regular minute-long buckets based on 
    the raw time tracking column values.
    """
    df_bin = df_norm.copy()
    df_bin['bin'] = np.floor(df_bin['time_min']).astype(int)
    
    # Group operations collapse values by tracking interval buckets
    binned = df_bin.groupby('bin').agg(
        time_norm_mean=('time_norm', 'mean'),
        mother_norm_mean=('mother_norm', 'mean'),
        mother_norm_std=('mother_norm', 'std'),
        bud_norm_mean=('bud_norm', 'mean'),
        bud_norm_std=('bud_norm', 'std')
    ).reset_index()
    
    # Fill in zero values if only a single instance occupied a specific time bucket
    binned['mother_norm_std'] = binned['mother_norm_std'].fillna(0.0)
    binned['bud_norm_std'] = binned['bud_norm_std'].fillna(0.0)
    
    return binned

def aggregate_condition(vesicle_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Combines individual tracking evaluations from matching concentration levels 
    to determine cross-sample mean tracking values.
    """
    if not vesicle_dfs:
        return pd.DataFrame()
        
    combined = pd.concat(vesicle_dfs, ignore_index=True)
    
    # Regroup data frames across standard time interval bins
    summary = combined.groupby('bin').agg(
        time_norm=('time_norm_mean', 'mean'),
        mother_mean=('mother_norm_mean', 'mean'),
        mother_std=('mother_norm_mean', 'std'),
        bud_mean=('bud_norm_mean', 'mean'),
        bud_std=('bud_norm_mean', 'std')
    ).reset_index()
    
    summary['mother_std'] = summary['mother_std'].fillna(0.0)
    summary['bud_std'] = summary['bud_std'].fillna(0.0)
    
    return summary