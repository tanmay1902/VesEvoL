import numpy as np
import matplotlib.pyplot as plt  # NO 'Agg' backend - keeps interactive plots working
import cv2
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import least_squares
from scipy.interpolate import splprep, splev  # For smooth contours


# Curvature Plots


def interpolate_contour(contour, num_points=500):
    """Smooths contour - fills gaps, makes continuous curve"""
    # Handle cv2 contour format
    if contour.ndim == 3 and contour.shape[1] == 1:
        contour = np.squeeze(contour, axis=1)
  
    x = contour[:, 0]
    y = contour[:, 1]
  
    # Close contour if needed
    if not (np.isclose(x[0], x[-1]) and np.isclose(y[0], y[-1])):
        x = np.append(x, x[0])
        y = np.append(y, y[0])
  
    try:
        tck, u = splprep([x, y], s=0.01, per=True)  # Slightly relaxed smoothing
        u_new = np.linspace(0, 1, num_points)
        x_new, y_new = splev(u_new, tck)
        return np.column_stack([x_new, y_new])
    except:
        return contour

def compute_curvature(contour, remove_outliers=True, interpolate=True, num_points=500):
    """
        contour : numpy array
        Can be either:
        - Nx2 array (standard format)
        - Nx1x2 array (cv2.findContours format) - will be automatically reshaped
    """

    # Handle cv2 contour format (N, 1, 2) -> reshape to (N, 2)
    if contour.ndim == 3 and contour.shape[1] == 1:
        contour = np.squeeze(contour, axis=1)
  
    # SMOOTH CONTOUR - NO GAPS
    if interpolate and len(contour) < num_points:
        contour = interpolate_contour(contour, num_points)
  
    # Extract x and y coordinates
    x_s = contour[:, 0]
    y_s = contour[:, 1]
  
    # Curvature Computation
    dx = np.gradient(x_s)
    dy = np.gradient(y_s)
    speed_sq = dx**2 + dy**2
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
  
    # Add numerical stability
    kappa = (dx * ddy - dy * ddx) / np.clip(speed_sq, 1e-12, None)**(3/2)
  
    # FIXED Arc length - matches kappa length exactly using gradient speed
    speed = np.sqrt(speed_sq)
    ds = speed / np.mean(speed) * (np.mean(np.diff(x_s)**2 + np.diff(y_s)**2)**0.5)  # Normalize to physical units
    s = np.concatenate(([0], np.cumsum(ds[:-1])))
  
    # Outlier Removal - FIXED for smooth shapes
    kappa_clean = kappa.copy()
  
    if remove_outliers:
        medK = np.nanmedian(kappa)
        madK = np.nanmedian(np.abs(kappa - medK))
        
        # Skip outlier removal if too smooth (round shapes)
        if madK > 1e-6:
            thresh = 3
            mask = np.abs(kappa - medK) <= thresh * madK
        
            kappa_clean[~mask] = np.nan
        
            # Linear interpolation for missing values
            nans = np.isnan(kappa_clean)
            if np.any(nans):
                x_indices = np.arange(len(kappa_clean))
                kappa_clean[nans] = np.interp(x_indices[nans], x_indices[~nans], kappa_clean[~nans])
  
    return kappa, kappa_clean, s, x_s, y_s

def plot_curvature_analysis(kappa, kappa_clean, s, x_s, y_s, save_path=None):
  
    # FIXED Normalization - handles zero/constant curvature safely
    kappa_max = np.max(np.abs(kappa_clean))
    if kappa_max < 1e-8:
        # Nearly constant/zero curvature (perfect circle) - show raw variation
        kappa_max = np.max(np.abs(kappa))
    kappa_max = max(kappa_max, 1e-10)  # Safety
    
    kappa_normalized = kappa_clean / kappa_max
    kappa_raw_normalized = kappa / kappa_max
  
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
  
    # Plot 1: Normalized curvature vs index
    axes[0, 0].plot(kappa_raw_normalized, 'r--', linewidth=1, label='Raw')
    axes[0, 0].plot(kappa_normalized, 'b-', linewidth=1.5, label='Cleaned')
    axes[0, 0].legend()
    axes[0, 0].set_title('Normalized curvature along contour index')
    axes[0, 0].set_xlabel('Contour index')
    axes[0, 0].set_ylabel(r'Normalized $\kappa$ [-1, 1]')
    axes[0, 0].set_ylim(-1.1, 1.1)
    axes[0, 0].grid(True, alpha=0.3)
  
    # Plot 2: Normalized curvature vs arc length (with raw) - FIXED lengths match
    axes[0, 1].plot(s[:len(kappa_raw_normalized)], kappa_raw_normalized, 'r--', linewidth=1, label='Raw')
    axes[0, 1].plot(s[:len(kappa_normalized)], kappa_normalized, 'b-', linewidth=1.5, label='Cleaned')
    axes[0, 1].legend()
    axes[0, 1].set_title('Normalized curvature vs arc length')
    axes[0, 1].set_xlabel('Arc length s (µm)')
    axes[0, 1].set_ylabel(r'Normalized $\kappa$ [-1, 1]')
    axes[0, 1].set_ylim(-1.1, 1.1)
    axes[0, 1].grid(True, alpha=0.3)
  
    # Plot 3: Cleaned normalized curvature vs arc length - FIXED
    axes[1, 0].plot(s[:len(kappa_normalized)], kappa_normalized, 'b-', linewidth=1.5, label='Cleaned Curvature')
    axes[1, 0].legend()
    axes[1, 0].set_title('Normalized curvature vs arc length')
    axes[1, 0].set_xlabel('Arc length s (µm)')
    axes[1, 0].set_ylabel(r'Normalized $\kappa$ [-1, 1]')
    axes[1, 0].set_ylim(-1.1, 1.1)
    axes[1, 0].grid(True, alpha=0.3)
  
    # Plot 4: Spatial variation colormap with normalized curvature
    scatter = axes[1, 1].scatter(x_s, y_s, s=25, c=kappa_normalized, cmap='jet', vmin=-1, vmax=1)
    axes[1, 1].axis('equal')
    cbar = plt.colorbar(scatter, ax=axes[1, 1])
    cbar.set_label(r'Normalized $\kappa$ [-1, 1]')
    axes[1, 1].set_title('Spatial curvature variation (normalized)')
    axes[1, 1].set_xlabel('x (µm)')
    axes[1, 1].set_ylabel('y (µm)')
  
    plt.tight_layout()
  
    if save_path:
        plt.savefig(save_path, format='jpeg', bbox_inches='tight', dpi=150)
        plt.close(fig)
    else:
        plt.show()

