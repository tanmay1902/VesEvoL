import os
from typing import Union
import matplotlib.pyplot as plt
import matplotlib.axes as axes

def apply_style() -> None:
    """Applies the custom matplotlib style sheet globally."""
    style_path = os.path.join(os.path.dirname(__file__), 'mystyle.mplstyle')
    if os.path.exists(style_path):
        plt.style.use(style_path)
    else:
        plt.style.use('ggplot')  # Fallback gracefully if style missing

def square_axes(ax: axes.Axes) -> None:
    """Forces the plot viewport window to maintain a strict 1:1 aspect ratio."""
    ax.set_box_aspect(1.0)

def panel_label(ax: axes.Axes, label: str, x_offset: float = -0.18, y_offset: float = 1.05) -> None:
    """Inserts bolded panel identification tags (e.g., 'A', 'B') into specific subplots."""
    ax.text(x_offset, y_offset, label, transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top', ha='right')

def format_axis(ax: axes.Axes, x_label: str = "", y_label: str = "") -> None:
    """Cleans up bounding spine properties and inserts clean string label updates."""
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def set_minor_grid(ax: axes.Axes) -> None:
    """Enables subtle back-grid styling for data parsing tracking if requested."""
    ax.grid(True, which='both', color='#e5e5e5', linestyle=':', linewidth=0.5)

def save_pdf_png(fig: plt.Figure, name: str) -> None:
    """Saves high-resolution image targets in both vector PDF and pixel rasterized formats."""
    # Ensure file string doesn't duplicate extensions
    base_name = os.path.splitext(name)[0]
    fig.savefig(f"{base_name}.pdf", dpi=600, bbox_inches='tight')
    fig.savefig(f"{base_name}.png", dpi=600, bbox_inches='tight')
    fig.savefig(f"{base_name}.svg", format="svg", dpi=600, bbox_inches='tight')