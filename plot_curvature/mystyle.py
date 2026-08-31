import os
import matplotlib.pyplot as plt

CONC_COLORS = {
    "1uM": "#E66101",
    "1.5uM": "#FDB863",
    "2uM": "#B2ABD2",
    "2.5uM": "#5E3C99"
}
GRAY_INDIVIDUAL = "#7F7F7F"

def apply_custom_style():
    style_path = "mystyle.mplstyle"
    if os.path.exists(style_path):
        plt.style.use(style_path)
    else:
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans'],
            'axes.spines.top': False,
            'axes.spines.right': False,
            'xtick.direction': 'out',
            'ytick.direction': 'out'
        })

def configure_legend(ax):
    # Only try to create a legend if there are actually labeled handles present
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    leg = ax.legend(
        loc='upper right', fontsize=7, frameon=True, fancybox=True,
        framealpha=0.75, edgecolor='0.7', handlelength=1.2,
        borderpad=0.3, labelspacing=0.3, markerscale=0.9
    )
    if leg:
        leg.get_frame().set_facecolor('white')
    return leg

def save_publication_figure(fig, base_filename, output_dir="output_figures"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    path_base = os.path.join(output_dir, base_filename)
    fig.savefig(f"{path_base}.svg", format="svg")
    fig.savefig(f"{path_base}.pdf", format="pdf")
    fig.savefig(f"{path_base}.png", format="png", dpi=600)
    print(f"Saved: {base_filename} (.svg, .pdf, .png 600dpi)")