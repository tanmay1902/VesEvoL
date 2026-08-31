import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def select_frame_range(frames):
    """
    Interactive tool to select first and last frame using sliders.

    Parameters:
        frames (list or np.array): List of images (H x W or H x W x 3)

    Returns:
        (start_idx, end_idx)
    """

    num_frames = len(frames)

    # Initial indices
    start_idx = 0
    end_idx = num_frames - 1

    # Setup figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    plt.subplots_adjust(bottom=0.25)

    ax_start, ax_end = axes

    # Display initial frames
    img_start = ax_start.imshow(frames[start_idx], cmap='gray')
    ax_start.set_title(f"Start Frame: {start_idx}")

    img_end = ax_end.imshow(frames[end_idx], cmap='gray')
    ax_end.set_title(f"End Frame: {end_idx}")

    # Sliders
    ax_slider_start = plt.axes([0.15, 0.1, 0.7, 0.03])
    ax_slider_end = plt.axes([0.15, 0.05, 0.7, 0.03])

    slider_start = Slider(ax_slider_start, 'Start', 0, num_frames - 1, valinit=start_idx, valstep=1)
    slider_end = Slider(ax_slider_end, 'End', 0, num_frames - 1, valinit=end_idx, valstep=1)

    def update(val):
        s = int(slider_start.val)
        e = int(slider_end.val)

        # Ensure valid range
        if s > e:
            return

        img_start.set_data(frames[s])
        ax_start.set_title(f"Start Frame: {s}")

        img_end.set_data(frames[e])
        ax_end.set_title(f"End Frame: {e}")

        fig.canvas.draw_idle()

    slider_start.on_changed(update)
    slider_end.on_changed(update)

    plt.show()

    return int(slider_start.val), int(slider_end.val)