from matplotlib.widgets import RectangleSelector, Slider, Button
import matplotlib.pyplot as plt
import numpy as np

def select_rect_roi(frames):
    n = len(frames)
    idx = 0
    roi = {}

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.25)

    im = ax.imshow(frames[0], cmap="gray")
    ax.set_title("Draw ROI → ENTER")

    # ---- Rectangle selector ----
    def onselect(eclick, erelease):
        roi["x0"] = int(eclick.xdata)
        roi["y0"] = int(eclick.ydata)
        roi["x1"] = int(erelease.xdata)
        roi["y1"] = int(erelease.ydata)

    rect = RectangleSelector(
        ax, onselect,
        interactive=True,
        useblit=False
    )

    # ---- Frame slider ----
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, "Frame", 0, n - 1, valinit=0, valstep=1)

    def update(val):
        nonlocal idx
        idx = int(val)
        im.set_data(frames[idx])
        ax.set_title(f"Draw ROI → ENTER | Frame {idx}/{n-1}")
        fig.canvas.draw_idle()

    slider.on_changed(update)

    # ---- Key handling ----
    def on_key(event):
        if event.key == "enter":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()

    if not roi:
        h, w = frames[0].shape
        return 0, h, 0, w

    y0, y1 = sorted([roi["y0"], roi["y1"]])
    x0, x1 = sorted([roi["x0"], roi["x1"]])

    h, w = frames[0].shape
    return max(0, y0), min(h, y1), max(0, x0), min(w, x1)


def select_initial_roi(frames):
    roi = {}

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(frames[0], cmap='gray')
    ax.set_title("Draw ROI (size will be locked) → ENTER")

    def onselect(eclick, erelease):
        roi["x0"] = int(eclick.xdata)
        roi["y0"] = int(eclick.ydata)
        roi["x1"] = int(erelease.xdata)
        roi["y1"] = int(erelease.ydata)

    rect = RectangleSelector(ax, onselect, interactive=True)

    def on_key(event):
        if event.key == "enter":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

    if not roi:
        raise RuntimeError("ROI not selected")

    y0, y1 = sorted([roi["y0"], roi["y1"]])
    x0, x1 = sorted([roi["x0"], roi["x1"]])

    return y0, y1, x0, x1

from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt

def drag_fixed_roi(frames, roi_shape, start_frame):
    H, W = roi_shape
    n = len(frames)
    idx = start_frame

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.25)

    im = ax.imshow(frames[idx], cmap="gray")
    ax.set_title("Drag ROI → ENTER")

    h_img, w_img = frames[0].shape
    rect = Rectangle((0, 0), W, H, edgecolor="red", fill=False)
    ax.add_patch(rect)

    dragging = False

    def clamp(x, y):
        x = max(0, min(w_img - W, x))
        y = max(0, min(h_img - H, y))
        return x, y

    def on_press(event):
        nonlocal dragging
        if event.inaxes == ax:
            dragging = True

    def on_release(event):
        nonlocal dragging
        dragging = False

    def on_move(event):
        if not dragging or event.xdata is None:
            return
        x, y = clamp(int(event.xdata - W//2), int(event.ydata - H//2))
        rect.set_xy((x, y))
        fig.canvas.draw_idle()

    # ---- Frame slider ----
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, "Frame", start_frame, n - 1,
                    valinit=start_frame, valstep=1)

    def update(val):
        nonlocal idx
        idx = int(val)
        im.set_data(frames[idx])
        ax.set_title(f"Drag ROI → ENTER | Frame {idx}/{n-1}")
        fig.canvas.draw_idle()

    slider.on_changed(update)

    def on_key(event):
        if event.key == "enter":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()

    x0, y0 = map(int, rect.get_xy())
    return y0, y0 + H, x0, x0 + W, idx

from matplotlib.widgets import Slider, Button
import matplotlib.pyplot as plt

def select_frame_range_gui(frames, start, max_frame, roi):
    """
    frames : full frames list
    start  : starting frame index
    max_frame : last frame index
    roi    : (y0, y1, x0, x1)
    """
    y0, y1, x0, x1 = roi
    end = start

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.3)

    im = ax.imshow(frames[start][y0:y1, x0:x1], cmap="gray")
    ax.set_title(f"Select END frame → CONFIRM\nFrame {start}")

    # ---- Slider ----
    ax_slider = plt.axes([0.2, 0.15, 0.6, 0.05])
    slider = Slider(
        ax_slider,
        "End",
        start,
        max_frame,
        valinit=start,
        valstep=1
    )

    def update(val):
        nonlocal end
        end = int(val)
        im.set_data(frames[end][y0:y1, x0:x1])
        ax.set_title(f"Select END frame → CONFIRM\nFrame {end}")
        fig.canvas.draw_idle()

    slider.on_changed(update)

    # ---- Confirm ----
    ax_btn = plt.axes([0.4, 0.05, 0.2, 0.08])
    btn = Button(ax_btn, "CONFIRM")

    btn.on_clicked(lambda _: plt.close(fig))

    plt.show()
    return start, end

def select_roi_per_frame_ranges(frames,fixed_size=None,returnsize=False):
    n = len(frames)
    cropped = [None] * n
    covered = np.zeros(n, dtype=bool)

    # ---- Initial ROI ----
    if fixed_size == None:
        y0, y1, x0, x1 = select_initial_roi(frames)
        H, W = y1 - y0, x1 - x0
    else:
        H, W = fixed_size
    while not covered.all():
        first_uncovered = np.where(~covered)[0][0]

        # ---- Drag ROI (fixed size) ----
        y0, y1, x0, x1, _ = drag_fixed_roi(
            frames, (H, W), first_uncovered
        )

        # ---- Select END frame ONLY ----
        _, end = select_frame_range_gui(
            frames,
            first_uncovered,
            n - 1,
            (y0, y1, x0, x1)
        )

        for i in range(first_uncovered, end + 1):
            cropped[i] = frames[i][y0:y1, x0:x1]
            covered[i] = True

        print(f"Applied ROI to frames {first_uncovered} → {end}")
    if returnsize:
        return cropped,(H,W)
    else:
        return cropped


from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
import numpy as np

def view_cropped_frames(frames, title="Cropped frames preview"):
    """
    frames : list of 2D numpy arrays (no None!)
    """

    # ---- Safety checks ----
    for i, f in enumerate(frames):
        if f is None:
            raise ValueError(f"Frame {i} is None — ROI selection incomplete")

    shapes = {f.shape for f in frames}
    if len(shapes) != 1:
        raise ValueError(f"Inconsistent frame shapes: {shapes}")

    n = len(frames)

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.2)

    im = ax.imshow(frames[0], cmap="gray")
    ax.set_title(f"{title}\nFrame 0/{n-1}")

    # ---- Slider ----
    ax_slider = plt.axes([0.2, 0.08, 0.6, 0.04])
    slider = Slider(
        ax_slider,
        "Frame",
        0,
        n - 1,
        valinit=0,
        valstep=1
    )

    def update(val):
        idx = int(val)
        im.set_data(frames[idx])
        ax.set_title(f"{title}\nFrame {idx}/{n-1}")
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()
