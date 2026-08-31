from skimage.exposure import rescale_intensity
from skimage.filters import gaussian
from skimage import exposure, filters, morphology
import numpy as np
from scipy.ndimage import gaussian_filter
import cv2

def gamma_correction(frame,gamma=0.5):
    if frame.dtype != np.uint8:
        # Assuming the frame is scaled correctly (0-255) for this conversion.
        frame = frame.astype(np.uint8)
        
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	
    # apply gamma correction using the lookup table
    return cv2.LUT(frame, table)
    

def preprocess_phase_contrast(frame, clahe_clip=0.01):
    """
    Enhance phase contrast GUV frames (dark vesicles).
    Output: preprocessed grayscale image, same shape as input.
    """
    
    # --- Normalize to [0,1]
    frame = frame.astype(np.float32)
    frame -= frame.min()
    if frame.max() > 0:
        frame /= frame.max()
    #frame = 1 - frame

    # --- CLAHE for local contrast
    frame = exposure.equalize_adapthist(frame, clip_limit=clahe_clip)
    return frame
    

def daughter_preprocess(frame):
    #frame = preprocess_phase_contrast(frame)
    #frame = cv2.bitwise_not(frame)
    return frame.astype(np.uint8)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
def preprocess_gui(frames):
    frames = np.asarray(frames, dtype=np.float32)

    if frames.ndim != 3:
        raise ValueError("Expected frames with shape (T, H, W)")

    n_frames, H, W = frames.shape

    # 🔒 Immutable original
    original_frames = frames.copy()

    # 🧱 Applied baseline (changes only on APPLY)
    applied_frames = original_frames.copy()

    # 🎚️ Preview parameters
    state = {
        "brightness": 0.0,
        "contrast": 1.0,
        "gamma": 1.0
    }

    # ---------------- Processing ----------------
    def process(img, brightness, contrast, gamma):
        out = img * contrast + brightness
        out = np.clip(out, 0, None)

        if gamma != 1.0:
            m = out.max()
            if m > 0:
                out = (out / m) ** gamma * m
        return out

    # ---------------- Figure layout ----------------
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.7, 0.3])

    ax_img = fig.add_subplot(gs[0])
    img_disp = ax_img.imshow(applied_frames[0], cmap="gray")
    ax_img.axis("off")

    gs_r = gs[1].subgridspec(100, 1)
    ax_hist = fig.add_subplot(gs_r[:20])
    ax_ctrl = fig.add_subplot(gs_r[20:90])
    ax_btn = fig.add_subplot(gs_r[90:])

    ax_ctrl.axis("off")
    ax_btn.axis("off")

    # ---------------- Histogram ----------------
    def update_hist(img):
        ax_hist.clear()
        ax_hist.hist(img.ravel(), bins=256, color="black")
        ax_hist.set_yticks([])
        ax_hist.set_title("Histogram")

    update_hist(applied_frames[0])

    # ---------------- Sliders ----------------
    ax_frame = plt.axes([0.15, 0.05, 0.45, 0.03])
    s_frame = Slider(
        ax_frame, "Frame",
        valmin=0, valmax=n_frames - 1,
        valinit=0, valstep=1
    )

    ax_b = plt.axes([0.75, 0.65, 0.2, 0.03])
    s_b = Slider(
        ax_b, "Brightness",
        valmin=-1, valmax=1, valinit=0
    )

    ax_c = plt.axes([0.75, 0.60, 0.2, 0.03])
    s_c = Slider(
        ax_c, "Contrast",
        valmin=0.1, valmax=5.0, valinit=1.0
    )

    ax_g = plt.axes([0.75, 0.55, 0.2, 0.03])
    s_g = Slider(
        ax_g, "Gamma",
        valmin=0.2, valmax=3.0, valinit=1.0
    )

    # ---------------- Preview Update ----------------
    def update(_=None):
        i = int(s_frame.val)

        preview = process(
            applied_frames[i],
            s_b.val,
            s_c.val,
            s_g.val
        )

        img_disp.set_data(preview)
        ax_img.set_title(f"Frame {i} (preview)")
        update_hist(preview)
        fig.canvas.draw_idle()

    for s in (s_frame, s_b, s_c, s_g):
        s.on_changed(update)

    # ---------------- Buttons ----------------
    ax_auto = plt.axes([0.60, 0.01, 0.08, 0.05])
    ax_apply = plt.axes([0.69, 0.01, 0.08, 0.05])
    ax_ok = plt.axes([0.78, 0.01, 0.08, 0.05])
    ax_cancel = plt.axes([0.87, 0.01, 0.08, 0.05])

    btn_auto = Button(ax_auto, "Auto")
    btn_apply = Button(ax_apply, "Apply")
    btn_ok = Button(ax_ok, "OK")
    btn_cancel = Button(ax_cancel, "Cancel")

    # ---------------- Auto (ImageJ-like) ----------------
    def auto_contrast(event):
        i = int(s_frame.val)
        img = applied_frames[i]
        lo, hi = np.percentile(img, (1, 99))

        s_b.set_val(-lo)
        s_c.set_val(1.0 / (hi - lo + 1e-6))

    btn_auto.on_clicked(auto_contrast)

    # ---------------- Apply ----------------
    def on_apply(event):
        nonlocal applied_frames
        for i in range(n_frames):
            applied_frames[i] = process(
                applied_frames[i],
                s_b.val,
                s_c.val,
                s_g.val
            )

        # reset preview sliders
        s_b.set_val(0)
        s_c.set_val(1)
        s_g.set_val(1)

        update()

    btn_apply.on_clicked(on_apply)

    # ---------------- OK / Cancel ----------------
    def on_ok(event):
        plt.close(fig)

    def on_cancel(event):
        nonlocal applied_frames
        applied_frames = original_frames.copy()
        plt.close(fig)

    btn_ok.on_clicked(on_ok)
    btn_cancel.on_clicked(on_cancel)

    plt.show()

    return applied_frames
