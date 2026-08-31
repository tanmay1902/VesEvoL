import numpy as np
from skimage.restoration import richardson_lucy, wiener
from alive_progress import alive_bar
from modules.read_czi_tiff import read_wavelength_czi, read_na_czi
from matplotlib.widgets import Slider, Button, RadioButtons
from skimage.restoration import richardson_lucy, wiener
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from skimage.restoration import richardson_lucy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from skimage.restoration import richardson_lucy, wiener
from stardist.models import StarDist2D
from csbdeep.utils import normalize

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from alive_progress import alive_bar
def deconvolve_frames(frames, psf, iterations=5, method="RL", balance=1e-4, background=0.0):
    out = []
    with alive_bar(len(frames), title="Deconvulating frames: ") as deconvbar:
        for i, f in enumerate(frames):
            if f is None:
                raise ValueError(f"Frame {i} is None")

            img = f.astype(np.float32)
            img -= img.min()
            img /= img.max() + 1e-6

            # background subtraction
            if background > 0:
                img = np.clip(img - background, 0, None)

            if method == "RL":
                deconv = richardson_lucy(img, psf, iterations, clip=True)
            elif method == "Wiener":
                deconv = wiener(img, psf, balance=balance)
            else:
                raise ValueError("Unknown deconvolution method")

            out.append(deconv.astype(np.float32))
            deconvbar()

        return out

def gaussian_psf(sigma_px, size):
    """
    sigma_px : PSF sigma in pixels
    size     : odd integer (support size)
    """
    size = int(size)
    if size % 2 == 0:
        size += 1

    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx**2 + yy**2) / (2.0 * sigma_px**2))
    psf /= psf.sum()
    return psf

def gaussian_psf_from_czi(px_um, wavelength_um=0.55, NA=1.4, size=21):
    # Diffraction-limited lateral sigma (Abbe approx)
    sigma_um = 0.21 * wavelength_um / NA
    sigma_px = sigma_um / px_um

    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx**2 + yy**2) / (2. * sigma_px**2))
    return psf / psf.sum()

def psf_from_czi(path, px_um, size=21):
    wl = read_wavelength_czi(path) or 0.55
    na = read_na_czi(path) or 1.4
    sigma_um = 0.21 * wl / na      # Abbe
    sigma_px = sigma_um / px_um

    ax = np.arange(-size//2 + 1., size//2 + 1.)
    xx, yy = np.meshgrid(ax, ax)

    psf = np.exp(-(xx**2 + yy**2) / (2 * sigma_px**2))
    return psf / psf.sum()



def tune_psf_gui_old(frame, px_um):
    frame = frame.astype(np.float32)
    frame -= frame.min()
    frame /= frame.max() + 1e-6

    # ---- SAFE defaults for px_um ≈ 0.227 ----
    sigma_px = 2.0
    size = 25
    iterations = 3
    background = 0.02
    balance_log10 = -4.0
    gamma = 1.0
    method = "RL"

    def apply_deconv():
        img = np.clip(frame - background, 0, None)
        psf = gaussian_psf(sigma_px, size)

        if method == "RL":
            out = richardson_lucy(img, psf, iterations, clip=True)
        else:
            out = wiener(img, psf, balance=10**balance_log10, clip=True)

        return out

    out = apply_deconv()

    # ---- Figure ----
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
    plt.subplots_adjust(bottom=0.45)

    im0 = ax0.imshow(frame, cmap="gray")
    ax0.set_title("Original")

    im1 = ax1.imshow(out**gamma, cmap="gray")
    ax1.set_title("Deconvolved")

    # ---- Sliders ----
    def add_slider(pos, label, vmin, vmax, init, step):
        ax = plt.axes(pos)
        return Slider(ax, label, vmin, vmax, valinit=init, valstep=step)

    s_sigma = add_slider([0.15, 0.35, 0.7, 0.03], "σ (px)", 1.2, 4.0, sigma_px, 0.1)
    s_size  = add_slider([0.15, 0.30, 0.7, 0.03], "PSF size", 15, 51, size, 2)
    s_iter  = add_slider([0.15, 0.25, 0.7, 0.03], "Iterations", 1, 8, iterations, 1)
    s_bg    = add_slider([0.15, 0.20, 0.7, 0.03], "Background", 0.0, 0.2, background, 0.005)
    s_bal   = add_slider([0.15, 0.15, 0.7, 0.03], "Wiener log10(balance)", -6, -2, balance_log10, 0.2)
    s_gamma = add_slider([0.15, 0.10, 0.7, 0.03], "Gamma (view)", 0.4, 2.0, gamma, 0.05)

    # ---- Method selector ----
    ax_radio = plt.axes([0.02, 0.15, 0.1, 0.18])
    radio = RadioButtons(ax_radio, ["RL", "Wiener"], active=0)

    def update(val=None):
        nonlocal sigma_px, size, iterations, background, balance_log10, gamma

        sigma_px = s_sigma.val
        size = int(s_size.val)
        iterations = int(s_iter.val)
        background = s_bg.val
        balance_log10 = s_bal.val
        gamma = s_gamma.val

        out = apply_deconv()
        im1.set_data(out**gamma)
        ax1.set_title(
            f"{method} | σ={sigma_px:.2f}px | size={size} | "
            f"iter={iterations} | bg={background:.3f}"
        )
        fig.canvas.draw_idle()

    for s in (s_sigma, s_size, s_iter, s_bg, s_bal, s_gamma):
        s.on_changed(update)

    def set_method(label):
        nonlocal method
        method = label
        update()

    radio.on_clicked(set_method)

    # ---- Confirm ----
    ax_btn = plt.axes([0.42, 0.02, 0.16, 0.06])
    btn = Button(ax_btn, "CONFIRM")

    result = {}

    def confirm(event):
        result.update(
            sigma_px=sigma_px,
            size=size,
            iterations=iterations,
            background=background,
            method=method,
            balance=10**balance_log10
        )
        plt.close(fig)

    btn.on_clicked(confirm)
    plt.show()

    return result

from skimage.color import label2rgb

def overlay_seg(img, labels, alpha=0.4):
    if labels is None or labels.max() == 0:
        return np.dstack([img]*3)
    return label2rgb(labels, image=img, alpha=alpha, bg_label=0)


def tune_psf_gui(frame, px_um):
    frame = frame.astype(np.float32)
    frame -= frame.min()
    frame /= frame.max() + 1e-6

    # ---- SAFE defaults ----
    sigma_px = 2.0
    size = 25
    iterations = 3
    background = 0.02
    balance_log10 = -4.0
    gamma = 1.0
    method = "RL"

    # ---- StarDist (single instance) ----
    stardist_model = StarDist2D.from_pretrained("2D_versatile_fluo")

    def apply_deconv():
        img = np.clip(frame - background, 0, None)
        psf = gaussian_psf(sigma_px, size)

        if method == "RL":
            out = richardson_lucy(img, psf, iterations, clip=True)
        else:
            out = wiener(img, psf, balance=10**balance_log10, clip=True)

        return out

    def run_stardist(img):
        img_norm = normalize(img, 1, 99.8)
        try:
            labels, _ = stardist_model.predict_instances(
                img_norm,
                prob_thresh=0.5,
                nms_thresh=0.4
            )
            return labels
        except Exception as e:
            print("StarDist failed:", e)
            return None

    # ---- Initial ----
    deconv = apply_deconv()
    sd_labels = run_stardist(deconv)

    # ---- Figure (4 rows) ----
    fig, axs = plt.subplots(1, 4, figsize=(14, 8))
    plt.subplots_adjust(bottom=0.45)

    axs[0].imshow(frame, cmap="gray")
    axs[0].set_title("1. Raw")

    axs[1].imshow(deconv**gamma, cmap="gray")
    axs[1].set_title("2. Deconvolved")

    axs[2].imshow(sd_labels if sd_labels is not None else np.zeros_like(frame),
                  cmap="tab20")
    axs[2].set_title("3. Deconv + StarDist (labels)")

    axs[3].imshow(overlay_seg(deconv, sd_labels), cmap="gray")
    axs[3].set_title("4. Deconv + StarDist (overlay)")

    for ax in axs:
        ax.axis("off")

    # ---- Sliders ----
    def add_slider(pos, label, vmin, vmax, init, step):
        ax = plt.axes(pos)
        return Slider(ax, label, vmin, vmax, valinit=init, valstep=step)

    s_sigma = add_slider([0.15, 0.35, 0.7, 0.03], "σ (px)", 1.2, 4.0, sigma_px, 0.1)
    s_size  = add_slider([0.15, 0.30, 0.7, 0.03], "PSF size", 15, 51, size, 2)
    s_iter  = add_slider([0.15, 0.25, 0.7, 0.03], "Iterations", 1, 8, iterations, 1)
    s_bg    = add_slider([0.15, 0.20, 0.7, 0.03], "Background", 0.0, 0.2, background, 0.005)
    s_bal   = add_slider([0.15, 0.15, 0.7, 0.03], "Wiener log10(balance)", -6, -2, balance_log10, 0.2)
    s_gamma = add_slider([0.15, 0.10, 0.7, 0.03], "Gamma (view)", 0.4, 2.0, gamma, 0.05)

    # ---- Method selector ----
    ax_radio = plt.axes([0.02, 0.15, 0.1, 0.18])
    radio = RadioButtons(ax_radio, ["RL", "Wiener"], active=0)

    def update(val=None):
        nonlocal sigma_px, size, iterations, background, balance_log10, gamma

        sigma_px = s_sigma.val
        size = int(s_size.val)
        iterations = int(s_iter.val)
        background = s_bg.val
        balance_log10 = s_bal.val
        gamma = s_gamma.val

        deconv = apply_deconv()
        labels = run_stardist(deconv)

        axs[1].images[0].set_data(deconv**gamma)
        axs[2].images[0].set_data(labels if labels is not None else np.zeros_like(frame))
        axs[3].images[0].set_data(overlay_seg(deconv, labels))

        axs[1].set_title(
            f"2. Deconvolved ({method}) | σ={sigma_px:.2f}px | it={iterations}"
        )

        fig.canvas.draw_idle()

    for s in (s_sigma, s_size, s_iter, s_bg, s_bal, s_gamma):
        s.on_changed(update)

    def set_method(label):
        nonlocal method
        method = label
        update()

    radio.on_clicked(set_method)

    # ---- Confirm ----
    ax_btn = plt.axes([0.42, 0.02, 0.16, 0.06])
    btn = Button(ax_btn, "CONFIRM")

    result = {}

    def confirm(event):
        result.update(
            sigma_px=sigma_px,
            size=size,
            iterations=iterations,
            background=background,
            method=method,
            balance=10**balance_log10
        )
        plt.close(fig)

    btn.on_clicked(confirm)
    plt.show()

    return result
