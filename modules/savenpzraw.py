import numpy as np
import os
def save_npz_raw(out_dir, **data):
    """
    Save arbitrary python objects as a compressed npz.
    Uses allow_pickle-compatible object arrays.
    """
    path = os.path.join(out_dir, "detections.npz")

    np.savez_compressed(
        path,
        **{k: np.array(v, dtype=object) for k, v in data.items()}
    )
