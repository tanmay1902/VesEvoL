from tifffile import imread as tiff_imread
from czifile import CziFile
import numpy as np
import os
import easygui as eg
import tifffile as tiff

def read_directory():
    directory = eg.diropenbox("Select a directory", "SMBL")
    if directory is None:
        print("Please re-run the program and select a directory to proceed. Quitting.")
        exit(1)

    print(f"Dir: {directory}")

    files = []
    for file in os.listdir(directory):
        if file.lower().endswith((".czi", ".tiff", ".tif")):
            full_path = os.path.join(directory, file)
            files.append(full_path)

    # 🔑 sort by filename WITHOUT extension
    files.sort(key=lambda p: os.path.splitext(os.path.basename(p))[0])

    # Optional: pretty print after sorting
    for path in files:
        size_bytes = os.path.getsize(path)
        size_gb = size_bytes / (1024 ** 3)
        
    return files, directory

def read_directory_old():
    # Ask the directory
    directory = eg.diropenbox("Select a directory", "SMBL")
    if directory is None:
        print("Please re-run the program and select a directory to proceed. Quitting.")
        exit(1)

    print(f"Dir: {directory}")

    files_in_dir = []
    for file in os.listdir(directory):
        if file.lower().endswith((".czi", ".tiff", ".tif")):
            full_path = os.path.join(directory, file)

            # ---- FILE SIZE IN GB ----
            size_bytes = os.path.getsize(full_path)
            size_gb = size_bytes / (1024 ** 3)

            print(f"  Found: {file} | Size: {size_gb:.2f} GB")

            files_in_dir.append(full_path)

    return files_in_dir,directory

from tifffile import imread as tiff_imread, TiffFile
from czifile import CziFile
import numpy as np
import os

def _ensure_z_first(arr, axes_str=None):
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr[np.newaxis, ...]
    if axes_str:
        axes = axes_str.upper()
        try:
            pos = {ax: axes.find(ax) for ax in ("Z", "Y", "X")}
            if pos["Y"] == -1 or pos["X"] == -1:
                raise ValueError
            if pos["Z"] != -1:
                z_idx = pos["Z"]
                arr = np.moveaxis(arr, z_idx, 0)
                return arr
            else:
                if arr.ndim > 3:
                    leading = arr.shape[:-2]
                    Z = int(np.prod(leading))
                    H, W = arr.shape[-2], arr.shape[-1]
                    arr = arr.reshape((Z, H, W))
                    return arr
                else:
                    if arr.ndim == 3:
                        return arr
                    else:
                        return arr[np.newaxis, ...]
        except Exception:
            pass
    if arr.ndim == 3:
        a0, a1, a2 = arr.shape
        if a1 == a2:
            return arr
        if a2 <= 50 and a2 < a0 and a2 < a1:
            arr = np.moveaxis(arr, 2, 0)
            return arr
        if a0 <= 50 and a0 < a1 and a0 < a2:
            return arr
        return arr
    elif arr.ndim > 3:
        leading = arr.shape[:-2]
        Z = int(np.prod(leading))
        H, W = arr.shape[-2], arr.shape[-1]
        arr = arr.reshape((Z, H, W))
        return arr
    else:
        return arr[np.newaxis, ...]

import czifile
import xml.etree.ElementTree as ET

def read_voxel_size_czi(path):
    with czifile.CziFile(path) as czi:
        metadata = czi.metadata()
    
    root = ET.fromstring(metadata)
    
    scaling = root.find(".//Scaling")
    items = scaling.findall("Items/Distance")
    
    voxel = {}
    for item in items:
        axis = item.attrib['Id']  # X, Y, Z
        value = float(item.find('Value').text)
        voxel[axis] = value * 1e6  # meters → microns
    
    return voxel.get('X')

def read_voxel_size_tiff(path):
    with tiff.TiffFile(path) as tif:
        page = tif.pages[0]
        tags = page.tags
        
        # X/Y pixel size
        if 'XResolution' in tags and 'YResolution' in tags:
            x_res = tags['XResolution'].value  # (num, denom)
            y_res = tags['YResolution'].value
            px_size_x = x_res[1] / x_res[0]
            px_size_y = y_res[1] / y_res[0]
        else:
            px_size_x = px_size_y = None
        
        # Z spacing (ImageJ or OME)
        metadata = tif.imagej_metadata
        z_size = None
        if metadata is not None:
            z_size = metadata.get('spacing', None)
        
        return px_size_x

def read_voxel_size(path):
    ext = os.path.splitext(path)[1].lower()
    
    if ext in ['.tif', '.tiff']:
        return read_voxel_size_tiff(path)
    
    elif ext == '.czi':
        return read_voxel_size_czi(path)
    
    else:
        raise ValueError("Unsupported file format")

def read_wavelength_czi(path):
    import czifile, xml.etree.ElementTree as ET

    with czifile.CziFile(path) as czi:
        meta = czi.metadata()

    root = ET.fromstring(meta)

    # Emission wavelength (nm)
    wl = root.findtext(
        ".//Channels/Channel/EmissionWavelength"
    )

    if wl is not None:
        return float(wl) / 1000.0  # nm → µm

    return None

def read_na_czi(path):
    import czifile, xml.etree.ElementTree as ET

    with czifile.CziFile(path) as czi:
        meta = czi.metadata()

    root = ET.fromstring(meta)

    na = root.findtext(".//Objective/NumericalAperture")
    if na:
        return float(na)

    return None

def read_immersion_czi(path):
    import czifile, xml.etree.ElementTree as ET

    with czifile.CziFile(path) as czi:
        meta = czi.metadata()

    root = ET.fromstring(meta)

    medium = root.findtext(".//Objective/Immersion")
    return medium  # Oil / Water / Air / Glycerol


def read_pinhole_czi(path):
    import czifile, xml.etree.ElementTree as ET

    with czifile.CziFile(path) as czi:
        meta = czi.metadata()

    root = ET.fromstring(meta)

    ph = root.findtext(".//PinholeDiameter")
    return float(ph) if ph else None

from alive_progress import alive_bar
def old_load_stack(files):
    stacks = []
    
    # Wrap the file loop with alive_bar
    with alive_bar(len(files), title="Loading Stacks", spinner="dots_waves") as bar:
        for path in files:
            path = str(path)
            size_bytes = os.path.getsize(path)
            size_gb = size_bytes / (1024 ** 3)
            # Update the progress bar text to show current file being read
            bar.text(f"Reading {os.path.basename(path)}.. Size: {size_gb:.2f} GB")
            
            arr = None
            if path.lower().endswith((".tif", ".tiff")):
                try:
                    with TiffFile(path) as tif:
                        series = tif.series[0]
                        axes = getattr(series, "axes", None)
                        arr_series = series.asarray()
                        arr_series = np.asarray(arr_series)
                        arr_series = _ensure_z_first(arr_series, axes_str=axes)
                        # If series returned Z==1 but file has many pages, stack pages explicitly
                        if arr_series.shape[0] == 1 and len(tif.pages) > 1:
                            pages = []
                            for p in tif.pages:
                                try:
                                    pages.append(np.asarray(p.asarray()))
                                except Exception:
                                    pages.append(np.asarray(p.asarray(0)))  # try explicit
                            arr = np.stack(pages, axis=0)
                        else:
                            arr = arr_series
                except Exception:
                    arr = np.asarray(tiff_imread(path))
                    arr = _ensure_z_first(arr, axes_str=None)

            elif path.lower().endswith(".czi"):
                try:
                    with CziFile(path) as czi:
                        arr_czi = czi.asarray()
                        axes = getattr(czi, 'axes', None)
                        arr_czi = np.squeeze(arr_czi)
                        arr = _ensure_z_first(arr_czi, axes_str=axes)
                except Exception:
                    arr = np.asarray(tiff_imread(path))
                    arr = _ensure_z_first(arr, axes_str=None)
            else:
                raise ValueError(f"Only TIFF and CZI supported, got {path}")

            arr = np.asarray(arr)
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            if arr.ndim != 3:
                leading = arr.shape[:-2]
                Z = int(np.prod(leading))
                H, W = arr.shape[-2], arr.shape[-1]
                arr = arr.reshape((Z, H, W))

            arr = arr.astype(np.float32)
            for z in range(arr.shape[0]):
                stacks.append(arr[z])

            # Optional: Keep the debug print, alive_bar will handle it cleanly
            # print(f"Loaded {path} -> shape (Z,H,W) = {arr.shape}")
            
            # Advance the progress bar
            bar()
    #skips = len(stacks)//100
    return stacks

import numpy as np
from tifffile import imread
from czifile import CziFile
from alive_progress import alive_bar

def load_stack(path):
    with alive_bar(1) as bred:
        if isinstance(path,list): path = path[0]
        """Load TIFF or CZI and return (Z, H, W) float32 stack."""
        path = str(path)
        if path.lower().endswith((".tif", ".tiff")):
            arr = imread(path)
            print("File Read")
        elif path.lower().endswith(".czi"):
            with CziFile(path) as czi:
                arr = np.squeeze(czi.asarray())

        else:
            raise ValueError("Only TIFF and CZI supported")

        arr = np.asarray(arr)

        # force (Z, H, W)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        elif arr.ndim > 3:
            arr = arr.reshape((-1, arr.shape[-2], arr.shape[-1]))
        bred()
        return arr.astype(np.float32)[::5]


import numpy as np
import tifffile as tiff
from aicspylibczi import CziFile
from alive_progress import alive_bar

def load_stack_efficient(path, step=10):
    if isinstance(path,list): path = path[0]
    
    if path.lower().endswith((".tif", ".tiff")):
        with tiff.TiffFile(path) as tif:
            # Get dimensions without loading data
            pages = tif.pages
            total_z = len(pages)
            h, w = pages[0].shape
            
            indices = range(0, total_z, step)
            # Pre-allocate the small output array (32-bit float)
            stack = np.empty((len(indices), h, w), dtype=np.float32)
            
            with alive_bar(len(indices), title="Reading TIFF Slices") as bar:
                for i, idx in enumerate(indices):
                    # Load only one page at a time
                    stack[i] = pages[idx].asarray().astype(np.float32)
                    bar()
            return stack

    elif path.lower().endswith(".czi"):
        czi = CziFile(path)
        dims = czi.get_dims_shape()[0]

        total_t = dims['T'][1]
        h, w = dims['Y'][1], dims['X'][1]

        indices = range(0, total_t, step)
        stack = np.empty((len(indices), h, w), dtype=np.float32)

        with alive_bar(len(indices), title=f"Reading {path} CZI Time Frames (step={step})") as bar:
            for i, t_idx in enumerate(indices):
                frame, _ = czi.read_image(T=t_idx)

                # frame shape will be (1, 1, Y, X)
                frame = np.squeeze(frame)

                stack[i] = frame.astype(np.float32)
                bar()

        return stack
    else:
        raise ValueError("Unsupported format")