"""
Skull-strip engines behind one interface.

Every engine has the signature

    extract(vol, spacing, dirs, head=None) -> (mask: bool ndarray, name: str)

so the pipeline can pick one by name or let "auto" choose the most
contrast-robust engine that is actually installed. This is what lets the same
command handle T1, T2, FLAIR, etc. without hand-tuning: SynthStrip is
contrast-agnostic; deepbet is a fast CPU T1 model; morphology is the
dependency-free floor.
"""
from __future__ import annotations
import os
import shutil
import subprocess
import tempfile
import numpy as np
from scipy import ndimage as ndi
from skimage import morphology

from .geometry import to_nifti, largest_component


# --------------------------------------------------------------------------- #
# 1. Morphology (no deep-learning deps) — the deterministic floor.
# --------------------------------------------------------------------------- #
def morphology_extract(vol, spacing, dirs, head=None,
                       lo=0.22, hi=0.90, erode_r=5, grow=8, close_r=3):
    """
    Pure-morphology skull-strip of a T1 volume (numpy/scipy/skimage only).

    Keep a GM/WM intensity band (fat excluded), erode to an unambiguous core,
    geodesically regrow within the band, then close+fill to the pial envelope.
    Robust but T1-specific and the least accurate of the three; used only when
    no learned engine is available.
    """
    if head is None:
        head = np.ones(vol.shape, bool)
    p99 = np.percentile(vol[head], 99)
    band = ((vol > lo * p99) & (vol < hi * p99)) & head
    band = ndi.binary_fill_holes(band)

    core, _ = largest_component(ndi.binary_erosion(band, morphology.ball(erode_r)))
    brain = core.copy()
    for _ in range(grow):
        brain = ndi.binary_dilation(brain, morphology.ball(1)) & band
    brain = ndi.binary_closing(brain, morphology.ball(close_r))
    brain = ndi.binary_fill_holes(brain) & head
    brain, _ = largest_component(brain)
    return brain, "morphology"


# --------------------------------------------------------------------------- #
# 2. deepbet (lightweight 3D U-Net, CPU, T1).
# --------------------------------------------------------------------------- #
_DEEPBET_WEIGHTS = {
    "model.pt": "https://raw.githubusercontent.com/wwu-mmll/deepbet/main/data/models/model.pt",
    "bbox_model.pt": "https://raw.githubusercontent.com/wwu-mmll/deepbet/main/data/models/bbox_model.pt",
}


def _ensure_deepbet_weights():
    import urllib.request
    import deepbet
    models_dir = os.path.join(os.path.dirname(os.path.dirname(deepbet.__file__)),
                              "data", "models")
    os.makedirs(models_dir, exist_ok=True)
    for name, url in _DEEPBET_WEIGHTS.items():
        dst = os.path.join(models_dir, name)
        if not os.path.exists(dst) or os.path.getsize(dst) < 100_000:
            urllib.request.urlretrieve(url, dst)
    return models_dir


def deepbet_available():
    try:
        import torch  # noqa: F401
        import deepbet  # noqa: F401
        return True
    except Exception:
        return False


def deepbet_extract(vol, spacing, dirs, head=None):
    """Learned T1 skull-strip via deepbet. Needs a correctly oriented affine."""
    from deepbet.bet import BrainExtraction
    _ensure_deepbet_weights()
    img = to_nifti(vol, dirs, spacing)
    bet = BrainExtraction(no_gpu=True)
    _, mask_img, _ = bet.run(img)
    m = np.asarray(mask_img.dataobj).astype(bool)
    m = m[..., 0] if m.ndim == 4 else m
    if head is not None:
        m &= head
    m, _ = largest_component(m)
    return m, "deepbet"


# --------------------------------------------------------------------------- #
# 3. SynthStrip (contrast-agnostic; shells out to mri_synthstrip if present).
# --------------------------------------------------------------------------- #
def synthstrip_available():
    return shutil.which("mri_synthstrip") is not None


def synthstrip_extract(vol, spacing, dirs, head=None):
    """
    Contrast-agnostic skull-strip via FreeSurfer's mri_synthstrip. Works across
    T1/T2/FLAIR/CT and resolutions with no tuning — the best fit for "no
    case-by-case intervention" when the binary is available.
    """
    import nibabel as nib
    with tempfile.TemporaryDirectory() as td:
        in_p = os.path.join(td, "in.nii.gz")
        mask_p = os.path.join(td, "mask.nii.gz")
        nib.save(to_nifti(vol, dirs, spacing), in_p)
        subprocess.run(["mri_synthstrip", "-i", in_p, "-m", mask_p],
                       check=True, capture_output=True)
        m = np.asarray(nib.load(mask_p).dataobj).astype(bool)
    if head is not None:
        m &= head
    m, _ = largest_component(m)
    return m, "synthstrip"


# --------------------------------------------------------------------------- #
# Registry + dispatcher
# --------------------------------------------------------------------------- #
_ENGINES = {
    "synthstrip": (synthstrip_available, synthstrip_extract),
    "deepbet": (deepbet_available, deepbet_extract),
    "morphology": (lambda: True, morphology_extract),
}
# Priority order for "auto": most contrast-robust first.
_AUTO_ORDER = ["synthstrip", "deepbet", "morphology"]


def available_engines():
    return [name for name in _AUTO_ORDER if _ENGINES[name][0]()]


def extract(vol, spacing, dirs, engine="auto", head=None):
    """
    Run a skull-strip engine. `engine="auto"` walks the priority order and uses
    the first that is installed; a named engine is used directly (falling back
    to morphology if its deps are missing). Returns (mask, engine_name).
    """
    order = _AUTO_ORDER if engine == "auto" else [engine]
    errors = []
    for name in order:
        if name not in _ENGINES:
            raise ValueError(f"unknown engine {name!r}; choose from {list(_ENGINES)}")
        is_avail, fn = _ENGINES[name]
        if not is_avail():
            errors.append(f"{name}: not available")
            continue
        try:
            return fn(vol, spacing, dirs, head=head)
        except Exception as e:                       # pragma: no cover - runtime guard
            errors.append(f"{name}: {type(e).__name__}: {e}")
    # Last resort: morphology always runs.
    mask, name = morphology_extract(vol, spacing, dirs, head=head)
    return mask, f"morphology (after: {'; '.join(errors)})"
