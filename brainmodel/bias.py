"""
N4 bias-field correction.

MP-RAGE (and MRI generally) has a smooth low-frequency intensity shading from
coil sensitivity. Correcting it makes intensity-based steps — the morphological
fallback's thresholds and the pial iso-surface level — behave the same across
scans and scanners, which is what "no per-case tuning" requires. Uses
SimpleITK's N4; degrades to a no-op if SimpleITK is unavailable.
"""
from __future__ import annotations
import numpy as np


def n4_correct(vol: np.ndarray, spacing, shrink: int = 4, iterations=(50, 50, 30, 20)):
    """
    Return an N4 bias-corrected copy of `vol` (float32). Falls back to the input
    unchanged (with a flag) if SimpleITK isn't installed.

    Returns (corrected, applied: bool).
    """
    try:
        import SimpleITK as sitk
    except Exception:
        return vol.astype(np.float32), False

    img = sitk.GetImageFromArray(vol.astype(np.float32))
    img.SetSpacing(tuple(float(s) for s in spacing[::-1]))  # sitk is x,y,z

    # A foreground mask keeps N4 from fitting the field to air noise.
    mask = sitk.OtsuThreshold(img, 0, 1, 200)

    shrunk = sitk.Shrink(img, [shrink] * img.GetDimension())
    shrunk_mask = sitk.Shrink(mask, [shrink] * img.GetDimension())

    n4 = sitk.N4BiasFieldCorrectionImageFilter()
    n4.SetMaximumNumberOfIterations(list(iterations))
    n4.Execute(shrunk, shrunk_mask)

    # Apply the estimated log-bias field at full resolution.
    log_bias = n4.GetLogBiasFieldAsImage(img)
    corrected = img / sitk.Exp(log_bias)
    out = sitk.GetArrayFromImage(corrected).astype(np.float32)
    out[~np.isfinite(out)] = 0.0
    return out, True
