"""
Synthetic head/brain phantom — a PHI-free stand-in for tests and demos.

Nested ellipsoids (scalp / skull void / brain) with a corrugated cortical
surface and a low-frequency bias field, sampled anisotropically like an
MP-RAGE (1.0 x 0.5 x 0.5 mm). Lets the whole pipeline run end-to-end in CI
without touching real patient data.
"""
from __future__ import annotations
import numpy as np


def make_phantom(shape=(96, 128, 128), spacing=(1.0, 0.5, 0.5), seed=0):
    """Return (vol[z,y,x] float32, spacing, dirs 3x3) resembling a T1 head."""
    rng = np.random.default_rng(seed)
    z, y, x = shape
    sp = np.asarray(spacing, float)
    # physical coordinate grid centred at 0 (mm)
    zz, yy, xx = np.meshgrid(
        (np.arange(z) - z / 2) * sp[0],
        (np.arange(y) - y / 2) * sp[1],
        (np.arange(x) - x / 2) * sp[2], indexing="ij")
    r = np.sqrt(zz**2 + yy**2 + xx**2)

    # radii (mm)
    scalp_r, skull_out, skull_in, brain_r = 78.0, 74.0, 70.0, 66.0
    # corrugate the brain boundary to fake gyri/sulci
    theta = np.arctan2(yy, xx)
    phi = np.arctan2(zz, np.sqrt(xx**2 + yy**2) + 1e-6)
    corr = 3.0 * (np.sin(6 * theta) * np.cos(5 * phi) + np.sin(7 * phi))
    brain_bnd = brain_r + corr

    vol = np.zeros(shape, np.float32)
    vol[r < scalp_r] = 400.0                     # scalp/fat (bright)
    vol[r < skull_out] = 60.0                    # skull (dark void)
    vol[r < skull_in] = 250.0                    # CSF/GM ring
    brain = r < brain_bnd
    vol[brain] = 800.0                            # white matter (bright)
    # a gray-matter shell just inside the pial boundary
    gm = brain & (r > brain_bnd - 6.0)
    vol[gm] = 520.0

    vol += rng.normal(0, 12, shape).astype(np.float32)   # noise
    bias = 1.0 + 0.25 * (xx / (x * sp[2]))               # smooth shading
    vol *= bias
    vol = np.clip(vol, 0, None)

    dirs = np.eye(3)                             # axis-aligned LPS
    return vol.astype(np.float32), sp, dirs, brain


def make_brain_mask_pair(shape=(64, 64, 64), seed=0):
    """A clean brain mask + a version with a hole and a speck, for QC tests."""
    z, y, x = shape
    zz, yy, xx = np.meshgrid(*[np.arange(n) - n / 2 for n in shape], indexing="ij")
    r = np.sqrt(zz**2 + yy**2 + xx**2)
    good = r < min(shape) * 0.35
    bad = good.copy()
    bad[z // 2, y // 2, x // 2] = False           # interior hole
    bad[2, 2, 2] = True                           # detached speck
    return good, bad
