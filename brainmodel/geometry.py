"""
Volume geometry: patient-space affines, canonical orientation, resampling.

Correct geometry is the part learned skull-strip tools are most sensitive to:
they expect a NIfTI whose affine describes the true patient orientation. We
build that affine from the DICOM direction cosines recorded at ingest.
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi

# LPS (DICOM) -> RAS (NIfTI): negate the L->R and P->A axes.
_LPS_TO_RAS = np.array([-1.0, -1.0, 1.0])


def affine_from_dirs(dirs: np.ndarray, spacing, shape) -> np.ndarray:
    """
    Build a 4x4 RAS affine from LPS voxel-axis directions + spacing.

    `dirs` rows are the LPS unit vectors of voxel axes (axis0, axis1, axis2),
    as recorded by ingest (slice normal, column dir, row dir).
    """
    dirs = np.asarray(dirs, float)
    spacing = np.asarray(spacing, float)
    R = (dirs * _LPS_TO_RAS).T * spacing            # columns scaled by voxel size
    affine = np.eye(4)
    affine[:3, :3] = R
    affine[:3, 3] = -R @ (np.asarray(shape, float) / 2.0)   # centre the origin
    return affine


def to_nifti(vol: np.ndarray, dirs, spacing):
    """Wrap a volume in a nibabel NIfTI carrying the true patient affine."""
    import nibabel as nib
    affine = affine_from_dirs(dirs, spacing, vol.shape)
    return nib.Nifti1Image(np.asanyarray(vol, np.float32), affine)


def resample_iso(vol: np.ndarray, spacing, iso: float, order: int = 3):
    """
    Resample to `iso` mm isotropic voxels.

    With the 1.0 x 0.5 x 0.5 mm MP-RAGE and iso=0.5 this UPSAMPLES the slice
    axis and KEEPS the native in-plane detail, instead of downsampling to 1 mm
    (which the old pipeline did, discarding half the resolution). Cubic order
    also softens the slice-direction staircase before meshing.
    """
    spacing = np.asarray(spacing, float)
    zoom = spacing / float(iso)
    out = ndi.zoom(vol.astype(np.float32), zoom, order=order)
    return out, np.array([iso, iso, iso], float)


def largest_component(mask: np.ndarray):
    """Keep the largest connected component of a boolean mask."""
    lbl, n = ndi.label(mask)
    if n <= 1:
        return mask.astype(bool), int(mask.sum())
    sizes = ndi.sum(np.ones_like(lbl), lbl, range(1, n + 1))
    k = 1 + int(np.argmax(sizes))
    return lbl == k, int(sizes.max())
