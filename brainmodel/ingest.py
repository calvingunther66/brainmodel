"""
Ingest: turn an arbitrary study into one volume + geometry, with automatic
series selection so nothing is hardcoded to "SER2".

Accepts either a NIfTI file (already a single volume) or a DICOM tree/series
directory. For DICOM it groups files by SeriesInstanceUID, scores each series
for surface-reconstruction suitability, and loads the winner.
"""
from __future__ import annotations
import glob
import os
import numpy as np


# --------------------------------------------------------------------------- #
# DICOM
# --------------------------------------------------------------------------- #
def _iter_dicom_files(root):
    if os.path.isfile(root):
        yield root
        return
    for dirpath, _, names in os.walk(root):
        for n in names:
            if n.lower().endswith(".dcm") or n.upper().startswith("IMG"):
                yield os.path.join(dirpath, n)


def scan_series(dicom_root):
    """Group DICOMs by series; return {uid: {meta..., 'files': [...]}}."""
    import pydicom
    series = {}
    for f in _iter_dicom_files(dicom_root):
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
            uid = str(ds.SeriesInstanceUID)
        except Exception:
            continue
        s = series.setdefault(uid, {
            "SeriesNumber": getattr(ds, "SeriesNumber", None),
            "SeriesDescription": str(getattr(ds, "SeriesDescription", "")),
            "MRAcquisitionType": str(getattr(ds, "MRAcquisitionType", "")),
            "Rows": int(getattr(ds, "Rows", 0)),
            "Columns": int(getattr(ds, "Columns", 0)),
            "PixelSpacing": [float(x) for x in getattr(ds, "PixelSpacing", [0, 0])],
            "SliceThickness": float(getattr(ds, "SliceThickness", 0) or 0),
            "ImageType": list(getattr(ds, "ImageType", [])),
            "files": [],
        })
        s["files"].append(f)
    return series


def score_series(meta):
    """
    Heuristic suitability score for surface reconstruction. Higher is better.
    Prefers: true 3D acquisitions, isotropic-ish voxels, whole-head slice count,
    T1/MPRAGE-like descriptions, primary (not derived) images.
    """
    score = 0.0
    desc = meta["SeriesDescription"].upper()
    if meta["MRAcquisitionType"] == "3D":
        score += 100
    n = len(meta["files"])
    score += min(n, 300) * 0.5                       # coverage
    ps = meta["PixelSpacing"]
    thick = meta["SliceThickness"] or 99
    if ps and ps[0] > 0:
        aniso = thick / ps[0]                        # 1.0 == isotropic
        score += max(0.0, 20.0 - abs(aniso - 1.0) * 8.0)
        score += max(0.0, 10.0 - ps[0] * 10.0)       # finer in-plane -> better
    if "MP" in desc and "RAGE" in desc:
        score += 40
    if "T1" in desc:
        score += 15
    if any(t in ("DERIVED", "SECONDARY", "PROJECTION IMAGE") for t in meta["ImageType"]):
        score -= 60
    if any(k in desc for k in ("LOC", "SCOUT", "ASL", "CBF", "ADC", "REFORMAT")):
        score -= 50
    return score


def select_best_series(series):
    """Return (uid, meta, ranking) with the best-scoring series first."""
    ranking = sorted(
        ((uid, m, score_series(m)) for uid, m in series.items()),
        key=lambda t: t[2], reverse=True)
    if not ranking:
        raise SystemExit("no readable DICOM series found")
    uid, meta, _ = ranking[0]
    return uid, meta, ranking


def load_series_volume(files):
    """
    Sort a series along its true slice normal, apply rescale, and return
    (volume[z,y,x], spacing[dz,dy,dx], dirs[3x3 LPS axis directions]).
    """
    import pydicom
    ds = [pydicom.dcmread(f, force=True) for f in files]
    o = np.array(ds[0].ImageOrientationPatient, float)
    normal = np.cross(o[:3], o[3:])
    ds.sort(key=lambda d: float(np.dot(np.array(d.ImagePositionPatient, float), normal)))

    locs = np.array([np.dot(np.array(d.ImagePositionPatient, float), normal) for d in ds])
    dz = float(np.median(np.diff(locs))) if len(ds) > 1 else float(ds[0].SliceThickness)
    ps = [float(v) for v in ds[0].PixelSpacing]

    def rescaled(d):
        a = d.pixel_array.astype(np.float32)
        return a * float(getattr(d, "RescaleSlope", 1)) + float(getattr(d, "RescaleIntercept", 0))

    vol = np.stack([rescaled(d) for d in ds]).astype(np.float32)
    dirs = np.array([normal, o[3:], o[:3]], float)     # slice normal, col dir, row dir
    spacing = np.array([abs(dz), ps[0], ps[1]], float)
    return vol, spacing, dirs


# --------------------------------------------------------------------------- #
# NIfTI (already-a-volume path, e.g. dcm2niix output)
# --------------------------------------------------------------------------- #
def load_nifti(path):
    """Load a NIfTI volume; derive spacing + LPS dirs from its affine."""
    import nibabel as nib
    img = nib.as_closest_canonical(nib.load(path))
    vol = np.asanyarray(img.dataobj, np.float32)
    zooms = img.header.get_zooms()[:3]
    # canonical (RAS) -> our axis0=k,1=j,2=i convention with LPS dirs
    R = img.affine[:3, :3]
    axis_dirs = (R / np.linalg.norm(R, axis=0)).T      # RAS dir of each voxel axis
    lps = axis_dirs * np.array([-1.0, -1.0, 1.0])
    # reorder to (axis2,axis1,axis0)->(z,y,x) to match DICOM convention
    vol = np.transpose(vol, (2, 1, 0))
    spacing = np.array([zooms[2], zooms[1], zooms[0]], float)
    dirs = np.array([lps[2], lps[1], lps[0]], float)
    return vol, spacing, dirs


def load_input(path):
    """
    Dispatch on the input path. Returns (vol, spacing, dirs, info).
    """
    if os.path.isfile(path) and (path.endswith(".nii") or path.endswith(".nii.gz")):
        vol, spacing, dirs = load_nifti(path)
        return vol, spacing, dirs, {"source": "nifti", "path": path}
    series = scan_series(path)
    uid, meta, ranking = select_best_series(series)
    vol, spacing, dirs = load_series_volume(meta["files"])
    info = {
        "source": "dicom",
        "selected_series": meta["SeriesDescription"],
        "series_number": meta["SeriesNumber"],
        "n_slices": len(meta["files"]),
        "ranking": [(m["SeriesDescription"], round(s, 1)) for _, m, s in ranking[:5]],
    }
    return vol, spacing, dirs, info
