#!/usr/bin/env python3
"""
Assemble a sorted 3D intensity volume from a directory of MRI DICOM slices.

Sorts slices along the true slice normal (from ImageOrientationPatient), applies
the rescale slope/intercept, and records the real voxel spacing in millimetres.

Usage:
    python3 scripts/build_volume.py [SERIES_DIR]

Default SERIES_DIR is data/dicom_staging/SER2 (the SAG 3D T1 MP-RAGE volume).
Outputs data/volume_raw.npy (z, y, x) and data/spacing.npy (dz, dy, dx mm).
"""
import glob
import os
import sys
import numpy as np
import pydicom

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
SERIES_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA, "dicom_staging", "SER2")


def main():
    files = glob.glob(os.path.join(SERIES_DIR, "*.dcm"))
    if not files:
        raise SystemExit(f"no DICOM files in {SERIES_DIR}")
    ds = [pydicom.dcmread(f) for f in files]

    # sort slices along the slice normal = cross(row_dir, col_dir)
    o = np.array(ds[0].ImageOrientationPatient, float)
    normal = np.cross(o[:3], o[3:])
    ds.sort(key=lambda d: float(np.dot(np.array(d.ImagePositionPatient, float), normal)))

    locs = np.array([np.dot(np.array(d.ImagePositionPatient, float), normal) for d in ds])
    dz = float(np.median(np.diff(locs)))
    ps = [float(v) for v in ds[0].PixelSpacing]

    def rescaled(d):
        a = d.pixel_array.astype(np.float32)
        return a * float(getattr(d, "RescaleSlope", 1)) + float(getattr(d, "RescaleIntercept", 0))

    vol = np.stack([rescaled(d) for d in ds]).astype(np.int16)

    print(f"series      : {ds[0].SeriesDescription} ({ds[0].MRAcquisitionType})")
    print(f"slices      : {len(ds)}")
    print(f"volume(zyx) : {vol.shape}  range {int(vol.min())}..{int(vol.max())}")
    print(f"spacing(mm) : dz={dz:.3f}  dy={ps[0]:.3f}  dx={ps[1]:.3f}")
    print(f"extent(mm)  : {len(ds) * dz:.0f} x {vol.shape[1] * ps[0]:.0f} x {vol.shape[2] * ps[1]:.0f}")

    # Record the patient-space direction of each voxel axis (LPS unit vectors),
    # so reconstruct.py can build a correct NIfTI affine for learned tools.
    #   axis0 = slice normal, axis1 = column dir (rows), axis2 = row dir (cols)
    dirs = np.array([normal, o[3:], o[:3]], float)

    os.makedirs(DATA, exist_ok=True)
    np.save(os.path.join(DATA, "volume_raw.npy"), vol)
    np.save(os.path.join(DATA, "spacing.npy"), np.array([dz, ps[0], ps[1]]))
    np.save(os.path.join(DATA, "dirs.npy"), dirs)
    print("saved data/volume_raw.npy + data/spacing.npy + data/dirs.npy")


if __name__ == "__main__":
    main()
