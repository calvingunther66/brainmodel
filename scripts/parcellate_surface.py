#!/usr/bin/env python3
"""
Build a parcellation-colored pial surface from a FastSurfer segmentation.

FastSurfer's ASEGDKT (`aparc.DKTatlas+aseg.deep.mgz`) gives a clean, topologically
sound whole-brain segmentation with cortical parcels (DKT atlas) — no dura or
sagittal-sinus leak. We mesh the folded pial surface from the conformed T1
intensity inside that mask and color each vertex by its cortical region using the
FreeSurfer color LUT.

Usage:
    python3 scripts/parcellate_surface.py FS_SUBJECT_DIR LUT_TXT OUT_STEM
"""
import sys
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi

sys.path.insert(0, "/home/user/brainmodel")
from brainmodel.surface import pial_surface  # noqa: E402
import trimesh  # noqa: E402


def load_lut(path):
    """FreeSurferColorLUT.txt -> {label_id: (r,g,b)}."""
    lut = {}
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 5 and parts[0].isdigit():
            lut[int(parts[0])] = (int(parts[2]), int(parts[3]), int(parts[4]))
    return lut


def main():
    subj, lut_path, out_stem = sys.argv[1], sys.argv[2], sys.argv[3]
    seg_img = nib.load(f"{subj}/mri/aparc.DKTatlas+aseg.deep.mgz")
    orig_img = nib.load(f"{subj}/mri/orig.mgz")
    seg = np.asarray(seg_img.dataobj).astype(np.int32)
    orig = np.asarray(orig_img.dataobj).astype(np.float32)
    spacing = [float(z) for z in orig_img.header.get_zooms()[:3]]
    lut = load_lut(lut_path)

    # Solid brain mask from the segmentation (fill ventricles/interior).
    brain = ndi.binary_fill_holes(seg > 0)

    # Only surface-visible structures should paint the pial surface: cortical
    # parcels (DKT 1000-2999), cerebellar cortex (8/47) and brainstem (16).
    # White matter (2/41), ventricles and CSF are interior — zeroing them stops
    # the surface being speckled white where sulcal walls graze WM voxels.
    surf_lab = seg.copy()
    keep = ((seg >= 1000) | np.isin(seg, [8, 47, 16]))
    surf_lab[~keep] = 0

    # Nearest surface label at every voxel, so each pial vertex gets a region.
    _, inds = ndi.distance_transform_edt(surf_lab == 0, return_indices=True)
    nearest = surf_lab[tuple(inds)]

    # Folded pial surface from the conformed T1 intensity inside the clean mask.
    mesh, iso = pial_surface(orig, brain, spacing, erode=1, field_sigma=0.9,
                             smooth_iterations=12, decimate_faces=600_000)
    print(f"pial: {len(mesh.vertices)} verts / {len(mesh.faces)} faces, iso={iso:.2f}")

    # Color each vertex by its parcellation label.
    vi = np.round(mesh.vertices / np.array(spacing)).astype(int)
    for d in range(3):
        vi[:, d] = np.clip(vi[:, d], 0, nearest.shape[d] - 1)
    labels = nearest[vi[:, 0], vi[:, 1], vi[:, 2]]
    default = (200, 200, 205)
    colors = np.array([[*lut.get(int(l), default), 255] for l in labels], np.uint8)
    mesh.visual.vertex_colors = colors

    n_regions = len(set(int(l) for l in labels if l >= 1000))
    print(f"colored {len(labels)} verts across {n_regions} cortical regions")

    mesh.export(f"{out_stem}.stl")
    with open(f"{out_stem}.glb", "wb") as fh:
        fh.write(trimesh.exchange.gltf.export_glb(mesh, include_normals=True))
    print(f"wrote {out_stem}.glb / .stl")


if __name__ == "__main__":
    main()
