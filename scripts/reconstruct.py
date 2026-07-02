#!/usr/bin/env python3
"""
Reconstruct a 3D model from a stack of MRI DICOM slices.

Pipeline:
  1. Load the sorted intensity volume + voxel spacing (produced by build_volume).
  2. Resample to isotropic 1 mm voxels so proportions and meshes are correct.
  3. Segment two surfaces:
       - skin / head surface  (simple intensity threshold on the head vs. air)
       - brain surface        (morphological skull-strip of the T1 volume)
  4. Marching cubes -> triangle meshes, exported as STL / PLY / glB (mm units).

Everything is deterministic and depends only on numpy / scipy / scikit-image /
trimesh, so it runs headless without FSL/FreeSurfer/ANTs.
"""
import os
import numpy as np
from scipy import ndimage as ndi
from skimage import measure, filters, morphology
import trimesh

DATA = "/home/user/brainmodel/data"
OUT = "/home/user/brainmodel/out"
os.makedirs(OUT, exist_ok=True)


def load_volume():
    vol = np.load(f"{DATA}/volume_raw.npy").astype(np.float32)  # (z, y, x)
    spacing = np.load(f"{DATA}/spacing.npy").astype(float)      # (dz, dy, dx) mm
    return vol, spacing


def to_isotropic(vol, spacing, iso=1.0):
    """Resample the volume to `iso` mm isotropic voxels."""
    zoom = np.array(spacing) / iso
    iso_vol = ndi.zoom(vol, zoom, order=1)
    return iso_vol, np.array([iso, iso, iso])


def skin_surface_mask(vol):
    """Whole-head (skin/air boundary) mask: everything brighter than air, filled."""
    thr = filters.threshold_otsu(vol) * 0.5
    mask = vol > thr
    mask = ndi.binary_closing(mask, iterations=2)
    mask = ndi.binary_fill_holes(mask)
    # keep the largest connected component (the head), drop stray noise
    lbl, n = ndi.label(mask)
    if n > 1:
        sizes = ndi.sum(np.ones_like(lbl), lbl, range(1, n + 1))
        mask = lbl == (1 + int(np.argmax(sizes)))
    return mask


def _largest_cc(mask):
    lbl, n = ndi.label(mask)
    if n == 0:
        return np.zeros_like(mask), 0
    sizes = ndi.sum(np.ones_like(lbl), lbl, range(1, n + 1))
    k = 1 + int(np.argmax(sizes))
    return lbl == k, int(sizes.max())


def brain_mask(vol, head, lo=0.25, hi=0.85, erode_r=5, grow=7):
    """
    Skull-strip a T1 volume with pure morphology (no FSL/FreeSurfer/ANTs).

    Two anatomical facts make this robust:
      * On T1 the cortical skull is a dark signal void, so it forms a natural
        gap between brain and scalp.
      * Orbital / scalp / marrow FAT is brighter than brain GM/WM, so an upper
        intensity bound removes the orbits, which are otherwise the main path
        by which the brain leaks into the face.

    So we keep a GM/WM intensity *band* (fat excluded), erode to a compact core
    that is unambiguously brain, keep the largest component, then geodesically
    reconstruct the brain back within the band. `lo`/`hi` are fractions of the
    99th-percentile head intensity.
    """
    p99 = np.percentile(vol[head], 99)
    band = ((vol > lo * p99) & (vol < hi * p99)) & head
    band = ndi.binary_fill_holes(band)

    core, _ = _largest_cc(ndi.binary_erosion(band, morphology.ball(erode_r)))

    brain = core.copy()
    for _ in range(grow):                      # geodesic dilation within band
        brain = ndi.binary_dilation(brain, morphology.ball(1)) & band
    brain = ndi.binary_closing(brain, morphology.ball(3))
    brain = ndi.binary_fill_holes(brain)
    brain, _ = _largest_cc(brain)
    return brain


def mask_to_mesh(mask, spacing, step=1, smooth_iter=10):
    """Marching cubes on a binary mask -> smoothed trimesh (coords in mm)."""
    vol = ndi.gaussian_filter(mask.astype(np.float32), 0.8)
    verts, faces, normals, _ = measure.marching_cubes(
        vol, level=0.5, spacing=tuple(spacing), step_size=step
    )
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals,
                           process=True)
    trimesh.smoothing.filter_taubin(mesh, iterations=smooth_iter)
    # keep only the largest connected surface (drops speckle)
    comps = mesh.split(only_watertight=False)
    if len(comps):
        mesh = max(comps, key=lambda m: m.area)
    return mesh


def export(mesh, name, color):
    mesh = mesh.copy()
    mesh.visual.vertex_colors = np.tile(color, (len(mesh.vertices), 1))
    _ = mesh.vertex_normals  # ensure normals are computed before export
    mesh.export(f"{OUT}/{name}.stl")                       # 3D printing / CAD
    with open(f"{OUT}/{name}.glb", "wb") as fh:            # web / three.js
        fh.write(trimesh.exchange.gltf.export_glb(mesh, include_normals=True))
    print(f"  {name}: {len(mesh.vertices):>7d} verts  {len(mesh.faces):>7d} faces"
          f"  -> {name}.stl/.glb")
    return mesh


def main():
    print("Loading volume ...")
    vol, spacing = load_volume()
    print(f"  raw {vol.shape} @ {spacing} mm")
    vol, spacing = to_isotropic(vol, spacing, iso=1.0)
    print(f"  isotropic {vol.shape} @ {spacing} mm")
    np.save(f"{DATA}/volume_iso.npy", vol.astype(np.float32))

    print("Segmenting head (skin) ...")
    head = skin_surface_mask(vol)
    print(f"  head voxels: {int(head.sum()):,}")

    print("Skull-stripping brain ...")
    brain = brain_mask(vol, head)
    print(f"  brain voxels: {int(brain.sum()):,}")
    np.save(f"{DATA}/brain_mask.npy", brain)
    np.save(f"{DATA}/head_mask.npy", head)

    print("Marching cubes -> meshes ...")
    skin_mesh = mask_to_mesh(head, spacing, step=2, smooth_iter=12)
    brain_mesh = mask_to_mesh(brain, spacing, step=1, smooth_iter=15)

    print("Exporting ...")
    export(skin_mesh, "skin", [230, 200, 180, 255])
    export(brain_mesh, "brain", [210, 190, 195, 255])
    print("Done.")


if __name__ == "__main__":
    main()
