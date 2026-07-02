#!/usr/bin/env python3
"""
Reconstruct a 3D model from a stack of MRI DICOM slices.

Pipeline:
  1. Load the sorted intensity volume + voxel spacing (produced by build_volume).
  2. Resample to isotropic 1 mm voxels so proportions and meshes are correct.
  3. Segment two surfaces:
       - skin / head surface  (simple intensity threshold on the head vs. air)
       - brain surface        (learned skull-strip via deepbet, with a
                               pure-morphology fallback)
  4. Marching cubes -> triangle meshes, exported as STL / glB (mm units).

The brain extraction prefers a learned model (deepbet, a lightweight 3D U-Net):
it captures the whole cortex out to the pial surface -- including the vertex --
and cleanly seals the skull base (orbits, foramen magnum) that intensity-only
morphology cannot. If torch / deepbet / its weights are unavailable, it falls
back to the deterministic morphological skull-strip below, which needs only
numpy / scipy / scikit-image.
"""
import os
import urllib.request
import numpy as np
from scipy import ndimage as ndi
from skimage import measure, filters, morphology
import trimesh

DATA = "/home/user/brainmodel/data"
OUT = "/home/user/brainmodel/out"
os.makedirs(OUT, exist_ok=True)

# deepbet ships its traced weights out-of-band; fetch them from the repo on
# first run (raw.githubusercontent.com is reachable; the PyTorch index is not,
# but the CPU torch wheel installs fine from PyPI).
DEEPBET_WEIGHTS = {
    "model.pt": "https://raw.githubusercontent.com/wwu-mmll/deepbet/main/data/models/model.pt",
    "bbox_model.pt": "https://raw.githubusercontent.com/wwu-mmll/deepbet/main/data/models/bbox_model.pt",
}


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


def brain_mask(vol, head, lo=0.22, hi=0.90, erode_r=5, grow=8, close_r=3):
    """
    Skull-strip a T1 volume with pure morphology (no FSL/FreeSurfer/ANTs).

    Two anatomical facts make this robust:
      * On T1 the cortical skull is a dark signal void, so it forms a natural
        gap between brain and scalp.
      * Orbital / scalp / marrow FAT is brighter than brain GM/WM, so an upper
        intensity bound removes the orbits, which are otherwise the main path
        by which the brain leaks into the face.

    Steps:
      1. Keep a GM/WM intensity *band* (fat excluded). `lo`/`hi` are fractions
         of the 99th-percentile head intensity.
      2. Erode to a compact core that is unambiguously brain; keep the largest
         connected component (severs the scalp, which is a thin shell).
      3. Geodesically dilate the core back within the band to recover the whole
         cortex.
      4. Close + fill so the mask reaches the *pial envelope* (the sulcal CSF is
         bridged and the interior is solid). Without this the mask dips into
         every sulcus, which both under-counts the brain and makes the surface
         look chunky. Result ≈ 1360 cc for a normal adult brain.
    """
    p99 = np.percentile(vol[head], 99)
    band = ((vol > lo * p99) & (vol < hi * p99)) & head
    band = ndi.binary_fill_holes(band)

    core, _ = _largest_cc(ndi.binary_erosion(band, morphology.ball(erode_r)))

    brain = core.copy()
    for _ in range(grow):                      # geodesic dilation within band
        brain = ndi.binary_dilation(brain, morphology.ball(1)) & band

    brain = ndi.binary_closing(brain, morphology.ball(close_r))   # bridge sulci
    brain = ndi.binary_fill_holes(brain)                          # solid interior
    brain = brain & head
    brain, _ = _largest_cc(brain)
    return brain


def _ensure_deepbet_weights():
    """deepbet expects model.pt / bbox_model.pt under <site-packages>/data/models."""
    import deepbet
    models_dir = os.path.join(
        os.path.dirname(os.path.dirname(deepbet.__file__)), "data", "models"
    )
    os.makedirs(models_dir, exist_ok=True)
    for name, url in DEEPBET_WEIGHTS.items():
        dst = os.path.join(models_dir, name)
        if not os.path.exists(dst) or os.path.getsize(dst) < 100_000:
            print(f"  fetching deepbet weight {name} ...")
            urllib.request.urlretrieve(url, dst)
    return models_dir


def dl_brain_mask(vol):
    """
    Learned skull-strip with deepbet (3D U-Net). Returns a boolean brain mask
    the same shape as `vol`, or raises if torch/deepbet/weights are unavailable.

    The network is trained in canonical RAS orientation, so we wrap the volume
    in a NIfTI carrying the true patient-space affine (from data/dirs.npy, the
    LPS direction of each voxel axis recorded by build_volume). Without a correct
    affine the extraction is mis-oriented and fails.
    """
    import nibabel as nib
    from deepbet.bet import BrainExtraction

    _ensure_deepbet_weights()

    dirs = np.load(f"{DATA}/dirs.npy")          # LPS unit vectors: axis0, axis1, axis2
    lps_to_ras = np.array([-1.0, -1.0, 1.0])    # negate L->R and P->A
    R = (dirs * lps_to_ras).T                   # columns = RAS dir of each voxel axis
    affine = np.eye(4)
    affine[:3, :3] = R                          # 1 mm isotropic
    affine[:3, 3] = -R @ (np.array(vol.shape) / 2.0)   # centre the origin

    img = nib.Nifti1Image(vol.astype(np.float32), affine)
    bet = BrainExtraction(no_gpu=True)
    _, mask, _ = bet.run(img)
    m = np.asarray(mask.dataobj).astype(bool)
    return m[..., 0] if m.ndim == 4 else m


def mask_to_mesh(mask, spacing, step=1, smooth_iter=10, sigma=0.8):
    """Marching cubes on a binary mask -> smoothed trimesh (coords in mm)."""
    vol = ndi.gaussian_filter(mask.astype(np.float32), sigma)
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
    try:
        brain = dl_brain_mask(vol) & head
        brain, _ = _largest_cc(brain)
        print(f"  method: deepbet (learned U-Net)")
    except Exception as e:
        print(f"  deepbet unavailable ({type(e).__name__}: {e}); "
              f"falling back to morphology")
        brain = brain_mask(vol, head)
    print(f"  brain voxels: {int(brain.sum()):,}")
    np.save(f"{DATA}/brain_mask.npy", brain)
    np.save(f"{DATA}/head_mask.npy", head)

    print("Marching cubes -> meshes ...")
    skin_mesh = mask_to_mesh(head, spacing, step=2, smooth_iter=12, sigma=0.8)
    brain_mesh = mask_to_mesh(brain, spacing, step=1, smooth_iter=15, sigma=0.9)

    print("Exporting ...")
    export(skin_mesh, "skin", [230, 200, 180, 255])
    export(brain_mesh, "brain", [210, 190, 195, 255])
    print("Done.")


if __name__ == "__main__":
    main()
