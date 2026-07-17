"""
Surface reconstruction.

The old pipeline meshed the *binary* brain mask (0/1) with a tiny blur, which
produces two artifacts: a smooth blob with no gyri/sulci, and concentric
"onion-ring" terracing from the staircased mask along the anisotropic slice
axis. Two fixes here:

  * mesh a CONTINUOUS field, never the raw 0/1 array. Marching cubes on a
    signed-distance or intensity field is sub-voxel and antialiased -> no
    terracing.
  * for the brain, mesh the T1 INTENSITY iso-surface inside the mask (the
    GM/CSF boundary). The surface then dips into the sulci and shows real
    cortical folds instead of a hull.
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi
from skimage import measure
import trimesh

from .geometry import largest_component


def _signed_distance(mask: np.ndarray) -> np.ndarray:
    """Signed distance to the mask boundary (+inside), a smooth continuous field."""
    m = mask.astype(bool)
    inside = ndi.distance_transform_edt(m)
    outside = ndi.distance_transform_edt(~m)
    return inside - outside


def _decimate(mesh, target_faces):
    """Quadric-decimate to ~target_faces (keeps folds; shrinks web/print size)."""
    if not target_faces or len(mesh.faces) <= target_faces:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(face_count=int(target_faces))
    except Exception:
        return mesh          # backend missing -> keep full res rather than fail


def _finalize(verts, faces, normals, smooth_iterations, decimate_faces=0):
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals,
                           process=True)
    if smooth_iterations:
        # Taubin (lambda/mu) is volume-preserving: denoises without shrinking
        # or erasing folds the way plain Laplacian would.
        trimesh.smoothing.filter_taubin(mesh, iterations=smooth_iterations)
    comps = mesh.split(only_watertight=False)
    if len(comps):
        mesh = max(comps, key=lambda m: m.area)
    mesh = _decimate(mesh, decimate_faces)
    return mesh


def envelope_surface(mask, spacing, smooth_iterations=12, step=1, decimate_faces=0):
    """
    Smooth outer surface from the signed-distance field of `mask`.
    Terrace-free but featureless — used for the head/skin and as a brain fallback.
    """
    sdf = _signed_distance(mask).astype(np.float32)
    sdf = ndi.gaussian_filter(sdf, 0.6)
    verts, faces, normals, _ = measure.marching_cubes(
        sdf, level=0.0, spacing=tuple(spacing), step_size=step)
    return _finalize(verts, faces, normals, smooth_iterations, decimate_faces)


def pial_surface(vol, mask, spacing, iso=None, erode=1, field_sigma=0.9,
                 smooth_iterations=12, step=1, decimate_faces=0):
    """
    Folded cortical (pial) surface: mesh the T1 intensity iso-surface at the
    GM/CSF boundary, restricted to the brain mask. Shows gyri and sulci.

    `iso` (intensity level) defaults to the CSF/GM midpoint estimated from the
    in-mask histogram, so it adapts per scan without manual tuning.

    We restrict the intensity field to the mask *eroded* by `erode` (and keep its
    largest component) rather than dilated: the raw skull-strip boundary is ragged
    where it grazes the superior sagittal sinus / dura, and dilating there sprouts
    thin spurious sheets. `field_sigma` sub-voxel-blurs the field so marching
    cubes is antialiased (no terracing) without erasing the folds.
    """
    vol = vol.astype(np.float32)
    brain = mask.astype(bool)

    region = ndi.binary_erosion(brain, iterations=erode) if erode else brain
    region, _ = largest_component(region)

    inside = vol[region]
    lo, hi = np.percentile(inside, 2), np.percentile(inside, 98)
    norm = np.clip((vol - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

    if iso is None:
        # CSF sits low, GM/WM high; Otsu on in-mask intensities splits them.
        from skimage.filters import threshold_otsu
        try:
            iso = float(threshold_otsu(norm[region]))
        except Exception:
            iso = 0.5
        iso = float(np.clip(iso, 0.25, 0.6))

    # Real intensity inside the confident region; a value below `iso` elsewhere
    # so the surface closes cleanly at the brain edge.
    field = np.full(vol.shape, iso - 0.5, np.float32)
    field[region] = norm[region]
    field = ndi.gaussian_filter(field, field_sigma)

    verts, faces, normals, _ = measure.marching_cubes(
        field, level=iso, spacing=tuple(spacing), step_size=step)
    mesh = _finalize(verts, faces, normals, smooth_iterations, decimate_faces)
    return mesh, iso


def build_surface(vol, mask, spacing, mode="pial", smooth_iterations=12,
                  decimate_faces=0):
    """Dispatch on surface mode. Returns (mesh, info dict)."""
    if mode == "pial":
        mesh, iso = pial_surface(vol, mask, spacing,
                                 smooth_iterations=smooth_iterations,
                                 decimate_faces=decimate_faces)
        return mesh, {"mode": "pial", "iso_level": iso}
    elif mode == "envelope":
        mesh = envelope_surface(mask, spacing, smooth_iterations=smooth_iterations,
                                decimate_faces=decimate_faces)
        return mesh, {"mode": "envelope"}
    raise ValueError(f"unknown surface mode {mode!r}")


def export_mesh(mesh, out_stem, color):
    """Write .stl (print/CAD) and .glb (web/three.js), millimetre units."""
    mesh = mesh.copy()
    mesh.visual.vertex_colors = np.tile(np.asarray(color, np.uint8),
                                        (len(mesh.vertices), 1))
    _ = mesh.vertex_normals
    mesh.export(f"{out_stem}.stl")
    with open(f"{out_stem}.glb", "wb") as fh:
        fh.write(trimesh.exchange.gltf.export_glb(mesh, include_normals=True))
    return mesh
