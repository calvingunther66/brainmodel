"""
End-to-end pipeline:  ingest -> conform -> bias -> extract -> QC -> surface.

One code path for every study; behaviour is entirely governed by Config.
"""
from __future__ import annotations
import json
import os
import time
import numpy as np
from scipy import ndimage as ndi
from skimage import filters

from .config import Config
from . import ingest, bias, extract, surface, qc
from .geometry import resample_iso, largest_component


def _log(cfg, *a):
    if cfg.verbose:
        print(*a, flush=True)


def head_mask(vol):
    """Whole-head (skin/air) mask: everything above the air threshold, filled."""
    thr = filters.threshold_otsu(vol) * 0.5
    m = vol > thr
    m = ndi.binary_closing(m, iterations=2)
    m = ndi.binary_fill_holes(m)
    m, _ = largest_component(m)
    return m


def run(cfg: Config) -> dict:
    t0 = time.time()
    os.makedirs(cfg.output_dir, exist_ok=True)
    work = cfg.work_dir or os.path.join(cfg.output_dir, "work")
    os.makedirs(work, exist_ok=True)

    # 1. Ingest (auto series selection) --------------------------------------
    _log(cfg, "Ingesting", cfg.input_path)
    vol, spacing, dirs, in_info = ingest.load_input(cfg.input_path)
    _log(cfg, "  ", in_info)

    # 2. Conform to isotropic working resolution -----------------------------
    _log(cfg, f"Resampling to {cfg.iso_mm} mm iso (order {cfg.resample_order})")
    vol, spacing = resample_iso(vol, spacing, cfg.iso_mm, cfg.resample_order)
    _log(cfg, "  volume", vol.shape, "@", spacing, "mm")

    # 3. Bias-field correction ----------------------------------------------
    if cfg.do_bias_correction:
        vol, applied = bias.n4_correct(vol, spacing)
        _log(cfg, "  N4 bias correction:", "applied" if applied else "skipped (no SimpleITK)")

    # 4. Head + brain extraction --------------------------------------------
    _log(cfg, "Segmenting head")
    head = head_mask(vol)

    _log(cfg, f"Skull-stripping (engine={cfg.engine}; available={extract.available_engines()})")
    brain, engine_name = extract.extract(vol, spacing, dirs, engine=cfg.engine, head=head)
    _log(cfg, "  engine used:", engine_name, "| brain voxels:", int(brain.sum()))

    # 5. QC gate -------------------------------------------------------------
    report = qc.qc_report(brain, vol, spacing, cfg)
    _log(cfg, qc.format_report(report))

    if cfg.save_intermediate:
        np.save(os.path.join(work, "volume_iso.npy"), vol.astype(np.float32))
        np.save(os.path.join(work, "brain_mask.npy"), brain)
        np.save(os.path.join(work, "head_mask.npy"), head)

    # 6. Surfaces ------------------------------------------------------------
    _log(cfg, f"Reconstructing surfaces (mode={cfg.surface_mode})")
    brain_mesh, brain_info = surface.build_surface(
        vol, brain, spacing, mode=cfg.surface_mode,
        smooth_iterations=cfg.smooth_iterations, decimate_faces=cfg.brain_faces)
    skin_mesh = surface.envelope_surface(head, spacing,
                                         smooth_iterations=cfg.smooth_iterations,
                                         step=2, decimate_faces=cfg.skin_faces)

    surface.export_mesh(brain_mesh, os.path.join(cfg.output_dir, "brain"),
                        [210, 190, 195, 255])
    surface.export_mesh(skin_mesh, os.path.join(cfg.output_dir, "skin"),
                        [230, 200, 180, 255])
    _log(cfg, f"  brain: {len(brain_mesh.vertices)} verts / {len(brain_mesh.faces)} faces "
              f"({brain_info})")
    _log(cfg, f"  skin : {len(skin_mesh.vertices)} verts / {len(skin_mesh.faces)} faces")

    # 7. Manifest ------------------------------------------------------------
    manifest = {
        "input": in_info,
        "engine": engine_name,
        "surface": brain_info,
        "iso_mm": cfg.iso_mm,
        "qc": report,
        "brain_volume_cc": report["metrics"]["intracranial_volume_cc"],
        "meshes": {
            "brain": {"verts": len(brain_mesh.vertices), "faces": len(brain_mesh.faces)},
            "skin": {"verts": len(skin_mesh.vertices), "faces": len(skin_mesh.faces)},
        },
        "seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(cfg.output_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    _log(cfg, "Done in", manifest["seconds"], "s ->", cfg.output_dir)
    return manifest
