"""
Automated QC gate.

The point of the gate is to run a cohort unattended and have only the suspect
cases flagged for a human — so success isn't judged by eyeballing every scan.
Each check returns a number plus a pass/fail against a Config threshold; the
case passes iff every check passes.
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi


def _volume_cc(mask, spacing):
    return float(mask.sum()) * float(np.prod(spacing)) / 1000.0


def _n_components(mask):
    _, n = ndi.label(mask)
    return int(n)


def _has_interior_holes(mask):
    filled = ndi.binary_fill_holes(mask)
    return bool((filled & ~mask).sum() > 0)


def _boundary_edge_overlap(mask, vol):
    """
    Fraction of the mask's surface voxels that sit on a strong intensity edge.
    A good skull-strip hugs the brain/CSF/skull gradient, so this should be high;
    a mask that cuts through uniform tissue scores low. Scale-free, no tuning.
    """
    surface = mask & ~ndi.binary_erosion(mask)
    if surface.sum() == 0:
        return 0.0
    grad = ndi.gaussian_gradient_magnitude(vol.astype(np.float32), 1.0)
    g = grad[surface]
    strong = grad > np.percentile(grad[grad > 0], 60)
    return float(strong[surface].mean())


def qc_report(mask, vol, spacing, cfg):
    """Return a dict of metrics + per-check pass flags + overall `passed`."""
    vol_cc = _volume_cc(mask, spacing)
    ncomp = _n_components(mask)
    holes = _has_interior_holes(mask)
    overlap = _boundary_edge_overlap(mask, vol)

    checks = {
        "intracranial_volume_cc": (vol_cc, cfg.icv_min_cc <= vol_cc <= cfg.icv_max_cc),
        "connected_components": (ncomp, ncomp <= cfg.max_components),
        "no_interior_holes": (not holes, not holes),
        "boundary_edge_overlap": (round(overlap, 3), overlap >= cfg.min_boundary_overlap),
    }
    passed = all(ok for _, ok in checks.values())
    return {
        "passed": passed,
        "metrics": {k: v for k, (v, _) in checks.items()},
        "checks": {k: ok for k, (_, ok) in checks.items()},
    }


def format_report(report):
    lines = [f"QC: {'PASS' if report['passed'] else 'FLAGGED'}"]
    for k, ok in report["checks"].items():
        lines.append(f"  [{'ok' if ok else 'XX'}] {k} = {report['metrics'][k]}")
    return "\n".join(lines)
