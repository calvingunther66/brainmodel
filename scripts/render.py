#!/usr/bin/env python3
"""
QC slice overlay: draw the brain-mask contour on the (bias-corrected) MRI in all
three planes, so the skull-strip can be checked against the source at a glance.

The 3D preview renders are produced separately by the z-buffered three.js viewer
(scripts/shoot.mjs) -- matplotlib's Poly3DCollection has no depth buffer and
mis-occludes a concave brain, so it is no longer used for 3D.

Usage:
    python3 scripts/render.py [WORK_DIR] [OUT_DIR]

WORK_DIR holds volume_iso.npy + brain_mask.npy (written by the pipeline with
save_intermediate=True). Defaults: data/work and out/.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORK = sys.argv[1] if len(sys.argv) > 1 else "/home/user/brainmodel/data/work"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/home/user/brainmodel/out"


def slice_montage(vol, brain, out):
    fig, axes = plt.subplots(3, 4, figsize=(12, 9), facecolor="black")
    for r, (ax_i, name) in enumerate([(0, "plane A"), (1, "plane B"), (2, "plane C")]):
        n = vol.shape[ax_i]
        for c, frac in enumerate((0.35, 0.45, 0.55, 0.65)):
            k = int(n * frac)
            img = np.take(vol, k, ax_i)
            msk = np.take(brain, k, ax_i)
            a = axes[r, c]
            a.imshow(img.T, cmap="gray", origin="lower")
            a.contour(msk.T, [0.5], colors="#ff4444", linewidths=0.6)
            a.set_axis_off()
            if c == 0:
                a.set_title(name, color="white", loc="left", fontsize=10)
    fig.suptitle("Brain skull-strip overlay (N4-corrected T1 MP-RAGE, 0.5 mm iso)",
                 color="white")
    fig.tight_layout()
    fig.savefig(out, dpi=100, facecolor="black")
    plt.close(fig)
    print("wrote", out)


def main():
    vol = np.load(f"{WORK}/volume_iso.npy")
    brain = np.load(f"{WORK}/brain_mask.npy")
    slice_montage(vol, brain, f"{OUT}/slices_overlay.png")


if __name__ == "__main__":
    main()
