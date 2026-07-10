#!/usr/bin/env python3
"""Headless preview renders of the reconstructed meshes + a slice montage."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import ndimage as ndi
from skimage import measure

DATA = "/home/user/brainmodel/data"
OUT = "/home/user/brainmodel/out"


def lowpoly(mask, step):
    v = ndi.gaussian_filter(mask.astype(np.float32), 0.8)
    verts, faces, _, _ = measure.marching_cubes(v, 0.5, step_size=step)
    return verts, faces


def render_mesh(verts, faces, color, out, elevs_azims, title):
    fig = plt.figure(figsize=(4 * len(elevs_azims), 4), facecolor="black")
    for i, (elev, azim) in enumerate(elevs_azims, 1):
        ax = fig.add_subplot(1, len(elevs_azims), i, projection="3d")
        mesh = Poly3DCollection(verts[faces], linewidths=0)
        # Lambertian shading from face normals with an ambient floor
        tris = verts[faces]
        n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
        light = np.array([0.35, 0.45, 0.82])
        shade = 0.45 + 0.55 * np.clip(np.abs(n @ light), 0, 1)   # ambient+diffuse
        c = np.array(color) / 255.0
        facecols = np.clip(shade[:, None] * c[None, :], 0, 1)
        mesh.set_facecolor(facecols)
        ax.add_collection3d(mesh)
        mn, mx = verts.min(0), verts.max(0)
        ctr = (mn + mx) / 2
        r = (mx - mn).max() / 2
        ax.set_xlim(ctr[0] - r, ctr[0] + r)
        ax.set_ylim(ctr[1] - r, ctr[1] + r)
        ax.set_zlim(ctr[2] - r, ctr[2] + r)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_box_aspect((1, 1, 1))
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.set_pane_color((0, 0, 0, 0))
    fig.suptitle(title, y=0.98, fontsize=13, color="white")
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print("wrote", out)


def slice_montage(vol, brain, out):
    fig, axes = plt.subplots(3, 4, figsize=(12, 9), facecolor="black")
    # patient sagittal is axis 0 (data was a sagittal acquisition); show all 3 planes
    planes = [(0, "plane A"), (1, "plane B"), (2, "plane C")]
    for r, (ax_i, name) in enumerate(planes):
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
                a.set_title(f"{name}", color="white", loc="left", fontsize=10)
    fig.suptitle("Brain segmentation overlay (T1 MP-RAGE)", color="white")
    fig.tight_layout()
    fig.savefig(out, dpi=100, facecolor="black")
    plt.close(fig)
    print("wrote", out)


def main():
    vol = np.load(f"{DATA}/volume_iso.npy")
    brain = np.load(f"{DATA}/brain_mask.npy")
    head = np.load(f"{DATA}/head_mask.npy")

    bv, bf = lowpoly(brain, step=2)
    hv, hf = lowpoly(head, step=3)
    views = [(15, -75), (10, 15), (90, -90)]  # side, front-ish, top
    render_mesh(bv, bf, [205, 185, 190], f"{OUT}/render_brain.png", views,
                "Reconstructed brain surface")
    render_mesh(hv, hf, [225, 200, 180], f"{OUT}/render_head.png", views,
                "Reconstructed head / skin surface")
    slice_montage(vol, brain, f"{OUT}/slices_overlay.png")


if __name__ == "__main__":
    main()
