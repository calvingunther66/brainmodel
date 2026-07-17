# 3D Brain Model — Head MRI (2025)

A 3D reconstruction of Calvin Gunther's brain and head, built from a routine
head MRI, plus a **scalable, config-driven pipeline** that turns *any* head-MRI
study into brain + head meshes without per-case tuning. The pipeline pulls the
DICOM study, automatically picks the best volumetric sequence, assembles a 3D
volume, corrects the bias field, skull-strips with a pluggable learned engine,
runs an automated QC gate, and reconstructs a **folded pial surface** plus a
head surface as printable / web-ready meshes with an interactive viewer.

![brain render](out/render_brain.png)

The brain surface now shows real **gyri and sulci** (meshed from the T1
intensity iso-surface, not the binary mask) and is free of the "onion-ring"
staircase artifact that came from meshing a 0/1 mask at 1 mm. See
[`docs/REVIEW_AND_PLAN.md`](docs/REVIEW_AND_PLAN.md) for the before/after
rationale and the full roadmap.

## Interactive viewer

Open **[`out/viewer.html`](out/viewer.html)** in a browser (three.js is vendored
locally in `out/vendor/`, so it works offline — no CDN needed). Toggle the brain
and head surfaces, fade the skin to see the brain inside, rotate, zoom, and pan.

| Brain surface | Brain inside translucent head |
|---|---|
| ![viewer brain](out/viewer_screenshot.png) | ![viewer head](out/viewer_skin_screenshot.png) |

## The source data

The study is a standard "BRAIN ROUTINE" MRI (3 T, no contrast) exported as a
DICOM ISO bundle — **22 series, ~1,000 images**. Different sequences image the
*same* head with different contrast/orientation; you do **not** fuse them into
one mesh. Instead you reconstruct from the single best 3D isotropic sequence.
The pipeline scores every series and selects it automatically (no hardcoded
"SER2"):

| Series | Description | Type | Slice | In-plane | Use |
|--------|-------------|------|-------|----------|-----|
| 1 | LOC (localizer) | 2D | 15 mm | 0.94 mm | scout, skipped |
| **2** | **SAG 3D T1 MP-RAGE** | **3D** | **1 mm** | **0.5 mm** | **← auto-selected (score 276)** |
| 3–8, 10, 12 | T1/T2/FLAIR/DWI (2D) | 2D | 3–4 mm | — | clinical review |
| 9 | AX 3D SWAN (susceptibility) | 3D | 1.97 mm | 0.43 mm | veins/microbleeds |
| 11,15,16,21 | ASL / ADC / CBF maps | derived | — | — | perfusion/diffusion |
| 200/201 | MP-RAGE reformats | derived | — | — | duplicate of series 2 |
| 22 | Structured report | SR | — | — | radiology report |

> **Clinical note (series 22):** 3 T brain MRI without contrast for abnormal
> movement / convulsive episodes — read as **normal, no acute intracranial
> findings**. This project is for visualization only and is **not** a diagnostic
> tool.

## Pipeline

```
       any DICOM study / NIfTI
                │  ingest.load_input      auto-select best 3D series; sort by slice normal
                ▼
        volume + geometry (LPS dirs)
                │  geometry.resample_iso  → 0.5 mm isotropic, cubic (keeps native detail)
                │  bias.n4_correct        → N4 bias-field correction
                ▼
        conformed volume
                │  extract.extract        pluggable skull-strip: synthstrip → deepbet → morphology
                │  qc.qc_report           ICV range · connectivity · holes · edge overlap
                ▼
        brain mask (+ head mask)
                │  surface.build_surface  pial: T1 iso-surface inside the mask (folds!)
                │                         skin: signed-distance envelope (terrace-free)
                ▼
        out/{brain,skin}.{stl,glb} + manifest.json
                │  scripts/shoot.mjs      z-buffered three.js renders
                │  scripts/render.py      QC slice overlay
                ▼
        renders + interactive viewer
```

### Run it on any study

```bash
python -m brainmodel run STUDY_DIR      -o out/            # DICOM dir or series dir
python -m brainmodel run scan.nii.gz    -o out/ --engine synthstrip
python -m brainmodel engines            # list available skull-strip engines
```

Nothing is hardcoded to one scan or path — behaviour is governed entirely by
[`brainmodel/config.py`](brainmodel/config.py) (resolution, engine, surface
mode, QC thresholds).

### Why it looks right now

* **Folded pial surface, no terracing.** The brain is meshed from the *N4-corrected
  T1 intensity* iso-surface at the GM/CSF boundary (restricted to the skull-strip
  mask), not from the binary mask. Marching cubes runs on a continuous,
  sub-voxel-blurred field, so the surface dips into sulci (real gyri) and has no
  staircase rings. Work is done at **0.5 mm** isotropic — the native in-plane
  resolution — instead of downsampling to 1 mm.
* **Contrast-agnostic, pluggable skull-strip.** `extract.extract()` tries
  [SynthStrip](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/) (works across
  T1/T2/FLAIR/CT with no tuning), then [deepbet](https://github.com/wwu-mmll/deepbet)
  (fast CPU T1 U-Net, used here), then a dependency-free morphology floor —
  whichever is installed. The NIfTI affine is rebuilt from the DICOM direction
  cosines so the learned models orient correctly.
* **Automated QC gate.** Every run reports intracranial volume, connectivity,
  interior holes, and how well the mask boundary sits on a real image edge, and
  flags outliers — so a cohort runs unattended and only suspect cases need a look.

The skull-strip was verified against the source slices (mask contour in red):

![segmentation overlay](out/slices_overlay.png)

## Outputs (`out/`)

| File | What |
|------|------|
| `viewer.html` (+ `vendor/`) | interactive three.js viewer (offline) |
| `brain.glb` / `brain.stl` | pial brain surface — web / 3D-print |
| `skin.glb` / `skin.stl` | head-and-face surface — web / 3D-print |
| `render_brain.png`, `render_head.png` | z-buffered multi-angle previews |
| `viewer_*.png` | viewer screenshots |
| `slices_overlay.png` | skull-strip QC overlay on the MRI |
| `manifest.json` | run metadata + QC metrics |

`.glb` for web/AR/Blender; `.stl` for 3D printing and CAD. Units are millimetres.
The committed meshes are quadric-decimated (brain 600k / skin 200k faces) so they
stay small enough for git and a browser while keeping the folds.

## Reproduce

```bash
pip install numpy scipy scikit-image trimesh pydicom nibabel SimpleITK matplotlib
pip install fast-simplification                 # mesh decimation
pip install torch deepbet                       # learned skull-strip (CPU wheel is fine)
#   optional: FreeSurfer's mri_synthstrip on PATH for contrast-agnostic stripping

# DICOMs are pulled from Google Drive and decoded into data/dicom_staging/SER2/.
python3 scripts/build_volume.py                 # (legacy helper) -> data/volume_raw.npy
python -m brainmodel run data/dicom_staging/SER2 -o out/   # full pipeline
node    scripts/shoot.mjs                        # -> out/*.png viewer renders
python3 tests/... ; pytest                       # run the test suite
```

The raw DICOM data and reconstructed volumes live under `data/`, which is
**git-ignored** because it contains protected health information (patient name,
MRN, DOB in the DICOM headers). Only derived surface meshes and renders are
committed.

## Roadmap

[`docs/REVIEW_AND_PLAN.md`](docs/REVIEW_AND_PLAN.md) tracks the plan. Delivered so
far: 0.5 mm isotropic meshing, intensity iso-surface (folds), N4 correction,
pluggable contrast-agnostic engine, automated QC gate, config-driven CLI, tests.
Next up for full topological perfection: a FastSurfer path (true `?h.pial` /
`?h.white` surfaces with parcellation) and a public-dataset Dice/Hausdorff
benchmark.
