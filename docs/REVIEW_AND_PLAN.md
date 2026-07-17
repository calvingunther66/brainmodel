# Brain Model — Review & Improvement Plan

*Reviewer pass over the pipeline in `scripts/` and the committed outputs in `out/`.
Goal: (1) assess what past runs built, (2) a plan to make the model **picture‑perfect**,
(3) a plan for a **scalable** brain‑from‑skull extractor that needs no per‑case tuning.*

---

## 1. What exists today

The current pipeline turns one head‑MRI DICOM study into two surface meshes (brain, head)
plus an offline viewer:

```
harvest.py      DICOMs pulled via the Drive MCP bridge → data/dicom_staging/SER2/
build_volume.py sort by slice normal, rescale, record spacing + direction cosines
reconstruct.py  resample → 1 mm iso → segment (deepbet U‑Net) → marching cubes → STL/GLB
render.py       matplotlib previews + slice‑overlay QC
viewer.html     three.js viewer (three.js vendored locally)
```

### What it gets right — keep these

- **Correct source sequence.** It reconstructs from the SAG 3D T1 MP‑RAGE (true 3D, 208
  slices, 1.0 × 0.5 × 0.5 mm) and treats the 2D clinical series as complementary. That is
  the right call for surface work.
- **Correct DICOM geometry.** `build_volume.py` sorts along the true slice normal
  (`cross(row, col)`), applies rescale slope/intercept, and records the LPS direction
  cosines. `reconstruct.py` rebuilds a valid NIfTI affine from them — which is exactly why
  the learned skull‑strip orients correctly. This is the part people most often get wrong,
  and it is right here.
- **A learned skull‑strip with a fallback.** deepbet (a 3D U‑Net) with a pure‑morphology
  fallback if torch is unavailable — a good instinct toward robustness.
- **Sound engineering hygiene.** Offline‑vendored viewer, STL **and** GLB exports, a QC
  overlay, and PHI kept out of git (`data/` is git‑ignored). Good.

### What holds it back

The verdict from the renders and the overlay: the segmentation is *roughly correct* but
the **surface is a smooth blob with no gyri or sulci**, wrapped in **concentric "onion‑ring"
staircase artifacts**. Those two things are why it does not look like a brain. The causes
are specific and fixable.

| # | Problem | Root cause | Evidence |
|---|---------|-----------|----------|
| **P1** | **No gyri/sulci** — surface is the smooth outer *envelope* | We mesh the **binary brain mask** (deepbet returns a filled pial *mask*, not a folded surface), then Gaussian‑ + Taubin‑smooth it. The folds were never in the mask. | `reconstruct.py:161` `mask_to_mesh(brain,…)`; `viewer_screenshot.png` shows a smooth walnut‑less blob |
| **P2** | **Terracing / onion rings** on brain *and* head | Marching cubes on a **0/1 binary** field produces stairsteps; the anisotropic acquisition (1.0 mm slice vs 0.5 mm in‑plane) resampled with **linear** interpolation deepens them along the slice axis. σ=0.9 blur only softens them. | ring pattern in `render_head.png`, `viewer_screenshot.png` |
| **P3** | **Native resolution discarded** | `to_isotropic(iso=1.0)` **downsamples** the 0.5 mm in‑plane data to 1 mm before meshing — half the detail is thrown away up front. | `reconstruct.py:47` `to_isotropic(..., iso=1.0)` |
| **P4** | **No bias‑field correction** | No N4/N3. MP‑RAGE has smooth intensity shading; the morphology fallback's fixed intensity thresholds (`lo=0.22, hi=0.90`) are fragile, and even learned tools do better on corrected input. | absent from all scripts |
| **P5** | **Preview renders are not faithful** | `render.py` uses matplotlib `Poly3DCollection` — painter's‑algorithm sorting, no z‑buffer — so it shows false self‑intersections and mis‑occlusion. | `render_brain.png` / `render_head.png` |
| **P6** | **Segmentation edge cases** | Minor notch at the superior‑sagittal midline on coronal slices; slight temporal‑lobe / cerebellum boundary wobble; the brainstem cut is arbitrary (no standardized plane at the foramen magnum). | `slices_overlay.png`, plane B cols 3–4 |
| **P7** | **No quantitative validation** | Volume "≈1710 cc" is quoted but there is no Dice/Hausdorff vs a reference, no intracranial‑volume sanity range, no automated pass/fail. Success is judged by eye. | no metrics anywhere |

**On the volume number:** ~1710 cc for a "brain" is high. Normal adult *brain* volume is
~1300–1500 cc and total *intracranial* volume ~1400–1700 cc. A filled pial‑envelope mask
that bridges every sulcus will land near the intracranial volume, not the brain volume —
another symptom of P1 (we are meshing the envelope, not the brain).

### Scalability gaps (the "no case‑by‑case intervention" ask)

The pipeline is a **single‑case script**, not a system:

- Hardcoded absolute paths (`/home/user/brainmodel/data`) and a hardcoded series (`SER2`).
- DICOM acquisition is a bespoke scrape of the MCP tool‑results directory (`harvest.py`),
  not a repeatable ingest.
- Single‑contrast assumption: deepbet is **T1‑trained**; a T2/FLAIR/post‑contrast/CT case
  fails or degrades silently.
- The morphology fallback's thresholds are hand‑tuned to this one scan and will not
  transfer across scanners or field strengths.
- No automatic series selection, no bias correction, no QC gate, no batch driver, no
  container, no config, no tests.

---

## 2. Plan A — make the model picture‑perfect

Two independent tracks: make the **surface anatomically real**, and make the **render
faithful**. The single most important change is to **stop meshing the binary mask.**

### Track 1 — a real cortical surface

**Recommended (the "picture‑perfect" path): run a cortical‑surface pipeline.**
Use **FastSurfer** (deep‑learning FreeSurfer). It produces topologically‑correct
`lh.pial` / `rh.pial` (outer, with every gyrus and sulcus) and `lh.white` / `rh.white`
surfaces, plus parcellation, in **~1 hour on a GPU** versus 7+ hours for classic
FreeSurfer. Convert the FreeSurfer surfaces to GLB/STL for the viewer. This is the
standard, anatomically faithful way to get a folded brain surface — the folds come from a
model trained to reproduce them, not from smoothing a blob.

> **Status — partially delivered.** FastSurfer's **ASEGDKT segmentation** (seg‑only,
> CPU, minutes) is now a first‑class engine (`--engine fastsurfer`): a topologically
> clean brain mask (1437 cc vs deepbet's 1697 cc envelope; Dice 0.92 agreement = a free
> two‑engine QC signal) plus a DKT cortical parcellation that colors the surface by region
> (`scripts/parcellate_surface.py`, `out/brain_parcellated.glb`, viewer "Parcellation" toggle).
> The full **`recon-surf`** stage (true `?h.pial`/`?h.white` surfaces) still needs a
> FreeSurfer license + GPU/hours and remains open — that is what would finally clean up the
> fine cerebellar‑folia / sagittal‑sinus regions the interim intensity iso‑surface renders busily.

- Output to keep: `?h.pial` (glossy folded cortex), optionally `?h.white` and the
  `aparc` parcellation for a colored‑lobes view.
- Deliverable: `pial.glb` (both hemispheres merged), `pial.stl`, and a viewer toggle for
  pial vs white vs parcellated.

**Cheaper interim (no GPU, keeps the current stack): intensity‑isosurface meshing.**
Instead of meshing the 0/1 mask, mesh the **T1 intensity field** at the gray‑matter/CSF
boundary, restricted to the brain mask:

1. Keep deepbet (or SynthStrip, see Plan B) only to get the **brain mask**.
2. Do **not** downsample — resample toward **0.5 mm isotropic** (upsample the slice
   axis) with cubic interpolation so the in‑plane detail survives (fixes P3).
3. Build a smooth scalar field for marching cubes that is **not binary** (fixes P2):
   either the N4‑corrected intensity masked to the brain, or a **signed‑distance /
   partial‑volume** field from the mask. Marching cubes on a continuous field with
   sub‑voxel `level` yields antialiased, terrace‑free surfaces.
4. Choose the iso‑level at the GM/CSF edge so the surface dips into sulci → real folds.
5. Light Taubin smoothing (λ/μ, volume‑preserving) only to denoise, not to erase folds.

Either path also fixes **P6**: FastSurfer standardizes the brainstem cut and topology;
for the interim path, add an explicit foramen‑magnum cut plane and keep the single largest
component after a morphological *opening* to shed the midline notch.

### Track 2 — a faithful render

- **Retire the matplotlib 3D preview (P5).** Generate the committed PNGs by shooting the
  three.js viewer headless (there is already `scripts/shoot.mjs`) — real z‑buffered,
  lit renders. Matplotlib `Poly3DCollection` cannot depth‑sort a concave brain.
- In the viewer, add MatCap or subsurface‑style shading and mild ambient occlusion so the
  folds read with depth; keep the existing skin‑opacity/brain‑inside toggle.

### Acceptance criteria for "picture‑perfect"

- Gyri and sulci are visible from every angle; no onion‑ring terracing at 100 % zoom.
- Brain (pial) volume lands in the physiological ~1300–1500 cc band, reported automatically.
- Overlay contour tracks the true GM/CSF boundary (dips into sulci), not a smoothed hull.
- Committed PNGs come from the z‑buffered viewer, not matplotlib.

---

## 3. Plan B — a scalable brain‑from‑skull extractor

Goal: point it at *any* head‑MRI study and get a correct brain mask with **no per‑case
tuning**. The engine already exists in the field — the work is choosing a contrast‑agnostic
model and wrapping it in a real pipeline with a QC gate.

### 3.1 Extraction engine — go contrast‑agnostic

| Tool | Contrast | Compute | Why |
|------|----------|---------|-----|
| **SynthStrip** *(recommend as default)* | **Any** — T1/T2/FLAIR/CT/any resolution | CPU ok, faster on GPU | Trained on synthetic images from label maps → agnostic to contrast/resolution/protocol. Single command, official Docker. Best fit for "no case‑by‑case intervention." |
| **HD‑BET** | T1/T2/FLAIR/T1c | GPU preferred | Very strong on standard clinical contrasts. |
| **deepbet** *(keep as fast CPU T1 path)* | T1 | CPU, seconds | Already integrated; fine when you know it's T1. |

Make the engine **pluggable** behind one interface (`extract_brain(volume) -> mask`), with
**SynthStrip as the default** and deepbet as the fast T1 fallback. Agreement between two
engines becomes a free QC signal (below).

### 3.2 Wrap it in a real pipeline

```
ingest → conform → bias‑correct → extract → QC gate → surface → report
```

1. **Ingest (any study).** Convert DICOM→NIfTI with **dcm2niix** (handles arbitrary
   studies, gz output). Drop the tool‑results scrape.
2. **Automatic series selection.** Rank series by metadata: prefer 3D acquisition,
   isotropic‑ish spacing, whole‑head coverage, T1 MP‑RAGE‑like; fall back gracefully.
   Removes the hardcoded `SER2`.
3. **Conform.** Reorient to canonical (RAS), resample to a fixed working grid (e.g.
   0.5–1.0 mm iso). One code path for every case.
4. **Bias‑field correction (fixes P4).** N4 (SimpleITK/ANTs) before extraction.
5. **Extract.** SynthStrip by default; configurable engine.
6. **QC gate (the key to "no intervention").** Auto‑flag suspect cases instead of failing
   silently:
   - intracranial volume in a plausible range,
   - mask is a single connected component, no interior holes, smooth boundary,
   - high overlap between the mask boundary and the image's strong‑gradient (skull) edge,
   - optional two‑engine Dice agreement; low agreement → flag for review.
   Green cases proceed untouched; only flagged cases get a human look.
7. **Surface + report.** Feed the mask to Plan A; emit a per‑case QC PDF/HTML (overlay
   montage + metrics) so a cohort can be scanned at a glance.

### 3.3 Package for scale

- One **containerized CLI**: `brainmodel run STUDY_DIR -o OUT/` — no hardcoded paths, all
  config in a file/flags. Batch driver over a cohort directory.
- **Validation before trusting it at scale.** Benchmark on public labeled datasets —
  **NFBS**, **CC‑359**, **LPBA40** — reporting **Dice** and **Hausdorff** vs ground truth
  across scanners/contrasts. That number is what proves it generalizes without tuning.
- Add a minimal **test suite** (geometry/affine round‑trip, QC‑gate logic on synthetic
  masks) so the pipeline can change safely.

---

## 4. Suggested order of work

| Phase | Work | Payoff |
|-------|------|--------|
| **0 — Quick wins** | Stop downsampling (mesh at 0.5 mm, P3); mesh a continuous field not the binary mask (P2); add N4 (P4); shoot previews from the viewer (P5). | Kills terracing and recovers detail with the current stack — the biggest visible jump for the least work. |
| **1 — Real surface** | Add FastSurfer path → true pial/white surfaces with folds (P1, P6); parcellated‑lobe view. | Anatomically faithful "picture‑perfect" brain. |
| **2 — Scalable engine** | Swap in SynthStrip behind a pluggable interface; dcm2niix ingest; auto series selection. | Any study, any contrast, no per‑case tuning. |
| **3 — Trust at scale** | QC gate + auto‑flagging; container + batch CLI; public‑dataset Dice/Hausdorff benchmark; tests. | A system you can run on a cohort and trust. |

**Biggest levers first:** Phase 0 removes the terracing and blur that make it look
synthetic; Phase 1 adds the folds that make it look like a brain; Phase 2 (SynthStrip +
QC gate) is what actually delivers "isolate the brain from the skull without case‑by‑case
intervention."

---

## 5. References

- SynthStrip — contrast‑agnostic skull stripping: <https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/>
  · paper: <https://pmc.ncbi.nlm.nih.gov/articles/PMC9465771/> · Docker: <https://hub.docker.com/r/freesurfer/synthstrip>
- FastSurfer — deep‑learning cortical surface pipeline: <https://deep-mi.org/research/fastsurfer/>
  · paper: <https://www.sciencedirect.com/science/article/pii/S1053811920304985>
- HD‑BET — <https://github.com/MIC-DKFZ/HD-BET>
- deepbet (current tool) — <https://github.com/wwu-mmll/deepbet>
- dcm2niix (DICOM→NIfTI) — <https://github.com/rordenlab/dcm2niix>
- Benchmarks: NFBS, CC‑359, LPBA40 labeled skull‑strip datasets.
