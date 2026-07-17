"""Configuration for the reconstruction pipeline (no hardcoded paths)."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Config:
    """
    All pipeline parameters in one place. Construct from the CLI, a JSON file,
    or directly. Nothing here is specific to one scan or one machine.
    """

    # --- I/O ---
    input_path: str                      # DICOM dir, a series subdir, or a .nii/.nii.gz
    output_dir: str                      # where meshes / renders / QC land
    work_dir: Optional[str] = None       # scratch for intermediate volumes (default: <out>/work)

    # --- geometry ---
    iso_mm: float = 0.5                  # isotropic working resolution (mm).
    #   The MP-RAGE is 0.5 mm in-plane; resampling to 0.5 (not 1.0) keeps that
    #   detail instead of throwing half of it away.
    resample_order: int = 3             # cubic; smoother than linear -> less terracing

    # --- preprocessing ---
    do_bias_correction: bool = True     # N4 before extraction (robust to shading)

    # --- skull-strip engine ---
    engine: str = "auto"                # auto | synthstrip | deepbet | morphology
    #   "auto" tries them in order of contrast-robustness and falls back.

    # --- surface reconstruction ---
    surface_mode: str = "pial"          # pial | envelope
    #   pial     -> mesh the intensity iso-surface inside the brain (shows folds)
    #   envelope -> mesh a smooth signed-distance field of the mask (smooth hull)
    smooth_iterations: int = 12         # volume-preserving Taubin passes
    # Quadric-decimate the web/print meshes to a target face count (keeps folds,
    # keeps .glb small enough for git + a browser). 0 = keep full resolution.
    brain_faces: int = 600_000
    skin_faces: int = 200_000

    # --- QC gate thresholds (flag, don't fail) ---
    icv_min_cc: float = 900.0
    icv_max_cc: float = 2200.0
    max_components: int = 1
    min_boundary_overlap: float = 0.55  # fraction of surface on a strong image edge

    # --- misc ---
    save_intermediate: bool = True
    verbose: bool = True
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, path: str) -> "Config":
        import json
        with open(path) as fh:
            return cls(**json.load(fh))
