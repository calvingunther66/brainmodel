"""
brainmodel — a scalable MRI brain/head reconstruction pipeline.

Turns an arbitrary head-MRI study into brain + head surface meshes without
per-case tuning:

    ingest -> conform -> bias-correct -> extract (skull-strip) -> QC -> surface

The skull-strip engine is pluggable (SynthStrip / deepbet / morphology) and the
surface stage meshes a continuous field (not a binary mask), so the output shows
real gyri/sulci and is free of the staircase "terracing" artifact.

See docs/REVIEW_AND_PLAN.md for the design rationale.
"""
from .config import Config

__all__ = ["Config"]
__version__ = "0.2.0"
