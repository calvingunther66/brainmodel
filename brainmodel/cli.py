"""
Command-line entry point.

    python -m brainmodel run STUDY_DIR -o OUT/ [options]
    python -m brainmodel run scan.nii.gz -o OUT/ --engine synthstrip

No paths are baked in; point it at any DICOM study or NIfTI and it selects the
series, strips the skull, and writes brain/skin meshes + a QC manifest.
"""
from __future__ import annotations
import argparse
import sys

from .config import Config
from . import pipeline, extract


def build_parser():
    p = argparse.ArgumentParser(prog="brainmodel", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="reconstruct one study")
    r.add_argument("input", help="DICOM dir / series dir / .nii(.gz)")
    r.add_argument("-o", "--output", required=True, help="output directory")
    r.add_argument("--engine", default="auto",
                   choices=["auto", "synthstrip", "deepbet", "morphology"])
    r.add_argument("--iso-mm", type=float, default=0.5)
    r.add_argument("--surface", default="pial", choices=["pial", "envelope"],
                   dest="surface_mode")
    r.add_argument("--no-bias", action="store_true", help="skip N4 bias correction")
    r.add_argument("--config", help="JSON config file (overrides the flags above)")
    r.add_argument("-q", "--quiet", action="store_true")

    sub.add_parser("engines", help="list available skull-strip engines")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.cmd == "engines":
        print("available:", extract.available_engines())
        return 0

    if args.cmd == "run":
        if args.config:
            cfg = Config.from_json(args.config)
            cfg.input_path, cfg.output_dir = args.input, args.output
        else:
            cfg = Config(
                input_path=args.input, output_dir=args.output,
                engine=args.engine, iso_mm=args.iso_mm,
                surface_mode=args.surface_mode,
                do_bias_correction=not args.no_bias,
                verbose=not args.quiet)
        pipeline.run(cfg)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
