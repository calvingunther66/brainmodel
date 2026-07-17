"""
Pipeline tests that run on the synthetic phantom (no PHI, no network).

Covers the parts that must not silently regress: geometry/affine, the
antialiasing surface fix, the QC gate logic, engine dispatch, and a full
end-to-end reconstruction.
"""
import numpy as np
import pytest

from brainmodel import phantom
from brainmodel.geometry import affine_from_dirs, resample_iso, largest_component
from brainmodel import surface, qc, extract
from brainmodel.config import Config


def test_affine_orientation_and_scale():
    dirs = np.eye(3)
    aff = affine_from_dirs(dirs, [1.0, 0.5, 0.5], (10, 20, 20))
    # RAS diagonal: LPS axis0/1 negate, spacing on the diagonal magnitude
    assert np.allclose(np.abs(np.diag(aff)[:3]), [1.0, 0.5, 0.5])
    assert aff[0, 0] < 0 and aff[1, 1] < 0 and aff[2, 2] > 0


def test_resample_preserves_detail_not_downsampled():
    vol, sp, _, _ = phantom.make_phantom(shape=(40, 60, 60))
    out, out_sp = resample_iso(vol, sp, iso=0.5, order=3)
    # 0.5 mm iso must UPSAMPLE the 1 mm slice axis, not shrink in-plane
    assert out.shape[0] > vol.shape[0]
    assert out.shape[1] >= vol.shape[1]
    assert np.allclose(out_sp, [0.5, 0.5, 0.5])


def test_largest_component():
    m = np.zeros((10, 10, 10), bool)
    m[1:5, 1:5, 1:5] = True
    m[8, 8, 8] = True            # speck
    big, size = largest_component(m)
    assert big[2, 2, 2] and not big[8, 8, 8]
    assert size == 64


def test_pial_surface_is_antialiased_and_folded():
    vol, sp, _, brain = phantom.make_phantom()
    mesh, iso = surface.pial_surface(vol, brain, [0.5, 0.5, 0.5],
                                     smooth_iterations=5)
    assert len(mesh.vertices) > 1000
    assert 0.25 <= iso <= 0.6
    # a corrugated surface has meaningfully more area than its convex hull
    assert mesh.area > mesh.convex_hull.area * 1.05


def test_envelope_surface_runs():
    _, _, _, brain = phantom.make_phantom()
    mesh = surface.envelope_surface(brain, [0.5, 0.5, 0.5], smooth_iterations=3)
    assert len(mesh.faces) > 100


def test_qc_gate_flags_bad_mask():
    good, bad = phantom.make_brain_mask_pair()
    vol = good.astype(np.float32) * 800 + 50
    cfg = Config(input_path="", output_dir="", icv_min_cc=0, icv_max_cc=1e9)
    good_rep = qc.qc_report(good, vol, [1, 1, 1], cfg)
    bad_rep = qc.qc_report(bad, vol, [1, 1, 1], cfg)
    assert good_rep["checks"]["no_interior_holes"]
    assert not bad_rep["checks"]["no_interior_holes"]
    assert not bad_rep["checks"]["connected_components"]


def test_qc_volume_threshold():
    good, _ = phantom.make_brain_mask_pair()
    vol = good.astype(np.float32) * 800 + 50
    cfg = Config(input_path="", output_dir="", icv_min_cc=1e6, icv_max_cc=1e9)
    rep = qc.qc_report(good, vol, [1, 1, 1], cfg)
    assert not rep["checks"]["intracranial_volume_cc"]   # volume too small -> flagged


def _clean_t1_ball(shape=(80, 80, 80)):
    """A fair morphology case: bright brain ball, dark skull ring, mid scalp,
    isotropic, no bias. Concentric so the dark skull gap cleanly separates
    brain from scalp the way the algorithm assumes."""
    c = np.array(shape) / 2
    zz, yy, xx = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    r = np.sqrt((zz - c[0])**2 + (yy - c[1])**2 + (xx - c[2])**2)
    vol = np.zeros(shape, np.float32)
    # T1 intensity ordering: fat/scalp brightest, brain mid, skull/bone dark.
    # The morphology upper band bound exists to drop the bright fat.
    vol[r < 38] = 950.0          # scalp / subcutaneous fat (brightest)
    vol[r < 32] = 40.0           # skull void (8-voxel gap: not bridged by close_r=3)
    vol[r < 24] = 600.0          # brain (mid-bright, uniform)
    brain = r < 24
    return vol, np.array([1.0, 1.0, 1.0]), np.eye(3), brain


def test_morphology_engine_extracts_brain():
    vol, sp, dirs, brain = _clean_t1_ball()
    head = np.ones(vol.shape, bool)
    mask, name = extract.morphology_extract(vol, sp, dirs, head=head)
    assert name == "morphology"
    # single blob, brain-located, recovers most of it without leaking to scalp
    assert extract.largest_component(mask)[1] == int(mask.sum())   # one component
    inter = (mask & brain).sum()
    assert inter > 0.7 * brain.sum()
    assert mask.sum() < 1.8 * brain.sum()


def test_engine_dispatch_falls_back():
    vol, sp, dirs, _ = phantom.make_phantom(shape=(48, 64, 64))
    head = np.ones(vol.shape, bool)
    mask, name = extract.extract(vol, sp, dirs, engine="morphology", head=head)
    assert mask.dtype == bool and mask.any()
    assert "morphology" in name


def test_fastsurfer_engine_registered_and_gated(monkeypatch):
    # FastSurfer is a first-class engine but only "available" when configured.
    assert "fastsurfer" in extract._ENGINES
    monkeypatch.delenv("FASTSURFER_HOME", raising=False)
    assert extract.fastsurfer_available() is False
    monkeypatch.setenv("FASTSURFER_HOME", "/nonexistent")
    assert extract.fastsurfer_available() is False   # env set but no run_prediction.py


@pytest.mark.slow
def test_end_to_end_reconstruction(tmp_path):
    import nibabel as nib
    vol, sp, dirs, _ = phantom.make_phantom(shape=(64, 96, 96))
    aff = affine_from_dirs(dirs, sp, vol.shape)
    p = tmp_path / "phantom.nii.gz"
    nib.save(nib.Nifti1Image(np.transpose(vol, (2, 1, 0)), aff), str(p))

    cfg = Config(input_path=str(p), output_dir=str(tmp_path / "out"),
                 engine="morphology", iso_mm=1.0, do_bias_correction=False,
                 icv_min_cc=0, icv_max_cc=1e9, min_boundary_overlap=0.0,
                 verbose=False)
    from brainmodel import pipeline
    manifest = pipeline.run(cfg)
    assert (tmp_path / "out" / "brain.glb").exists()
    assert (tmp_path / "out" / "skin.stl").exists()
    assert manifest["meshes"]["brain"]["faces"] > 100
