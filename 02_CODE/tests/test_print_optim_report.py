"""
Tests for hfe_utils/print_optim_report.py

Covers:
- OR_ult_load_disp
- compute_tissue_mineralization
- compute_bone_volume
- compute_bone_report_variables_no_psl
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from hfe_utils.print_optim_report import (
    OR_ult_load_disp,
    compute_bone_volume,
    compute_bone_report_variables_no_psl,
    compute_tissue_mineralization,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_optim(loadcase, disp_rows, force_rows):
    """Return a minimal optim dict for OR_ult_load_disp."""
    return {
        f"disp_{loadcase}": disp_rows,
        f"force_{loadcase}": force_rows,
    }


def make_bone_dict(size=(10, 10, 10), spacing=0.1):
    """Create a minimal bone dict with mask arrays."""
    cort = np.zeros(size, dtype=float)
    trab = np.zeros(size, dtype=float)
    # Fill cortical region (outer shell) and trabecular region (inner cube)
    cort[0:2, :, :] = 1.0
    cort[-2:, :, :] = 1.0
    trab[2:8, 2:8, 2:8] = 1.0
    return {
        "CORTMASK_array": cort.copy(),
        "TRABMASK_array": trab.copy(),
        "Spacing": np.array([spacing, spacing, spacing]),
    }


# ---------------------------------------------------------------------------
# OR_ult_load_disp
# ---------------------------------------------------------------------------


class TestORUltLoadDisp:
    def test_basic_fz_max(self):
        """Max force identified correctly for FZ_MAX loadcase."""
        disp = [
            [0, 0, 1.0, 0, 0, 0],
            [0, 0, 2.0, 0, 0, 0],
            [0, 0, 3.0, 0, 0, 0],
        ]
        force = [
            [0, 0, 100.0, 0, 0, 0],
            [0, 0, 200.0, 0, 0, 0],
            [0, 0, 150.0, 0, 0, 0],
        ]
        optim = make_optim("FZ_MAX", disp, force)
        result = OR_ult_load_disp(optim, "FZ_MAX")
        max_force_entry, disp_at_max = result["max_force_disp_FZ_MAX"]
        assert max_force_entry[2] == pytest.approx(200.0)
        assert disp_at_max == pytest.approx(2.0)

    def test_returns_optim_dict(self):
        disp = [[0, 0, 1.0, 0, 0, 0]]
        force = [[0, 0, 50.0, 0, 0, 0]]
        optim = make_optim("FZ_MAX", disp, force)
        result = OR_ult_load_disp(optim, "FZ_MAX")
        assert "max_force_disp_FZ_MAX" in result
        assert "max_moment_disp_FZ_MAX" in result

    def test_single_increment(self):
        disp = [[0.1, 0.2, 0.5, 0, 0, 0]]
        force = [[5.0, 10.0, 80.0, 1.0, 2.0, 3.0]]
        optim = make_optim("FZ_MAX", disp, force)
        result = OR_ult_load_disp(optim, "FZ_MAX")
        max_f, disp_max_f = result["max_force_disp_FZ_MAX"]
        assert max_f[2] == pytest.approx(80.0)
        assert disp_max_f == pytest.approx(0.5)

    def test_multiple_equal_max_forces(self):
        """When several entries share the maximum value, no exception is raised."""
        disp = [
            [0, 0, 1.0, 0, 0, 0],
            [0, 0, 2.0, 0, 0, 0],
            [0, 0, 3.0, 0, 0, 0],
        ]
        force = [
            [0, 0, 200.0, 0, 0, 0],
            [0, 0, 200.0, 0, 0, 0],
            [0, 0, 100.0, 0, 0, 0],
        ]
        optim = make_optim("FZ_MAX", disp, force)
        # Should not raise
        result = OR_ult_load_disp(optim, "FZ_MAX")
        assert "max_force_disp_FZ_MAX" in result

    def test_fx_loadcase_uses_correct_index(self):
        """FX uses disp index 0."""
        disp = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]
        force = [[90.0, 10.0, 5.0, 1.0, 1.0, 1.0]]
        optim = make_optim("FX", disp, force)
        result = OR_ult_load_disp(optim, "FX")
        _, disp_at_max = result["max_force_disp_FX"]
        # Dict index for FX is 0
        assert disp_at_max == pytest.approx(1.0)

    def test_my_loadcase_uses_correct_index(self):
        """MY uses disp index 4."""
        disp = [[1.0, 2.0, 3.0, 4.0, 5.5, 6.0]]
        force = [[0, 0, 0, 0, 50.0, 0]]
        optim = make_optim("MY", disp, force)
        result = OR_ult_load_disp(optim, "MY")
        _, disp_at_max = result["max_force_disp_MY"]
        assert disp_at_max == pytest.approx(5.5)

    def test_max_moment_stored(self):
        disp = [
            [0, 0, 1.0, 0, 0, 0],
            [0, 0, 2.0, 0, 0, 0],
        ]
        force = [
            [5.0, 5.0, 100.0, 10.0, 20.0, 30.0],
            [5.0, 5.0, 80.0, 50.0, 60.0, 70.0],
        ]
        optim = make_optim("FZ_MAX", disp, force)
        result = OR_ult_load_disp(optim, "FZ_MAX")
        assert "max_moment_disp_FZ_MAX" in result


# ---------------------------------------------------------------------------
# compute_tissue_mineralization
# ---------------------------------------------------------------------------


class TestComputeTissueMineralization:
    def test_basic_computation(self):
        size = (5, 5, 5)
        bone = make_bone_dict(size=size)
        SEG_array = np.ones(size, dtype=float)
        BMD_array = np.full(size, 500.0, dtype=float)
        bone = compute_tissue_mineralization(bone, SEG_array.copy(), BMD_array.copy(), "orig")
        assert "mean_BMD_SEG_CORTorig" in bone
        assert "mean_BMD_SEG_TRABorig" in bone

    def test_cortical_bmd_finite(self):
        size = (6, 6, 6)
        bone = make_bone_dict(size=size)
        SEG_array = np.ones(size, dtype=float)
        BMD_array = np.full(size, 300.0, dtype=float)
        bone = compute_tissue_mineralization(bone, SEG_array.copy(), BMD_array.copy(), "test")
        assert np.isfinite(bone["mean_BMD_SEG_CORTtest"])

    def test_trabecular_bmd_finite(self):
        size = (6, 6, 6)
        bone = make_bone_dict(size=size)
        SEG_array = np.ones(size, dtype=float)
        BMD_array = np.full(size, 300.0, dtype=float)
        bone = compute_tissue_mineralization(bone, SEG_array.copy(), BMD_array.copy(), "test")
        assert np.isfinite(bone["mean_BMD_SEG_TRABtest"])

    def test_zero_seg_gives_nan(self):
        """All-zero SEG produces NaN because there are no voxels where SEG == 1."""
        size = (5, 5, 5)
        bone = make_bone_dict(size=size)
        SEG_array = np.zeros(size, dtype=float)
        BMD_array = np.full(size, 300.0, dtype=float)
        bone = compute_tissue_mineralization(bone, SEG_array.copy(), BMD_array.copy(), "zero")
        assert np.isnan(bone["mean_BMD_SEG_CORTzero"])
        assert np.isnan(bone["mean_BMD_SEG_TRABzero"])

    def test_different_string_suffix(self):
        size = (5, 5, 5)
        bone = make_bone_dict(size=size)
        SEG_array = np.ones(size, dtype=float)
        BMD_array = np.full(size, 200.0, dtype=float)
        bone = compute_tissue_mineralization(bone, SEG_array.copy(), BMD_array.copy(), "scaled")
        assert "mean_BMD_SEG_CORTscaled" in bone
        assert "mean_BMD_SEG_TRABscaled" in bone

    def test_cortical_and_trabecular_different_when_masks_differ(self):
        size = (10, 10, 10)
        bone = {
            "CORTMASK_array": np.zeros((10, 10, 10), dtype=float),
            "TRABMASK_array": np.zeros((10, 10, 10), dtype=float),
            "Spacing": np.array([0.1, 0.1, 0.1]),
        }
        # Cortical voxels have BMD 800, trabecular have BMD 200
        bone["CORTMASK_array"][:2, :, :] = 1.0
        bone["TRABMASK_array"][5:8, 5:8, 5:8] = 1.0
        SEG_array = np.ones((10, 10, 10), dtype=float)
        BMD_array = np.zeros((10, 10, 10), dtype=float)
        BMD_array[:2, :, :] = 800.0
        BMD_array[5:8, 5:8, 5:8] = 200.0
        bone = compute_tissue_mineralization(bone, SEG_array.copy(), BMD_array.copy(), "sep")
        assert bone["mean_BMD_SEG_CORTsep"] != bone["mean_BMD_SEG_TRABsep"]


# ---------------------------------------------------------------------------
# compute_bone_volume
# ---------------------------------------------------------------------------


class TestComputeBoneVolume:
    def test_basic_volume_computed(self):
        size = (5, 5, 5)
        bone = make_bone_dict(size=size, spacing=0.1)
        SEG_array = np.ones(size, dtype=float)
        bone = compute_bone_volume(bone, SEG_array.copy())
        assert "BV_CORT_SEG" in bone
        assert "BV_TRAB_SEG" in bone

    def test_volumes_non_negative(self):
        size = (8, 8, 8)
        bone = make_bone_dict(size=size, spacing=0.2)
        SEG_array = np.ones(size, dtype=float)
        bone = compute_bone_volume(bone, SEG_array.copy())
        assert bone["BV_CORT_SEG"] >= 0
        assert bone["BV_TRAB_SEG"] >= 0

    def test_zero_seg_gives_zero_volume(self):
        size = (5, 5, 5)
        bone = make_bone_dict(size=size, spacing=0.1)
        SEG_array = np.zeros(size, dtype=float)
        bone = compute_bone_volume(bone, SEG_array.copy())
        assert bone["BV_CORT_SEG"] == pytest.approx(0.0)
        assert bone["BV_TRAB_SEG"] == pytest.approx(0.0)

    def test_volume_scales_with_spacing(self):
        """Doubling spacing → volume increases by 2^3 = 8x."""
        size = (6, 6, 6)
        SEG_array = np.ones(size, dtype=float)

        bone1 = make_bone_dict(size=size, spacing=0.1)
        bone1 = compute_bone_volume(bone1, SEG_array.copy())

        bone2 = make_bone_dict(size=size, spacing=0.2)
        bone2 = compute_bone_volume(bone2, SEG_array.copy())

        if bone1["BV_CORT_SEG"] > 0:
            ratio = bone2["BV_CORT_SEG"] / bone1["BV_CORT_SEG"]
            assert ratio == pytest.approx(8.0, rel=1e-6)

    def test_volume_matches_expected_count(self):
        """BV_CORT_SEG = n_cort_seg_voxels * voxel_volume."""
        spacing = 0.5
        voxel_vol = spacing ** 3
        size = (6, 6, 6)

        cort = np.zeros(size, dtype=float)
        trab = np.zeros(size, dtype=float)
        cort[:, :, 0] = 1.0  # 36 voxels in cortical

        bone = {
            "CORTMASK_array": cort,
            "TRABMASK_array": trab,
            "Spacing": np.array([spacing, spacing, spacing]),
        }
        SEG_array = np.ones(size, dtype=float)
        bone = compute_bone_volume(bone, SEG_array.copy())
        assert bone["BV_CORT_SEG"] == pytest.approx(36 * voxel_vol)


# ---------------------------------------------------------------------------
# compute_bone_report_variables_no_psl (integration)
# ---------------------------------------------------------------------------


class TestComputeBoneReportVariablesNoPsl:
    def _make_full_bone(self, size=(8, 8, 8)):
        cort = np.zeros(size, dtype=float)
        trab = np.zeros(size, dtype=float)
        cort[:2, :, :] = 1.0
        trab[2:6, 2:6, 2:6] = 1.0
        bmd = np.random.default_rng(42).uniform(100, 800, size).astype(float)
        return {
            "CORTMASK_array": cort.copy(),
            "TRABMASK_array": trab.copy(),
            "Spacing": np.array([0.1, 0.1, 0.1]),
            "SEG_array": np.ones(size, dtype=float),
            "BMD_array": bmd.copy(),
            "BMDscaled": bmd.copy() * 1.05,
        }

    def test_runs_without_error(self):
        bone = self._make_full_bone()
        result = compute_bone_report_variables_no_psl(bone)
        assert result is not None

    def test_expected_keys_present(self):
        bone = self._make_full_bone()
        result = compute_bone_report_variables_no_psl(bone)
        assert "BV_CORT_SEG" in result
        assert "BV_TRAB_SEG" in result
        assert "mean_BMD_SEG_CORTorig" in result
        assert "mean_BMD_SEG_TRABorig" in result

    def test_scaled_and_orig_may_differ(self):
        bone = self._make_full_bone()
        result = compute_bone_report_variables_no_psl(bone)
        # The scaled BMD (1.05×) should produce slightly different mean
        assert result["mean_BMD_SEG_CORTscaled"] != pytest.approx(
            result["mean_BMD_SEG_CORTorig"], rel=1e-3
        )
