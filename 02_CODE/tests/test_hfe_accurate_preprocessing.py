"""
Tests for hfe_accurate/preprocessing.py

Covers (pure numpy / OmegaConf, no vtk/numba/pyvista needed):
- calculate_bvtv   (IMTYPE="BMD" and IMTYPE="NATIVE")
- fmt_sanity_check
- __assign_to_mask__  (module-level private helper)
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Mock heavy dependencies BEFORE importing the module under test
_MOCKED = [
    "vtk",
    "vtk.util",
    "vtk.util.numpy_support",
    "vtk.numpy_interface",
    "vtk.numpy_interface.dataset_adapter",
    "numba",
    "pyvista",
    "SimpleITK",
    "fast_simplification",
    "mpl_toolkits",
    "mpl_toolkits.axes_grid1",
    "scipy.ndimage.filters",
]
for _mod in _MOCKED:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest
from omegaconf import OmegaConf

import hfe_accurate.preprocessing as pp

fmt_sanity_check = pp.fmt_sanity_check
calculate_bvtv = pp.calculate_bvtv
_assign_to_mask = pp.__dict__["__assign_to_mask__"]


# ---------------------------------------------------------------------------
# fmt_sanity_check
# ---------------------------------------------------------------------------


class TestFmtSanityCheck:
    def test_ndarray_passed_through_unchanged(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = fmt_sanity_check(arr)
        np.testing.assert_array_equal(result, arr)
        assert isinstance(result, np.ndarray)

    def test_list_converted_to_ndarray(self):
        result = fmt_sanity_check([1, 2, 3])
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_scalar_converted(self):
        result = fmt_sanity_check(5.0)
        assert isinstance(result, np.ndarray)

    def test_tuple_converted(self):
        result = fmt_sanity_check((10, 20, 30))
        assert isinstance(result, np.ndarray)

    def test_2d_list_converted(self):
        data = [[1, 2], [3, 4]]
        result = fmt_sanity_check(data)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)

    def test_none_converted(self):
        result = fmt_sanity_check(None)
        assert isinstance(result, np.ndarray)

    def test_identity_for_zero_array(self):
        arr = np.zeros((3, 3))
        result = fmt_sanity_check(arr)
        np.testing.assert_array_equal(result, arr)


# ---------------------------------------------------------------------------
# calculate_bvtv (preprocessing)
# ---------------------------------------------------------------------------


def _make_cfg(bvtv_scaling=0, bvtv_slope=1.0, bvtv_intercept=0.0):
    return OmegaConf.create(
        {
            "image_processing": {
                "bvtv_scaling": bvtv_scaling,
                "bvtv_slope": bvtv_slope,
                "bvtv_intercept": bvtv_intercept,
            }
        }
    )


class TestPreprocessingCalculateBvtv:
    """Tests for hfe_accurate.preprocessing.calculate_bvtv."""

    SIZE = (10, 10, 10)

    def _make_masks(self, cort_region=None, trab_region=None):
        cort = np.zeros(self.SIZE, dtype=float)
        trab = np.zeros(self.SIZE, dtype=float)
        if cort_region:
            cort[cort_region] = 1.0
        if trab_region:
            trab[trab_region] = 1.0
        return cort, trab

    def test_bmd_imtype_divides_by_1200(self):
        bmd = np.full(self.SIZE, 1200.0)
        cort, trab = self._make_masks((slice(0, 5), slice(None), slice(None)))
        cfg = _make_cfg(bvtv_scaling=0)
        BVTVscaled, BMDscaled, BVTVraw = calculate_bvtv(
            None, None, None, bmd, cort, trab, cfg, "BMD"
        )
        # BVTVraw = BMD / 1200 = 1.0 inside mask
        mask = cort
        assert BVTVraw[mask > 0].mean() == pytest.approx(1.0)

    def test_native_imtype_applies_calibration(self):
        scaling = 2.0
        slope = 0.8
        intercept = 50.0
        native_vals = np.full(self.SIZE, 240.0)  # (240/2)*0.8 + 50 = 146 mg/cc
        cort, trab = self._make_masks((slice(0, 5), slice(None), slice(None)))
        cfg = _make_cfg(bvtv_scaling=0)
        BVTVscaled, BMDscaled, BVTVraw = calculate_bvtv(
            scaling, slope, intercept, native_vals, cort, trab, cfg, "NATIVE"
        )
        expected_bmd = (240.0 / 2.0) * 0.8 + 50.0  # = 146.0
        expected_bvtv = expected_bmd / 1200.0
        assert BVTVraw[cort > 0].mean() == pytest.approx(expected_bvtv, rel=1e-4)

    def test_bvtv_scaling_applied_when_flag_is_one(self):
        bmd = np.full(self.SIZE, 600.0)
        cort, trab = self._make_masks((slice(None), slice(None), slice(None)))
        slope, intercept = 1.5, 0.05
        cfg = _make_cfg(bvtv_scaling=1, bvtv_slope=slope, bvtv_intercept=intercept)
        BVTVscaled, BMDscaled, BVTVraw = calculate_bvtv(
            None, None, None, bmd, cort, trab, cfg, "BMD"
        )
        bvtv_raw_val = 600.0 / 1200.0
        expected_scaled = slope * bvtv_raw_val + intercept
        assert BVTVscaled[cort > 0].mean() == pytest.approx(expected_scaled, rel=1e-4)

    def test_no_scaling_when_flag_is_zero(self):
        bmd = np.full(self.SIZE, 600.0)
        cort = np.ones(self.SIZE, dtype=float)
        trab = np.zeros(self.SIZE, dtype=float)
        cfg = _make_cfg(bvtv_scaling=0)
        BVTVscaled, _, BVTVraw = calculate_bvtv(None, None, None, bmd, cort, trab, cfg, "BMD")
        np.testing.assert_array_almost_equal(BVTVscaled, BVTVraw)

    def test_mask_zeros_outside_bone(self):
        """Voxels outside cort+trab mask should have BVTV = 0."""
        bmd = np.full(self.SIZE, 800.0)
        cort = np.zeros(self.SIZE, dtype=float)
        trab = np.zeros(self.SIZE, dtype=float)
        cort[:3, :, :] = 1.0  # only first 3 slices masked
        cfg = _make_cfg(bvtv_scaling=0)
        BVTVscaled, _, _ = calculate_bvtv(None, None, None, bmd, cort, trab, cfg, "BMD")
        assert BVTVscaled[3:, :, :].sum() == pytest.approx(0.0)

    def test_combined_mask_is_union(self):
        """MASK = CORT + TRAB; both contribute to the output mask."""
        bmd = np.full(self.SIZE, 400.0)
        cort = np.zeros(self.SIZE, dtype=float)
        trab = np.zeros(self.SIZE, dtype=float)
        cort[:3, :, :] = 1.0
        trab[3:6, :, :] = 1.0
        cfg = _make_cfg(bvtv_scaling=0)
        BVTVscaled, _, _ = calculate_bvtv(None, None, None, bmd, cort, trab, cfg, "BMD")
        # Both cortical and trabecular regions should have nonzero BVTV
        assert BVTVscaled[:3, :, :].sum() > 0
        assert BVTVscaled[3:6, :, :].sum() > 0
        assert BVTVscaled[6:, :, :].sum() == pytest.approx(0.0)

    def test_returns_three_arrays(self):
        bmd = np.ones(self.SIZE)
        cort = np.ones(self.SIZE)
        trab = np.zeros(self.SIZE)
        cfg = _make_cfg()
        result = calculate_bvtv(None, None, None, bmd, cort, trab, cfg, "BMD")
        assert len(result) == 3

    def test_bmdscaled_equals_bvtvscaled_times_1200_times_mask(self):
        bmd = np.full(self.SIZE, 600.0)
        cort = np.ones(self.SIZE, dtype=float)
        trab = np.zeros(self.SIZE, dtype=float)
        cfg = _make_cfg(bvtv_scaling=0)
        BVTVscaled, BMDscaled, _ = calculate_bvtv(None, None, None, bmd, cort, trab, cfg, "BMD")
        mask = cort + trab
        mask[mask > 0] = 1
        expected_bmd = BVTVscaled * 1200 * mask
        np.testing.assert_array_almost_equal(BMDscaled, expected_bmd)


# ---------------------------------------------------------------------------
# __assign_to_mask__
# ---------------------------------------------------------------------------


class TestAssignToMask:
    """Tests for the private __assign_to_mask__ helper in preprocessing."""

    def _make_cfg(self, orthotropic_cortex=False):
        return OmegaConf.create(
            {"homogenization": {"orthotropic_cortex": orthotropic_cortex}}
        )

    def test_all_in_trab_mask(self):
        """COG points inside the trabecular mask → all assigned to trab."""
        trabmask = np.ones((10, 10, 10), dtype=np.uint8)
        COG_temp = np.array([
            [5.0, 5.0, 3.0],
            [5.0, 5.0, 5.0],
        ])
        mask_cog = np.array([[5, 5, 3], [5, 5, 5]], dtype=np.int32)
        cfg = self._make_cfg(orthotropic_cortex=False)
        cog_trab, idx_trab, cog_cort, idx_cort = _assign_to_mask(
            cfg, COG_temp, trabmask, mask_cog,
            dimZ_min_tolerance=9.0, tolerance=0.5
        )
        assert len(cog_trab) == 2
        assert len(cog_cort) == 0

    def test_none_in_trab_mask_goes_to_cort(self):
        """COG points outside the trabecular mask → assigned to cort."""
        trabmask = np.zeros((10, 10, 10), dtype=np.uint8)
        COG_temp = np.array([
            [5.0, 5.0, 3.0],
            [5.0, 5.0, 5.0],
        ])
        mask_cog = np.array([[5, 5, 3], [5, 5, 5]], dtype=np.int32)
        cfg = self._make_cfg(orthotropic_cortex=False)
        cog_trab, idx_trab, cog_cort, idx_cort = _assign_to_mask(
            cfg, COG_temp, trabmask, mask_cog,
            dimZ_min_tolerance=9.0, tolerance=0.5
        )
        assert len(cog_trab) == 0
        assert len(cog_cort) == 2

    def test_orthotropic_cortex_returns_none_for_cort(self):
        """When orthotropic_cortex=True, cortical outputs should be None."""
        trabmask = np.ones((10, 10, 10), dtype=np.uint8)
        COG_temp = np.array([[5.0, 5.0, 5.0]])
        mask_cog = np.array([[5, 5, 5]], dtype=np.int32)
        cfg = self._make_cfg(orthotropic_cortex=True)
        cog_trab, idx_trab, cog_cort, idx_cort = _assign_to_mask(
            cfg, COG_temp, trabmask, mask_cog,
            dimZ_min_tolerance=9.0, tolerance=0.5
        )
        assert cog_cort is None
        assert idx_cort is None

    def test_z_tolerance_exclusion(self):
        """Points with z < tolerance or z > dimZ_min_tolerance are excluded."""
        trabmask = np.ones((10, 10, 10), dtype=np.uint8)
        COG_temp = np.array([
            [5.0, 5.0, 0.1],  # z = 0.1 < tolerance=0.5 → excluded
            [5.0, 5.0, 5.0],  # included
            [5.0, 5.0, 9.9],  # z = 9.9 > dimZ_min_tolerance=9.5 → excluded
        ])
        mask_cog = np.array([[5, 5, 0], [5, 5, 5], [5, 5, 9]], dtype=np.int32)
        cfg = self._make_cfg(orthotropic_cortex=False)
        cog_trab, idx_trab, cog_cort, idx_cort = _assign_to_mask(
            cfg, COG_temp, trabmask, mask_cog,
            dimZ_min_tolerance=9.5, tolerance=0.5
        )
        # Only z=5.0 should be included
        assert len(cog_trab) == 1
        assert cog_trab[0][2] == pytest.approx(5.0)
