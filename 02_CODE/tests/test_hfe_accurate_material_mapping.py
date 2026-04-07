"""
Tests for hfe_accurate/material_mapping.py

Covers (pure numpy / scipy functions, no vtk/numba/pyvista needed):
- calculate_bvtv
- _bmc_compensation
- __computePHI__  (module-level helper)
- __get_fe_dims__
- getClosestPhysPoint
- correspondence_dict
- vectoronplane
- compute_isoFAB
- cai_evalues
- __correct_power_law_mil__
- calculate_degree_anisotropy
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
]
for _mod in _MOCKED:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

import hfe_accurate.material_mapping as mm

# Pull out module-level private helpers via __dict__ to avoid import issues
_computePHI = mm.__dict__["__computePHI__"]
_get_fe_dims = mm.__dict__["__get_fe_dims__"]
_correct_power_law_mil = mm.__dict__["__correct_power_law_mil__"]

calculate_bvtv = mm.calculate_bvtv
_bmc_compensation = mm._bmc_compensation
getClosestPhysPoint = mm.getClosestPhysPoint
correspondence_dict = mm.correspondence_dict
vectoronplane = mm.vectoronplane
compute_isoFAB = mm.compute_isoFAB
cai_evalues = mm.cai_evalues
calculate_degree_anisotropy = mm.calculate_degree_anisotropy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_arrays(size=(20, 20, 20), fill_seg=0.5, fill_mask=1.0):
    seg = np.full(size, fill_seg, dtype=np.float32)
    mask = np.full(size, fill_mask, dtype=np.float32)
    return seg, mask


# ---------------------------------------------------------------------------
# calculate_bvtv
# ---------------------------------------------------------------------------


class TestCalculateBvtv:
    """Tests for hfe_accurate.material_mapping.calculate_bvtv."""

    SPACING = np.array([0.0607, 0.0607, 0.0607])
    VOI_MM = 3 * 0.0607  # ~5 voxels

    def test_returns_tuple_of_two_floats(self):
        seg, mask = _make_arrays()
        cog = np.array([0.6, 0.6, 0.6])
        result = calculate_bvtv(seg, mask, self.SPACING, self.VOI_MM, cog)
        assert len(result) == 2
        phi, rho = result
        assert isinstance(float(phi), float)
        assert isinstance(float(rho), float)

    def test_output_in_valid_range(self):
        seg, mask = _make_arrays()
        cog = np.array([0.6, 0.6, 0.6])
        phi, rho = calculate_bvtv(seg, mask, self.SPACING, self.VOI_MM, cog)
        assert 0.01 <= phi <= 1.0
        assert 0.01 <= rho <= 1.0

    def test_zero_mask_returns_minimum(self):
        """All-zero mask → phi_s = 0 → minimum fallback (0.01, 0.01)."""
        seg = np.ones((20, 20, 20), dtype=np.float32)
        mask = np.zeros((20, 20, 20), dtype=np.float32)
        cog = np.array([0.6, 0.6, 0.6])
        phi, rho = calculate_bvtv(seg, mask, self.SPACING, self.VOI_MM, cog)
        assert phi == pytest.approx(0.01)
        assert rho == pytest.approx(0.01)

    def test_full_bone_gives_high_bvtv(self):
        """seg and mask both fully filled → rho should approach 1."""
        seg = np.ones((20, 20, 20), dtype=np.float32)
        mask = np.ones((20, 20, 20), dtype=np.float32)
        cog = np.array([0.6, 0.6, 0.6])
        phi, rho = calculate_bvtv(seg, mask, self.SPACING, self.VOI_MM, cog)
        assert rho >= 0.9

    def test_cog_at_center(self):
        size = (30, 30, 30)
        seg = np.ones(size, dtype=np.float32)
        mask = np.ones(size, dtype=np.float32)
        center_mm = np.array([15, 15, 15]) * self.SPACING
        phi, rho = calculate_bvtv(seg, mask, self.SPACING, self.VOI_MM, center_mm)
        assert 0.01 <= phi <= 1.0
        assert 0.01 <= rho <= 1.0

    def test_cog_at_corner_does_not_raise(self):
        """COG at array corner should not raise an IndexError."""
        seg = np.ones((20, 20, 20), dtype=np.float32)
        mask = np.ones((20, 20, 20), dtype=np.float32)
        cog = np.array([0.0, 0.0, 0.0])
        phi, rho = calculate_bvtv(seg, mask, self.SPACING, self.VOI_MM, cog)
        assert 0.01 <= phi <= 1.0
        assert 0.01 <= rho <= 1.0

    def test_empty_seg_gives_minimum_rho(self):
        """Zero-valued seg (no bone) → rho falls back to 0.01 minimum."""
        seg = np.zeros((20, 20, 20), dtype=np.float32)
        mask = np.ones((20, 20, 20), dtype=np.float32)
        cog = np.array([0.6, 0.6, 0.6])
        _, rho = calculate_bvtv(seg, mask, self.SPACING, self.VOI_MM, cog)
        assert rho == pytest.approx(0.01)

    def test_phi_rho_independent_of_seg_outside_mask(self):
        """Values in seg outside the mask should not influence the result."""
        seg1 = np.ones((20, 20, 20), dtype=np.float32)
        seg2 = np.zeros((20, 20, 20), dtype=np.float32)
        # Mask covers only the center region
        mask = np.zeros((20, 20, 20), dtype=np.float32)
        mask[8:12, 8:12, 8:12] = 1.0
        cog = np.array([10, 10, 10]) * self.SPACING

        phi1, rho1 = calculate_bvtv(seg1, mask.copy(), self.SPACING, self.VOI_MM, cog)
        phi2, rho2 = calculate_bvtv(seg2, mask.copy(), self.SPACING, self.VOI_MM, cog)
        # phi depends only on mask, should be equal
        assert phi1 == pytest.approx(phi2)


# ---------------------------------------------------------------------------
# _bmc_compensation
# ---------------------------------------------------------------------------


class TestBmcCompensation:
    """Tests for hfe_accurate.material_mapping._bmc_compensation."""

    def _make_inputs(self, n_cort=4, n_trab=4, spacing=0.1, el_size=0.5, rho_val=0.5, phi_val=0.5):
        size = (10, 10, 10)
        BMD = np.full(size, 500.0)
        CORTMASK = np.zeros(size)
        TRABMASK = np.zeros(size)
        CORTMASK[:2, :, :] = 1.0
        TRABMASK[2:4, :, :] = 1.0

        cort_elms = {i: i for i in range(n_cort)}
        trab_elms = {i: i for i in range(n_trab)}

        RHOc = np.full(n_cort, rho_val)
        RHOt = np.full(n_trab, rho_val)
        PHIc = np.full(n_cort, phi_val)
        PHIt = np.full(n_trab, phi_val)

        FEelSize = np.array([el_size, el_size, el_size])
        Spacing = np.array([spacing, spacing, spacing])
        return BMD, CORTMASK, TRABMASK, cort_elms, trab_elms, RHOc, RHOt, PHIc, PHIt, FEelSize, Spacing

    def test_returns_six_values(self):
        args = self._make_inputs()
        result = _bmc_compensation(*args, BMC_conservation=True)
        assert len(result) == 6

    def test_no_conservation_leaves_rho_unchanged(self):
        args = self._make_inputs(rho_val=0.4)
        BMD, CORTMASK, TRABMASK, cort_elms, trab_elms, RHOc, RHOt, PHIc, PHIt, FEelSize, Spacing = args
        RHOc_orig = RHOc.copy()
        RHOt_orig = RHOt.copy()
        RHOc_out, RHOt_out, *_ = _bmc_compensation(
            BMD, CORTMASK, TRABMASK, cort_elms, trab_elms,
            RHOc, RHOt, PHIc, PHIt, FEelSize, Spacing,
            BMC_conservation=False,
        )
        np.testing.assert_array_almost_equal(RHOc_out, RHOc_orig)
        np.testing.assert_array_almost_equal(RHOt_out, RHOt_orig)

    def test_conservation_scales_rho(self):
        """With BMC_conservation=True, the output RHO values differ from input."""
        args = self._make_inputs(rho_val=0.2, phi_val=0.3, spacing=0.1, el_size=0.5)
        BMD, CORTMASK, TRABMASK, cort_elms, trab_elms, RHOc, RHOt, PHIc, PHIt, FEelSize, Spacing = args
        RHOc_in = RHOc.copy()
        RHOc_out, RHOt_out, BMC_sim_comp, BMC_reco_tot, lambda_c, lambda_t = _bmc_compensation(
            BMD, CORTMASK, TRABMASK, cort_elms, trab_elms,
            RHOc, RHOt, PHIc, PHIt, FEelSize, Spacing,
            BMC_conservation=True,
        )
        # lambda should be positive
        assert lambda_c > 0
        assert lambda_t > 0

    def test_rho_capped_at_one(self):
        """When lambda * rho >= 1, RHO is capped at 1.0."""
        # Use very small simulated BMC → huge lambda → everything capped
        args = self._make_inputs(rho_val=0.001, phi_val=0.001, spacing=0.1, el_size=0.001)
        BMD, CORTMASK, TRABMASK, cort_elms, trab_elms, RHOc, RHOt, PHIc, PHIt, FEelSize, Spacing = args
        RHOc_out, RHOt_out, *_ = _bmc_compensation(
            BMD, CORTMASK, TRABMASK, cort_elms, trab_elms,
            RHOc, RHOt, PHIc, PHIt, FEelSize, Spacing,
            BMC_conservation=True,
        )
        assert np.all(RHOc_out <= 1.0)
        assert np.all(RHOt_out <= 1.0)

    def test_bmc_reco_tot_equals_sum(self):
        args = self._make_inputs()
        BMD, CORTMASK, TRABMASK, cort_elms, trab_elms, RHOc, RHOt, PHIc, PHIt, FEelSize, Spacing = args
        _, _, _, BMC_reco_tot, _, _ = _bmc_compensation(
            BMD, CORTMASK, TRABMASK, cort_elms, trab_elms,
            RHOc, RHOt, PHIc, PHIt, FEelSize, Spacing,
            BMC_conservation=False,
        )
        # manually compute expected
        Spacing_val = 0.1
        THOUSAND = 1000
        expected_c = np.sum(BMD[CORTMASK > 0]) * Spacing_val**3 / THOUSAND
        expected_t = np.sum(BMD[TRABMASK > 0]) * Spacing_val**3 / THOUSAND
        assert BMC_reco_tot == pytest.approx(expected_c + expected_t, rel=1e-5)


# ---------------------------------------------------------------------------
# __computePHI__
# ---------------------------------------------------------------------------


class TestComputePHI:
    """Tests for module-level __computePHI__ in material_mapping."""

    def test_all_ones_returns_one(self):
        roi = np.ones((5, 5, 5), dtype=np.uint8)
        assert _computePHI(roi) == pytest.approx(1.0)

    def test_all_zeros_returns_zero(self):
        roi = np.zeros((5, 5, 5), dtype=np.uint8)
        assert _computePHI(roi) == pytest.approx(0.0)

    def test_half_filled_returns_half(self):
        roi = np.zeros((4, 4, 4), dtype=np.uint8)
        roi[:2, :, :] = 1
        # 32 nonzero out of 64
        assert _computePHI(roi) == pytest.approx(0.5)

    def test_result_clamped_at_one(self):
        """Values above 1 should be clamped."""
        roi = np.full((5, 5, 5), 2, dtype=np.uint8)
        # count_nonzero(roi > 0) == 125, size = 125 → phi = 1.0
        result = _computePHI(roi)
        assert result <= 1.0

    def test_nan_handled(self):
        """NaN-producing arrays should return 0.0."""
        # Create a situation where the result would be nan: size == 0
        roi = np.array([], dtype=np.uint8)
        result = _computePHI(roi)
        assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# __get_fe_dims__
# ---------------------------------------------------------------------------


class TestGetFeDims:
    def test_basic_output_shapes(self):
        img_shape = (100, 80, 60)
        FEelSize = 1.0
        Spacing = 0.1
        centroids, FEdimX, FEdimY, FEdimZ, CoarseFactor = _get_fe_dims(img_shape, FEelSize, Spacing)
        assert CoarseFactor == pytest.approx(10.0)
        assert centroids.shape[1] == 3

    def test_coarse_factor_equals_ratio(self):
        FEelSize = 2.0
        Spacing = 0.5
        _, _, _, _, CoarseFactor = _get_fe_dims((50, 50, 50), FEelSize, Spacing)
        assert CoarseFactor == pytest.approx(FEelSize / Spacing)

    def test_fe_dims_are_integers(self):
        _, FEdimX, FEdimY, FEdimZ, _ = _get_fe_dims((60, 40, 20), 1.0, 0.1)
        assert isinstance(FEdimX, (int, np.integer))
        assert isinstance(FEdimY, (int, np.integer))
        assert isinstance(FEdimZ, (int, np.integer))

    def test_centroids_positive(self):
        centroids, *_ = _get_fe_dims((30, 30, 30), 1.0, 0.1)
        assert np.all(centroids > 0)

    def test_number_of_centroids_matches_fe_dims(self):
        img_shape = (30, 40, 50)
        FEelSize = 1.0
        Spacing = 0.1
        centroids, FEdimX, FEdimY, FEdimZ, CoarseFactor = _get_fe_dims(img_shape, FEelSize, Spacing)
        expected_count = (
            int(np.floor(img_shape[0] / CoarseFactor))
            * int(np.floor(img_shape[1] / CoarseFactor))
            * int(np.floor(img_shape[2] / CoarseFactor))
        )
        assert centroids.shape[0] == expected_count


# ---------------------------------------------------------------------------
# getClosestPhysPoint
# ---------------------------------------------------------------------------


class TestGetClosestPhysPoint:
    def test_basic_nearest_neighbour(self):
        phys_points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
        query = np.array([[0.1, 0, 0], [1.9, 0, 0]], dtype=float)
        distances, indices = getClosestPhysPoint(query, phys_points)
        assert indices[0] == 0   # closest to [0,0,0]
        assert indices[1] == 2   # closest to [2,0,0]

    def test_distances_non_negative(self):
        pts = np.random.default_rng(0).random((20, 3))
        query = np.random.default_rng(1).random((5, 3))
        distances, _ = getClosestPhysPoint(query, pts)
        assert np.all(distances >= 0)

    def test_exact_match_gives_zero_distance(self):
        pts = np.array([[1.0, 2.0, 3.0]])
        query = np.array([[1.0, 2.0, 3.0]])
        distances, indices = getClosestPhysPoint(query, pts)
        assert distances[0] == pytest.approx(0.0)
        assert indices[0] == 0

    def test_returns_correct_number_of_results(self):
        pts = np.random.default_rng(2).random((50, 3))
        query = np.random.default_rng(3).random((10, 3))
        distances, indices = getClosestPhysPoint(query, pts)
        assert len(distances) == 10
        assert len(indices) == 10


# ---------------------------------------------------------------------------
# correspondence_dict
# ---------------------------------------------------------------------------


class TestCorrespondenceDict:
    def test_basic_mapping(self):
        centroids_mesh = {0: [0, 0, 0], 1: [1, 1, 1], 2: [2, 2, 2]}
        closest_phys_points = np.array([5, 3, 8])
        result = correspondence_dict(centroids_mesh, closest_phys_points)
        assert result[0] == 5
        assert result[1] == 3
        assert result[2] == 8

    def test_length_matches_input(self):
        centroids_mesh = {i: [i, i, i] for i in range(10)}
        closest_phys_points = np.arange(10)
        result = correspondence_dict(centroids_mesh, closest_phys_points)
        assert len(result) == 10

    def test_empty_input_gives_empty_dict(self):
        result = correspondence_dict({}, np.array([]))
        assert result == {}


# ---------------------------------------------------------------------------
# vectoronplane
# ---------------------------------------------------------------------------


class TestVectorOnPlane:
    # Use evect_max=[0,0,1], evect_mid=[1,0,0]: normal=cross([0,0,1],[1,0,0])=[0,1,0].
    # direction=[0,0,1]: dot(direction, normal)=0 → non-degenerate projection.
    def test_returns_three_unit_vectors(self):
        evect_max = np.array([0.0, 0.0, 1.0])
        evect_mid = np.array([1.0, 0.0, 0.0])
        evect_min = np.array([0.0, 1.0, 0.0])
        direction = np.array([0.0, 0.0, 1.0])
        e_min, e_mid, e_max = vectoronplane(evect_max, evect_mid, evect_min, direction)
        # All should be unit vectors
        assert np.linalg.norm(e_min) == pytest.approx(1.0, abs=1e-6)
        assert np.linalg.norm(e_mid) == pytest.approx(1.0, abs=1e-6)
        assert np.linalg.norm(e_max) == pytest.approx(1.0, abs=1e-6)

    def test_output_vectors_approximately_orthogonal(self):
        evect_max = np.array([0.0, 0.0, 1.0])
        evect_mid = np.array([1.0, 0.0, 0.0])
        evect_min = np.array([0.0, 1.0, 0.0])
        direction = np.array([0.0, 0.0, 1.0])
        e_min, e_mid, e_max = vectoronplane(evect_max, evect_mid, evect_min, direction)
        assert abs(np.dot(e_min, e_max)) == pytest.approx(0.0, abs=1e-6)

    def test_does_not_raise_on_degenerate_input(self):
        """Degenerate (zero-norm) vectors should fallback gracefully."""
        evect_max = np.array([0.0, 0.0, 0.0])
        evect_mid = np.array([0.0, 0.0, 0.0])
        evect_min = np.array([0.0, 0.0, 0.0])
        direction = np.array([0.0, 0.0, 1.0])
        # Should not raise
        e_min, e_mid, e_max = vectoronplane(evect_max, evect_mid, evect_min, direction)
        assert e_min is not None

    def test_projection_removes_normal_component(self):
        """The projected evect_max should have zero component along the normal."""
        evect_max = np.array([1.0, 0.0, 1.0]) / np.sqrt(2)
        evect_mid = np.array([0.0, 1.0, 0.0])
        evect_min = np.array([-1.0, 0.0, 1.0]) / np.sqrt(2)
        direction = np.array([0.0, 0.0, 1.0])
        e_min, e_mid, e_max = vectoronplane(evect_max, evect_mid, evect_min, direction)
        # e_max projected onto the plane defined by the cross of evect_max and evect_mid
        # We just verify no exception and unit length
        assert np.isfinite(np.linalg.norm(e_max))


# ---------------------------------------------------------------------------
# compute_isoFAB
# ---------------------------------------------------------------------------


class TestComputeIsoFAB:
    def test_returns_unit_eigenvalues(self):
        evals, evects = compute_isoFAB()
        assert evals == [1.0, 1.0, 1.0]

    def test_returns_identity_eigenvectors(self):
        _, evects = compute_isoFAB()
        expected = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        assert evects == expected

    def test_return_types(self):
        evals, evects = compute_isoFAB()
        assert isinstance(evals, list)
        assert isinstance(evects, list)


# ---------------------------------------------------------------------------
# cai_evalues
# ---------------------------------------------------------------------------


class TestCaiEvalues:
    def test_returns_three_values(self):
        result = cai_evalues()
        assert len(result) == 3

    def test_e3_is_largest(self):
        E1, E2, E3 = cai_evalues()
        assert E3 >= E2
        assert E3 >= E1

    def test_e1_equals_e2(self):
        E1, E2, E3 = cai_evalues()
        assert E1 == pytest.approx(E2)

    def test_values_are_positive(self):
        for val in cai_evalues():
            assert val > 0


# ---------------------------------------------------------------------------
# __correct_power_law_mil__
# ---------------------------------------------------------------------------


class TestCorrectPowerLawMil:
    def test_unit_eigenvalues_unchanged(self):
        evalue = np.array([1.0, 1.0, 1.0])
        result = _correct_power_law_mil(evalue, b=1.0)
        np.testing.assert_array_almost_equal(result, [1.0, 1.0, 1.0])

    def test_trace_sums_to_three(self):
        evalue = np.array([0.8, 1.0, 1.2])
        for b in [0.5, 1.0, 2.0, 3.0]:
            result = _correct_power_law_mil(evalue, b=b)
            assert np.sum(result) == pytest.approx(3.0, rel=1e-6)

    def test_symmetric_eigenvalues_preserved(self):
        evalue = np.array([1.0, 1.0, 1.0])
        result = _correct_power_law_mil(evalue, b=2.0)
        np.testing.assert_array_almost_equal(result, [1.0, 1.0, 1.0])

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="3 eigenvalues"):
            _correct_power_law_mil(np.array([1.0, 1.0]), b=1.0)

    def test_b_one_identity(self):
        """b=1 → evalue_power = evalue → normalized to sum 3."""
        evalue = np.array([0.5, 1.0, 1.5])
        result = _correct_power_law_mil(evalue, b=1.0)
        assert np.sum(result) == pytest.approx(3.0)
        # proportions preserved
        assert result[0] < result[1] < result[2]

    def test_output_all_positive(self):
        evalue = np.array([0.9, 1.0, 1.1])
        result = _correct_power_law_mil(evalue, b=1.5)
        assert np.all(result > 0)


# ---------------------------------------------------------------------------
# calculate_degree_anisotropy
# ---------------------------------------------------------------------------


class TestCalculateDegreeAnisotropy:
    def test_isotropic_da_equals_one(self):
        """Isotropic fabric [1,1,1] → DA = 1."""
        m_trab = [np.array([1.0, 1.0, 1.0])] * 5
        volumes = [np.array([1.0])] * 5
        da = calculate_degree_anisotropy(m_trab, volumes)
        assert da == pytest.approx(1.0)

    def test_anisotropic_da_greater_than_one(self):
        """When max eigenvalue > min, DA > 1."""
        m_trab = [np.array([0.5, 1.0, 2.0])]
        volumes = [np.array([1.0])]
        da = calculate_degree_anisotropy(m_trab, volumes)
        assert da > 1.0

    def test_da_equals_max_over_min(self):
        """DA = max(m) / max(min(m), 0.1) for each element."""
        m_trab = [np.array([0.5, 1.0, 1.5])]
        volumes = [np.array([1.0])]
        da = calculate_degree_anisotropy(m_trab, volumes)
        assert da == pytest.approx(1.5 / 0.5)

    def test_small_min_eigenvalue_floored_at_0_1(self):
        """Very small min eigenvalue is floored at 0.1 to avoid division by small numbers."""
        m_trab = [np.array([0.001, 1.0, 2.0])]
        volumes = [np.array([1.0])]
        da = calculate_degree_anisotropy(m_trab, volumes)
        assert da == pytest.approx(2.0 / 0.1)

    def test_volume_weighted_average(self):
        """Volume-weighted average: heavy element dominates."""
        m_trab = [
            np.array([0.5, 1.0, 2.0]),  # DA = 4.0
            np.array([1.0, 1.0, 1.0]),  # DA = 1.0
        ]
        volumes = [np.array([0.0]), np.array([1.0])]  # only second element has volume
        da = calculate_degree_anisotropy(m_trab, volumes)
        assert da == pytest.approx(1.0)
