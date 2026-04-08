"""
Tests for hfe_accurate/struct_voxel_indices.py

Covers (pure numpy functions, numba mocked):
- _range
- _subs
- index_cloud
- create_dict_cloud_to_voxels
- repeating_indices
- areadyadic_indices
- areadyadic_grid_safe
- map_isosurface (integration)
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Mock heavy dependencies BEFORE importing the module under test
_MOCKED = [
    "numba",
    "pyvista",
    "vtk",
    "vtk.util",
    "vtk.util.numpy_support",
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

import hfe_accurate.struct_voxel_indices as svi

_range = svi._range
_subs = svi._subs
index_cloud = svi.index_cloud
create_dict_cloud_to_voxels = svi.create_dict_cloud_to_voxels
repeating_indices = svi.repeating_indices
areadyadic_indices = svi.areadyadic_indices
areadyadic_grid_safe = svi.areadyadic_grid_safe
map_isosurface = svi.map_isosurface


# ---------------------------------------------------------------------------
# _range
# ---------------------------------------------------------------------------


class TestRange:
    def test_basic_2d(self):
        pts = np.array([[1.0, 2.0], [3.0, 0.0], [0.5, 4.0]])
        min_v, max_v = _range(pts)
        np.testing.assert_array_almost_equal(min_v, [0.5, 0.0])
        np.testing.assert_array_almost_equal(max_v, [3.0, 4.0])

    def test_basic_3d(self):
        pts = np.array([[1.0, 2.0, 3.0], [4.0, 0.0, 1.0], [2.0, 5.0, 2.0]])
        min_v, max_v = _range(pts)
        np.testing.assert_array_almost_equal(min_v, [1.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(max_v, [4.0, 5.0, 3.0])

    def test_single_point(self):
        pts = np.array([[7.0, 8.0, 9.0]])
        min_v, max_v = _range(pts)
        np.testing.assert_array_equal(min_v, max_v)

    def test_negative_values(self):
        pts = np.array([[-3.0, -1.0], [2.0, 4.0]])
        min_v, max_v = _range(pts)
        assert min_v[0] == pytest.approx(-3.0)
        assert max_v[1] == pytest.approx(4.0)

    def test_identical_points(self):
        pts = np.array([[1.0, 1.0, 1.0]] * 5)
        min_v, max_v = _range(pts)
        np.testing.assert_array_equal(min_v, max_v)


# ---------------------------------------------------------------------------
# _subs
# ---------------------------------------------------------------------------


class TestSubs:
    def test_basic(self):
        result = _subs(0.0, 5.0)
        assert result == 5

    def test_floored(self):
        # floor(7.9 - 0.1) = floor(7.8) = 7
        result = _subs(0.1, 7.9)
        assert result == 7

    def test_equal_values(self):
        result = _subs(3.0, 3.0)
        assert result == 0

    def test_returns_int(self):
        result = _subs(0.0, 10.0)
        assert isinstance(result, (int, np.integer))

    def test_large_range(self):
        result = _subs(0.0, 1000.7)
        assert result == 1000


# ---------------------------------------------------------------------------
# index_cloud
# ---------------------------------------------------------------------------


class TestIndexCloud:
    def test_basic_3d(self):
        """Points uniformly spanning [0,1] mapped to voxels [0, K-1]."""
        cloud = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5]])
        start = np.array([0.0, 0.0, 0.0])
        end = np.array([1.0, 1.0, 1.0])
        K = np.array([10, 10, 10])
        indices = index_cloud(cloud, start, end, K)
        assert indices.shape == (3, 3)
        # First point maps to [0, 0, 0]
        np.testing.assert_array_equal(indices[0], [0, 0, 0])
        # Last point maps to [10, 10, 10] — truncated to int from 10.0
        np.testing.assert_array_equal(indices[1], [10, 10, 10])
        # Middle point maps to [5, 5, 5]
        np.testing.assert_array_equal(indices[2], [5, 5, 5])

    def test_output_dtype_is_int(self):
        cloud = np.random.default_rng(0).random((20, 3))
        start = np.zeros(3)
        end = np.ones(3)
        K = np.array([10, 10, 10])
        indices = index_cloud(cloud, start, end, K)
        assert np.issubdtype(indices.dtype, np.integer)

    def test_non_zero_origin(self):
        cloud = np.array([[2.0, 2.0, 2.0]])
        start = np.array([1.0, 1.0, 1.0])
        end = np.array([3.0, 3.0, 3.0])
        K = np.array([4, 4, 4])
        indices = index_cloud(cloud, start, end, K)
        # (2-1)/(3-1) * 4 = 0.5 * 4 = 2
        np.testing.assert_array_equal(indices[0], [2, 2, 2])

    def test_shape_preserved(self):
        N = 50
        cloud = np.random.default_rng(1).random((N, 3))
        indices = index_cloud(cloud, np.zeros(3), np.ones(3), np.array([8, 8, 8]))
        assert indices.shape == (N, 3)


# ---------------------------------------------------------------------------
# create_dict_cloud_to_voxels
# ---------------------------------------------------------------------------


class TestCreateDictCloudToVoxels:
    def test_basic_mapping(self):
        cloud = np.array([[0, 0], [1, 1], [2, 2]])
        voxel_indices = np.array([[0, 0], [1, 1], [2, 2]])
        result = create_dict_cloud_to_voxels(cloud, voxel_indices)
        assert result[0] == (0, 0)
        assert result[1] == (1, 1)
        assert result[2] == (2, 2)

    def test_length_matches_cloud(self):
        N = 15
        cloud = np.random.default_rng(0).random((N, 3))
        voxels = np.random.randint(0, 10, size=(N, 3))
        result = create_dict_cloud_to_voxels(cloud, voxels)
        assert len(result) == N

    def test_values_are_tuples(self):
        cloud = np.array([[1.0, 2.0, 3.0]])
        voxels = np.array([[4, 5, 6]])
        result = create_dict_cloud_to_voxels(cloud, voxels)
        assert isinstance(result[0], tuple)

    def test_empty_cloud_gives_empty_dict(self):
        result = create_dict_cloud_to_voxels(np.empty((0, 3)), np.empty((0, 3), dtype=int))
        assert result == {}


# ---------------------------------------------------------------------------
# repeating_indices
# ---------------------------------------------------------------------------


class TestRepeatingIndices:
    def test_no_repeats(self):
        d = {0: (0, 0, 0), 1: (1, 1, 1), 2: (2, 2, 2)}
        result = repeating_indices(d)
        assert result == {}

    def test_one_repeat(self):
        d = {0: (0, 0, 0), 1: (0, 0, 0), 2: (1, 1, 1)}
        result = repeating_indices(d)
        # Key (0,0,0) maps to both point 0 and point 1
        assert (0, 0, 0) in result
        assert 0 in result[(0, 0, 0)]
        assert 1 in result[(0, 0, 0)]

    def test_single_entry_no_repeat(self):
        d = {0: (5, 5, 5)}
        result = repeating_indices(d)
        assert result == {}

    def test_all_same_voxel(self):
        d = {i: (3, 3, 3) for i in range(5)}
        result = repeating_indices(d)
        assert (3, 3, 3) in result
        assert len(result[(3, 3, 3)]) == 5


# ---------------------------------------------------------------------------
# areadyadic_indices
# ---------------------------------------------------------------------------


class TestAreadyadicIndices:
    def test_output_shape(self):
        voxel_indices = np.array([[2, 3, 4], [5, 6, 7], [1, 2, 3]])
        result = areadyadic_indices(voxel_indices)
        # shape should be (max_x, max_y, max_z, 3, 3)
        assert result.shape == (5, 6, 7, 3, 3)

    def test_all_zeros(self):
        voxel_indices = np.array([[1, 2, 3], [4, 5, 6]])
        result = areadyadic_indices(voxel_indices)
        assert np.all(result == 0)

    def test_dtype_float(self):
        voxel_indices = np.array([[2, 3, 4]])
        result = areadyadic_indices(voxel_indices)
        assert result.dtype == float

    def test_single_voxel(self):
        voxel_indices = np.array([[3, 4, 5]])
        result = areadyadic_indices(voxel_indices)
        assert result.shape == (3, 4, 5, 3, 3)


# ---------------------------------------------------------------------------
# areadyadic_grid_safe
# ---------------------------------------------------------------------------


class TestAreadyadicGridSafe:
    def test_adds_values_to_grid(self):
        ad = np.zeros((5, 5, 5, 3, 3))
        product = np.array([np.eye(3)])
        voxels = np.array([[2, 2, 2]])
        result = areadyadic_grid_safe(ad, product, voxels)
        np.testing.assert_array_almost_equal(result[2, 2, 2], np.eye(3))

    def test_out_of_bounds_skipped(self):
        """Out-of-bounds indices should be silently skipped."""
        ad = np.zeros((3, 3, 3, 3, 3))
        product = np.array([np.eye(3)])
        voxels = np.array([[10, 10, 10]])  # out of bounds
        result = areadyadic_grid_safe(ad, product, voxels)
        assert np.all(result == 0)

    def test_accumulates_multiple_products(self):
        ad = np.zeros((5, 5, 5, 3, 3))
        product = np.array([np.eye(3), np.eye(3)])
        voxels = np.array([[1, 1, 1], [1, 1, 1]])
        result = areadyadic_grid_safe(ad, product, voxels)
        np.testing.assert_array_almost_equal(result[1, 1, 1], 2 * np.eye(3))

    def test_different_voxels_accumulate_independently(self):
        ad = np.zeros((5, 5, 5, 3, 3))
        m1 = np.eye(3)
        m2 = 2 * np.eye(3)
        product = np.array([m1, m2])
        voxels = np.array([[0, 0, 0], [1, 1, 1]])
        result = areadyadic_grid_safe(ad, product, voxels)
        np.testing.assert_array_almost_equal(result[0, 0, 0], m1)
        np.testing.assert_array_almost_equal(result[1, 1, 1], m2)

    def test_negative_indices_skipped(self):
        ad = np.zeros((5, 5, 5, 3, 3))
        product = np.array([np.eye(3)])
        voxels = np.array([[-1, -1, -1]])
        result = areadyadic_grid_safe(ad, product, voxels)
        assert np.all(result == 0)

    def test_returns_same_array_object(self):
        ad = np.zeros((4, 4, 4, 3, 3))
        product = np.array([np.eye(3)])
        voxels = np.array([[1, 1, 1]])
        result = areadyadic_grid_safe(ad, product, voxels)
        assert result is ad


# ---------------------------------------------------------------------------
# map_isosurface (integration)
# ---------------------------------------------------------------------------


class TestMapIsosurface:
    def test_basic_run(self):
        """Integration: map_isosurface should run and return an array."""
        rng = np.random.default_rng(42)
        cloud = rng.uniform(0, 5, size=(20, 3))
        areadyadic = np.array([np.eye(3) for _ in range(20)])
        dims = np.array([4, 4, 4])
        result = map_isosurface(cloud, areadyadic, dims)
        assert result is not None
        assert result.ndim == 5
        assert result.shape[-2:] == (3, 3)

    def test_output_shape_last_two_dims(self):
        cloud = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [2.0, 3.0, 4.0]])
        areadyadic = np.array([np.eye(3)] * 3)
        dims = np.array([3, 3, 3])
        result = map_isosurface(cloud, areadyadic, dims)
        assert result.shape[-1] == 3
        assert result.shape[-2] == 3

    def test_all_same_point_raises_value_error(self):
        # TODO: All identical points cause RANGE_END == RANGE_START (zero range),
        # producing an integer overflow in index_cloud that yields negative voxel
        # indices.  areadyadic_indices then rejects them with ValueError.
        # This test documents the known limitation so a future fix is visible;
        # index_cloud should guard against zero-range axes before dividing.
        cloud = np.array([[1.0, 1.0, 1.0]] * 5)
        areadyadic = np.array([np.eye(3)] * 5)
        dims = np.array([2, 2, 2])
        with pytest.raises((ValueError, FloatingPointError)):
            map_isosurface(cloud, areadyadic, dims)
