#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Simone Poncioni, MSB Group, ARTORG Center for Biomedical Engineering Research, University of Bern
# Date: 25.07.2023

import logging
import multiprocessing
import os
from pathlib import Path

import numpy as np
from hfe_utils.io_utils import timeit
from numba import config, njit, prange, set_num_threads, threading_layer  # type: ignore

# flake8: noqa: E501
# set the threading layer before any parallel target compilation
config.THREADING_LAYER = "threadsafe"

# set the number of threads for parallel computation with numba
max_threads = multiprocessing.cpu_count()
try:
    if max_threads >= 16:
        set_num_threads(12)
    else:
        set_num_threads(max_threads - 2)
except ValueError:
    max_threads = 1


LOGGING_NAME = "HFE-ACCURATE"
logger = logging.getLogger(LOGGING_NAME)
logger.propagate = False


@timeit
def _range(pts):
    """
    Calculate the minimum and maximum values along each axis of a point cloud.

    Args:
        pts (numpy.ndarray): A numpy array of shape (N, D) representing a point cloud, where N is the number of points and D is the number of dimensions.

    Returns:
        tuple: A tuple of two numpy arrays representing the minimum and maximum values along each axis of the point cloud. Each array has shape (D,).
    """
    min_values = np.min(pts, axis=0)
    max_values = np.max(pts, axis=0)
    return min_values, max_values


@timeit
def _subs(min_value, max_value):
    """
    Computes the number of subdivisions between the given minimum and maximum values.

    Args:
        min_value (float): The minimum value.
        max_value (float): The maximum value.

    Returns:
        int: The number of subdivisions between the minimum and maximum values.
    """
    return np.floor(max_value - min_value).astype(int)


@timeit
def index_cloud(
    CLOUD: np.ndarray,
    RANGE_START: np.ndarray,
    RANGE_END: np.ndarray,
    VOXEL_K: np.ndarray,
):
    """
    Compute the voxel indices for a point cloud.

    Args:
        CLOUD (numpy.ndarray): The point cloud to index.
        RANGE_START (numpy.ndarray): The starting range of the voxel space.
        RANGE_END (numpy.ndarray): The ending range of the voxel space.
        VOXEL_K (numpy.ndarray): The number of voxels in each dimension of the voxel space.

    Returns:
        numpy.ndarray: The voxel indices for each point in the point cloud.

    Credits:
        https://stackoverflow.com/questions/61150380/efficiently-compute-voxel-indices-for-a-point-cloud

    """
    zero_based_points = CLOUD - RANGE_START
    fractional_points = zero_based_points / (RANGE_END - RANGE_START)
    voxelspace_points = fractional_points * VOXEL_K
    voxel_indices = voxelspace_points.astype(int)
    return voxel_indices


@timeit
def create_dict_cloud_to_voxels(CLOUD: np.ndarray, voxel_indices: np.ndarray):
    """
    Creates a dictionary that maps each point in a point cloud to its corresponding voxel index.

    Args:
    - CLOUD: a numpy array representing a point cloud
    - voxel_indices: a numpy array representing the voxel indices of each point in the point cloud
        ---> Assuming that the ordering of the points in the cloud is sorted, positive, and unique

    Returns:
    - cloud_to_voxels_dict: a dictionary that maps each point in the point cloud to its corresponding voxel index
    """
    cloud_to_voxels_dict = {}
    # assuming that the ordering of the points in the cloud is sorted, positive, and unique
    for i in range(len(CLOUD)):
        cloud_to_voxels_dict[i] = tuple(voxel_indices[i])
    return cloud_to_voxels_dict


@timeit
def repeating_indices(cloud_to_voxels_dict: dict):
    """
    Takes a dictionary of point cloud indices to voxel indices and returns a dictionary of voxel indices
    that are repeated in the input dictionary, along with the point cloud indices that map to each repeated voxel index.

    Args:
        cloud_to_voxels_dict (dict): A dictionary mapping point cloud indices to voxel indices.

    Returns:
        dict: A dictionary of voxel indices that are repeated in the input dictionary, along with the point cloud
        indices that map to each repeated voxel index.
    """
    repeating_indices = {
        tuple(v): [
            k
            for k in cloud_to_voxels_dict
            if tuple(cloud_to_voxels_dict[k]) == tuple(v)
        ]
        for v in set(cloud_to_voxels_dict.values())
        if list(cloud_to_voxels_dict.values()).count(v) > 1
    }
    logger.info(f"Repeating indices:\n{repeating_indices}")
    return repeating_indices


@timeit
def areadyadic_indices(voxel_indices: np.ndarray):
    """
    Takes in a numpy array of voxel indices and returns a numpy array of zeros with shape (x, y, z, 3, 3),
    where x, y, and z are the maximum values of the voxel indices in each dimension plus one. This array is used to store
    the areadyadic products 'Aint' for each voxel in the input array.

    Args:
    - voxel_indices: a numpy array of shape (n, 3) representing the voxel indices

    Returns:
    - areadyadic_indices: a numpy array of shape (x, y, z, 3, 3) representing the areadyadic indices for each voxel
    """
    voxel_indices_r = voxel_indices.reshape(-1, 3)
    x = np.max(voxel_indices_r[:, 0])
    y = np.max(voxel_indices_r[:, 1])
    z = np.max(voxel_indices_r[:, 2])
    areadyadic_indices = np.zeros((x, y, z, 3, 3))
    return areadyadic_indices


@timeit
@njit(parallel=True)
def areadyadic_grid(
    ad_indices: np.ndarray, areadyadic_product: np.ndarray, voxel_indices: np.ndarray
):
    """
    Adds the values in `areadyadic_trab_subset` to the corresponding indices in `ad_indices`, based on the voxel indices
    in `voxel_indices`.

    Args:
        ad_indices (numpy.ndarray): A 3D numpy array representing the areadyadic grid.
        areadyadic_product (numpy.ndarray): A 1D numpy array representing the 3x3 matrix Aint.
        voxel_indices (numpy.ndarray): A numpy array representing the voxel indices.

    Returns:
        numpy.ndarray: A 3D numpy array representing the areadyadic grid with the values in `areadyadic_product`
        added to the corresponding grid indices.
    """
    # for i in range(len(areadyadic_product)):
    for i in prange(len(areadyadic_product)):
        ad_indices[
            voxel_indices[i][0],
            voxel_indices[i][1],
            voxel_indices[i][2],
        ] += areadyadic_product[i]
    return ad_indices


@timeit
def standalone_testing_exec():
    """
    Helper function to test the voxel indexing algorithm on a subset of the bone point cloud.
    Not to be used in production.
    """
    basedir = Path("/home/simoneponcioni/Documents/01_PHD/03_Methods/HFE-ACCURATE/")
    _dir = Path(basedir / "99_TEMP/octree_prototyping/")
    os.chdir(_dir)

    # load areadyadic and cog_points
    cog_points_trab = np.load(r"cog_points_trab.npy", allow_pickle=True)
    areadyadic_trab = np.load(r"areadyadic_trab.npy", allow_pickle=True)
    cog_points_trab_reshaped = cog_points_trab.reshape(-1, 3)

    _slicing_coeff = 1
    trab_indices = np.arange(len(cog_points_trab_reshaped))[::_slicing_coeff]
    cog_points_trab_subset = cog_points_trab_reshaped[trab_indices]
    areadyadic_trab_subset = areadyadic_trab[trab_indices]

    trab_x, trab_y, trab_z = (
        cog_points_trab_subset[:, 0],
        cog_points_trab_subset[:, 1],
        cog_points_trab_subset[:, 2],
    )
    CLOUD = np.column_stack((trab_x, trab_y, trab_z))
    return CLOUD, areadyadic_trab_subset


@timeit
def map_isosurface(
    CLOUD: np.ndarray, areadyadic_compartment: np.ndarray, DIMS: np.ndarray
):
    """
    Maps the isosurface of a point cloud to a grid.

    Args:
        CLOUD (np.ndarray): The point cloud to map.
        areadyadic_compartment (np.ndarray): The areadyadic compartment data.
        DIMS (np.ndarray): The dimensions of the voxel grid.

    Returns:
        np.ndarray: The areadyadic grid.
    """
    RANGE_START, RANGE_END = _range(CLOUD)

    voxel_indices = index_cloud(CLOUD, RANGE_START, RANGE_END, DIMS)

    ad_idx = areadyadic_indices(voxel_indices)
    ad_grid = areadyadic_grid(ad_idx, areadyadic_compartment, voxel_indices)
    return ad_grid


if __name__ == "__main__":
    CLOUD, areadyadic_compartment = standalone_testing_exec()
    dims_s = np.array([36, 31, 24])
    ad_grid = map_isosurface(CLOUD, areadyadic_compartment, DIMS=dims_s)
    print(ad_grid)
    print(ad_grid.shape)
    print("---------")
