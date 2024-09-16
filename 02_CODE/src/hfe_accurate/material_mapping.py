import copy
import logging
import pickle
import sys
from pathlib import Path
from time import time

import hfe_utils.imutils as utils
import matplotlib

try:
    matplotlib.use("TkAgg")
except ImportError:
    pass
import matplotlib.pyplot as plt
import numpy as np
import scipy  # type: ignore
from hfe_abq.write_abaqus import AbaqusWriter
from scipy.spatial import KDTree  # type: ignore

from hfe_utils.io_utils import FileConfig

LOGGING_NAME = "HFE-ACCURATE"
logger = logging.getLogger(LOGGING_NAME)
logger.setLevel(logging.INFO)
logger.propagate = False


# flake8: noqa: E501


def calculate_bvtv(seg_array, mask_array, spacing, VOI_mm, cog):
    """
    POS, 06.02.2024
    Calculate the bone volume to total volume (BV/TV) ratio.

    This function calculates the BV/TV ratio, which is a measure of bone density.
    It does this by first calculating the region of interest (ROI) in the given 3D array (seg_array),
    then computing the bone partial volume (PHI), and finally computing the BV/TV ratio.

    Parameters:
    seg_array (np.array): The 3D array representing the segmented image.
    mask_array (np.array): The 3D array representing the mask image.
    spacing (np.array): The spacing between voxels in the 3D image.
    VOI_mm (float): The size of the Volume of Interest in millimeters.
    cog (np.array): The center of gravity of the FE element.

    Returns:
    tuple: A tuple containing the PHI and RHO values.
    """
    VOI = VOI_mm / spacing[0]
    x, y, z = cog / spacing
    semisizes_s = (float(VOI / 2),) * 3

    def __compute_roi_indices__():
        """
        Compute the start and end indices for a region of interest (ROI) in a 3D array.

        The ROI is centered at a point (x, y, z) and has a size of VOI (Volume of Interest) in each dimension.
        The function ensures that the indices do not go beyond the array boundaries.

        Parameters:
        x (int): The x-coordinate of the center of the ROI.
        y (int): The y-coordinate of the center of the ROI.
        z (int): The z-coordinate of the center of the ROI.
        VOI (int): The size of the Volume of Interest in each dimension.
        seg_array (np.array): The 3D array in which to calculate the ROI.

        Returns:
        tuple: A tuple containing the start and end indices for the x, y, and z dimensions.
        """
        x_start, x_end = int(np.rint(max(x - VOI / 2, 0))), int(
            np.rint(min(x + VOI / 2, seg_array.shape[0]))
        )
        y_start, y_end = int(np.rint(max(y - VOI / 2, 0))), int(
            np.rint(min(y + VOI / 2, seg_array.shape[1]))
        )
        z_start, z_end = int(np.rint(max(z - VOI / 2, 0))), int(
            np.rint(min(z + VOI / 2, seg_array.shape[2]))
        )
        return x_start, x_end, y_start, y_end, z_start, z_end

    def __computePHI__(ROI):
        """
        Compute the bone partial volume.

        Parameters:
        ROI (ndarray): The array containing the ROI of the image.

        Returns:
        PHI (float): The PHI value.
        """
        try:
            phi = float(np.count_nonzero(ROI == 1) / ROI.size)
        except ZeroDivisionError as e:
            print(e)
            phi = 0.0

        # check for meaningful output
        if np.isnan(phi):
            phi = 0.0
        if phi > 1:
            phi = 1.0
        return phi

    def __compute_bvtv__():
        """
        Compute the bone volume to total volume (BV/TV) ratio.

        This function computes the BV/TV ratio by first generating a sphere array within the ROI,
        then calculating the PHI and rho_s values. If the PHI value is greater than 0,
        the function calculates the rho_s value. If the PHI value is not greater than 0,
        the function sets both the PHI and rho_s values to 0.01.

        Returns:
        tuple: A tuple containing the PHI and rho_s values.
        """

        def __generate_sphere_array__(shape, position):
            """
            Generate a 3D array with a sphere of given radius and position.

            The function creates a 3D grid of the specified shape, and then calculates for each point in the grid
            if it's inside the sphere (represented as 1) or outside the sphere (represented as 0).

            Parameters:
            shape (tuple): The shape of the 3D array to be generated.
            radius (float): The radius of the sphere.
            position (tuple): The position of the center of the sphere.

            Returns:
            np.array: A 3D numpy array of the specified shape.
            """
            grid = np.ogrid[[slice(-x0, dim - x0) for x0, dim in zip(position, shape)]]
            arr = np.zeros(shape, dtype=float)
            for x_i, semisize in zip(grid, semisizes_s):
                arr += (x_i / semisize) ** 2
            return (arr <= 1.0).astype(int)

        x_start, x_end, y_start, y_end, z_start, z_end = __compute_roi_indices__()

        ROI_seg = seg_array[x_start:x_end, y_start:y_end, z_start:z_end]
        ROI_mask = mask_array[x_start:x_end, y_start:y_end, z_start:z_end]

        phi_s = __computePHI__(ROI_mask)

        if phi_s > 0.0:
            ROI_mask_sphere = __generate_sphere_array__(
                shape=np.shape(ROI_seg),
                position=tuple(
                    cog_i - start_i
                    for cog_i, start_i in zip([x, y, z], [x_start, y_start, z_start])
                ),
            )

            ROI_mask_sphere_mask = np.multiply(ROI_mask_sphere, ROI_mask)
            ROI_sphere_seg = np.multiply(ROI_mask_sphere_mask, ROI_seg)
            try:
                rho_s = np.sum(ROI_sphere_seg) / np.sum(ROI_mask_sphere_mask)
            except ZeroDivisionError:
                rho_s = 0.01

        else:
            # Ensure minimum bvtv, added by POS, 14.01.2024
            phi_s = 0.01
            rho_s = 0.01
        return max(0.01, min(1.0, phi_s)), max(0.01, min(1.0, rho_s))

    return __compute_bvtv__()


def __material_mapping__(
    cog,
    spacing,
    VOI_mm,
    mask_array,
    seg_array,
    SEG_correction,
    all_mask: bool,
    compartment_s: str,
):
    """
    Compartment agnostic material mapping function
    Simone Poncioni, MSB, 08.2023
    """
    phi = np.zeros(len(cog), dtype=np.float32)
    rho = np.zeros(len(cog), dtype=np.float32)
    rho_fe = np.zeros(len(cog), dtype=np.float32)

    seg_array = (seg_array > 0.1).astype(np.uint8)
    mask_array = (mask_array > 0.1).astype(np.uint8)

    timestart = time()
    for i, cog_s in enumerate(cog.values()):

        phi_s, rho_s = calculate_bvtv(seg_array, mask_array, spacing, VOI_mm, cog_s)

        if SEG_correction == True:
            TOL_SPACING = 0.01
            if abs(spacing[0] - 0.061) <= TOL_SPACING:
                # Correction curve from Varga et al. 2009 for XCTII
                rho_s = rho_s * 0.651 + 0.056462
            elif abs(spacing[0] - 0.082) <= TOL_SPACING:
                # Correction curve from Varga et al. 2009 for XCTI, added by MI
                rho_s = rho_s * 0.745745 - 0.0209902
            else:
                raise ValueError(
                    f"SEG_correction is True but 'spacing' is not 0.061 nor 0.082 (it is {spacing[0]}))"
                )
        else:
            pass

        if all_mask == True:
            if phi_s > 0.0 and rho_s < 0.01:
                rho_s = 0.01

        phi[i] = phi_s
        rho[i] = rho_s
        rho_fe[i] = 1  # ! adapt this

    timeend = time()
    elaps_time = timeend - timestart
    print(f"Elapsed Time: {elaps_time}")

    """
    plt.figure(figsize=(10, 10))
    plt.hist(phi.flatten(), bins=100)
    plt.savefig(f"phi_{compartment_s}_new.png")
    plt.close()

    plt.figure(figsize=(10, 10))
    plt.hist(rho.flatten(), bins=100)
    plt.savefig(f"rho_{compartment_s}_new.png")
    plt.close()
    """

    return phi, rho, rho_fe


def _bmc_compensation(
    BMD: np.ndarray,
    CORTMASK: np.ndarray,
    TRABMASK: np.ndarray,
    cort_elms: dict,
    trab_elms: dict,
    RHOc: np.ndarray,
    RHOt: np.ndarray,
    PHIc: np.ndarray,
    PHIt: np.ndarray,
    FEelSize,
    Spacing,
    BMC_conservation: bool,
):
    """
    # BMC compensation for all BVTV values in order to conserve bone mass during homogenization
    """
    ONE = float(1.0)
    THREE = int(3)
    THOUSAND = int(1000)
    THOUSAND_TWO_HUNDRED = int(1200)

    # fmt: off
    BMC_reco_c = np.sum(BMD[(CORTMASK) > 0]) * Spacing[0] ** THREE / THOUSAND
    BMC_reco_t = np.sum(BMD[(TRABMASK) > 0]) * Spacing[0] ** THREE / THOUSAND
    
    BMC_sim_c = np.sum(RHOc * PHIc * FEelSize[0] ** THREE) * THOUSAND_TWO_HUNDRED / THOUSAND
    BMC_sim_t = np.sum(RHOt * PHIt * FEelSize[0] ** THREE) * THOUSAND_TWO_HUNDRED / THOUSAND
    
    lambda_BMC_c = BMC_reco_c / BMC_sim_c
    lambda_BMC_t = BMC_reco_t / BMC_sim_t
    BMC_reco_tot = BMC_reco_c + BMC_reco_t

    # 2) Copy RHOc and RHOt dict to save uncompensated BVTV values
    RHOc_original = copy.deepcopy(RHOc)
    RHOt_original = copy.deepcopy(RHOt)

    # 3) Compensate RHOb BVTV values
    # TODO: check if it should be keys() or values()
    if BMC_conservation == True:
        # Cortical BVTV
        for elem in cort_elms.keys():
            if RHOc[elem] * lambda_BMC_c < ONE:
                RHOc[elem] = RHOc_original[elem] * lambda_BMC_c
            else:
                RHOc[elem] = ONE

        # Trabecular BVTV
        for elem in trab_elms.keys():
            if RHOt[elem] * lambda_BMC_t < ONE:
                RHOt[elem] = RHOt_original[elem] * lambda_BMC_t
            else:
                RHOt[elem] = ONE
    else:
        for elem in cort_elms.keys():
            RHOc[elem] = RHOc_original[elem]
        for elem in trab_elms.keys():
            RHOt[elem] = RHOt_original[elem]

    BMC_sim_comp = (
        (
            np.sum(RHOc * PHIc * FEelSize[0] ** THREE)
            + np.sum(RHOt * PHIt * FEelSize[0] ** THREE)
        )
        * THOUSAND_TWO_HUNDRED
        / THOUSAND
    )

    return RHOc, RHOt, BMC_sim_comp, BMC_reco_tot, lambda_BMC_c, lambda_BMC_t
    # fmt: on


def __get_masks__(
    cog,
    spacing,
    VOI_mm,
    seg_array,
    mask_array,
):
    x, y, z = cog / spacing
    VOI = VOI_mm / spacing[0]
    semisizes_s = (float(VOI / 2),) * 3

    def __generate_sphere_array__(shape, position):
        """
        Generate a 3D array with a sphere of given radius and position.

        The function creates a 3D grid of the specified shape, and then calculates for each point in the grid
        if it's inside the sphere (represented as 1) or outside the sphere (represented as 0).

        Parameters:
        shape (tuple): The shape of the 3D array to be generated.
        radius (float): The radius of the sphere.
        position (tuple): The position of the center of the sphere.

        Returns:
        np.array: A 3D numpy array of the specified shape.
        """
        grid = np.ogrid[[slice(-x0, dim - x0) for x0, dim in zip(position, shape)]]
        arr = np.zeros(shape, dtype=float)
        for x_i, semisize in zip(grid, semisizes_s):
            arr += (x_i / semisize) ** 2
        return (arr <= 1.0).astype(int)

    X = [max(x - VOI / 2, 0), min(x + VOI / 2, seg_array.shape[0])]
    Y = [max(y - VOI / 2, 0), min(y + VOI / 2, seg_array.shape[1])]
    Z = [max(z - VOI / 2, 0), min(z + VOI / 2, seg_array.shape[2])]

    ROI_seg = seg_array[
        int(np.rint(X[0])) : int(np.rint(X[1])),
        int(np.rint(Y[0])) : int(np.rint(Y[1])),
        int(np.rint(Z[0])) : int(np.rint(Z[1])),
    ]
    # ROI_seg contains 2 if using the scanco segmentation in the cortical compartment
    # Transforming them into 1s
    ROI_seg = np.where(ROI_seg == 2, 1, ROI_seg)

    ROI_mask = mask_array[
        int(np.rint(X[0])) : int(np.rint(X[1])),
        int(np.rint(Y[0])) : int(np.rint(Y[1])),
        int(np.rint(Z[0])) : int(np.rint(Z[1])),
    ]
    ROI_mask[ROI_mask > 0] = 1

    # calculate center of sphere in new image
    xc = x - X[0]
    yc = y - Y[0]
    zc = z - Z[0]

    ROI_mask_sphere = __generate_sphere_array__(np.shape(ROI_seg), [xc, yc, zc])
    return ROI_seg, ROI_mask, ROI_mask_sphere


def __computePHI__(ROI):
    """
    Compute the bone partial volume.

    Parameters:
    ROI (ndarray): The array containing the ROI of the image.

    Returns:
    PHI (float): The PHI value.
    """
    try:
        phi = float(np.count_nonzero(ROI)) / ROI.size
    except ZeroDivisionError:
        phi = 0.0

    # check for meaningful output
    if np.isnan(phi):
        phi = 0.0
    if phi > 1:
        phi = 1.0
    return phi


def __get_fe_dims__(img_shape: tuple, FEelSize: float, Spacing: float):
    """
    Helper function to retrieve same dimensions as FE mesh in Denis' mesh construction
    Simone Poncioni, MSB, 17.08.2023

    Args:
    - img_shape (tuple): shape of the original image
    - FEelSize (float): size of the finite elements in the FE mesh --> consistent with Denis's mesh construction
    - Spacing (float): voxel spacing of the original image

    Returns:
    - MESH_centroids_mm (numpy.ndarray): array of centroids of each element in the FE mesh, in millimeters
    - FEdimX (int): dimension of the FE mesh in the X direction
    - FEdimY (int): dimension of the FE mesh in the Y direction
    - FEdimZ (int): dimension of the FE mesh in the Z direction
    - CoarseFactor (float): coarsening factor used to construct the FE mesh
    """

    CoarseFactor = FEelSize / Spacing
    MESH = np.ones(
        ([int(dim) for dim in np.floor(np.array(img_shape) / CoarseFactor)]),
    )
    # Find dimensions of mesh
    FEdimX = MESH.shape[2]  # Dimension in X
    FEdimY = MESH.shape[1]  # Dimension in Y
    FEdimZ = MESH.shape[0]  # Dimension in Z

    # Calculate the centroids of each element in a sorted array (x, y, z)
    # Assuming that the mesh is centered in the middle of the image
    # Assuming that the voxel is isotropic
    x, y, z = np.meshgrid(
        np.arange(0.5, MESH.shape[0] + 0.5),
        np.arange(0.5, MESH.shape[1] + 0.5),
        np.arange(0.5, MESH.shape[2] + 0.5),
        indexing="ij",
    )
    x = x.flatten() * CoarseFactor
    y = y.flatten() * CoarseFactor
    z = z.flatten() * CoarseFactor
    MSL_centroids = np.column_stack((x, y, z))
    MSL_centroids_r = np.reshape(MSL_centroids, (-1, 3))
    MSL_centroids_mm = MSL_centroids_r * Spacing

    return MSL_centroids_mm, FEdimX, FEdimY, FEdimZ, CoarseFactor


def getClosestPhysPoint(centroids_mesh, phys_points):
    """
    Finds the closest physical point to each centroid in the cortical mesh.
    You can then mask the physical points with this array to get the closest physical point for each centroid

    Args:
    - centroids_mesh (numpy.ndarray): array of centroids of each element in the cortical mesh, in millimeters (m, 3)
    - phys_points (numpy.ndarray): array of physical points, in millimeters (n, 3)

    Returns:
    - distances (numpy.ndarray): array of distances between each centroid and its closest physical point
    - indices (numpy.ndarray): array of indices of the closest physical point for each centroid

    """
    kdTree = KDTree(phys_points, leafsize=100, copy_data=True, balanced_tree=True)
    distances, indices = kdTree.query(centroids_mesh, k=1)
    return distances, indices


def correspondence_dict(centroids_mesh, closest_phys_points):
    """
    Creates a correspondence dictionary between physical points and centroids in the mesh.

    Args:
    - centroids_mesh (numpy.ndarray): array of centroids of each element in the mesh, in millimeters
    - closest_phys_points (numpy.ndarray): array of indices of the closest physical point for each centroid in the MSL kernel

    Returns:
    - correspondence_dict (dict): dictionary mapping each physical point index to its corresponding centroid in the mesh
    """
    correspondence_dict = {}
    for i, idx in enumerate(centroids_mesh):
        correspondence_dict[i] = closest_phys_points[i]
    return correspondence_dict


def vectoronplane(
    evect_max,
    evect_mid,
    evect_min,
    direction,
):
    """

    Parameters
    ----------
    evect_max
    evect_mid
    evect_min
    direction

    Returns
    -------
    3 numpy arrays in order evect min, evect mid, evect max

    evect_max_projected is computed by projection of direction (usually [0,0,1]) into the plane evect_max, evect_mid.
    evect_min_projected is orthogonal to max and mid, so computed from their cross product
    evect_mid_projected is orthogonal to max and min, so computed from their cross produc
    Old implementation
    normal = numpy.cross(evect_max, evect_mid)
    normalized = normal / numpy.linalg.norm(normal)
    evect_max_projected_normalized = (
            direction - numpy.dot(direction, normalized) * normalized
    )
    evect_max_projected = evect_max_projected_normalized * numpy.linalg.norm(evect_max)

    evect_min_projected = normalized
    evect_mid_projected = numpy.cross(evect_max_projected, normalized)
    evect_max_proj = direction - numpy.dot(direction, normal)
    Projection acc. https://www.maplesoft.com/support/help/maple/view.aspx?path=MathApps%2FProjectionOfVectorOntoPlane
    """
    try:
        normal = np.cross(evect_max, evect_mid)

        evect_max_proj = (
            direction
            - (np.dot(direction, normal) / np.linalg.norm(normal) ** 2) * normal
        )

        evect_mid = np.cross(evect_max_proj, normal)

        evect_min = normal

        evect_max_projected = evect_max_proj / np.linalg.norm(evect_max_proj)
        evect_mid_projected = evect_mid / np.linalg.norm(evect_mid)
        evect_min_projected = evect_min / np.linalg.norm(evect_min)

        scal_max_mid = np.dot(evect_max_projected, evect_mid_projected)
        scal_max_min = np.dot(evect_max_projected, evect_min_projected)
        scal_mid_min = np.dot(evect_mid_projected, evect_min_projected)

        if scal_max_mid + scal_max_min + scal_mid_min > 0.001:
            logger.warning("projected vectors are not orthogonal!")

    except:
        evect_min_projected = evect_min
        evect_mid_projected = evect_mid
        evect_max_projected = evect_max
        logger.error(
            "Could not perform the vector projection in utils.vectoronplane()!"
        )

    return evect_min_projected, evect_mid_projected, evect_max_projected


def compute_isoFAB():
    """
    Returns isotropic fabric
    return [eval1, eval2, eval3], [evecxx, evecxy, evecxz], [evecyx, evecyy, evecyz], [eveczx, eveczy, eveczz]]
    """
    return [1.0, 1.0, 1.0], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def cai_evalues():
    """
    Returns evalues of Cai et al. (2019) adapted for transverse isotropy.
    From human cortical bone of the proximal femur.
    https://doi.org/10.1016/j.actbio.2019.03.043
    Returns
    -------
    Evalues E1, E2, E3
    """
    E3 = 1.124
    E2 = 0.938
    E1 = 0.938
    return [E1, E2, E3]


def __compute_eval_evect_projection__(elms, _evect):
    # Fabric projection for cortical bone
    evals = np.zeros((len(elms), 3))
    evects = np.zeros((len(elms), 3, 3))
    # TODO: make sure that the order of the evects is correct
    for i, idx in enumerate(elms.values()):
        evects[i, 0] = _evect[idx][:, 0]  # min
        evects[i, 1] = _evect[idx][:, 1]  # mid
        evects[i, 2] = _evect[idx][:, 2]  # max

        if np.isnan(np.sum(np.array(evects[i]))):
            logger.warning("warning: nan in evect")
            # return 3x3 identity matrix
            _, evects[i] = compute_isoFAB()
        evals[i] = cai_evalues()  # Cai et al, Acta Biomater. 2019

    return evals, evects


def __compute_eval_evect__(MSL_kernel_list, elms, BVseg, projection=True):
    """
    Computes Eigenvalues and Eigenvectors for a given element using MSL_kernel_list, which is a return value of
    preprocessing.compute_local_MSL (stored in bone: dict)

    Can be used for cortical or trabecular phase. Note that for cortical phase projection = True!

    Parameters
    ----------
    MSL_kernel_list:    List with areaweighted dyadic products of triangulation, after kernel homogenization
    BVseg               bone volume from segmentation for specific bone phase
    projection          Defines if projection of global Z on plane of MSL (used for cortical phase to have main
                        orientation along cortical shell

    Returns
    -------
    eval                Eigenvalues
    evect               Eigenvectors
    """

    evals = np.zeros((len(elms), 3))
    evects = np.zeros((len(elms), 3, 3))
    ee = 0
    eee = 0
    for i, idx in enumerate(elms.values()):
        msl_kernel_idx = MSL_kernel_list[idx]
        bv_i = BVseg[i][0]  # only return value
        if np.linalg.cond(msl_kernel_idx) > 1 / sys.float_info.epsilon:
            msl_kernel_idx = np.eye(3)

        # MSL method according to Hosseini Bone 2017
        H = 2.0 * bv_i * scipy.linalg.inv(msl_kernel_idx)
        try:
            MSL = 3.0 * H / np.trace(H)
            evalue, evect = scipy.linalg.eig(MSL)

        except Exception as e:
            print(
                f"Exception n. {ee}: {e}\nMSL_kernel_list[i]:\n{MSL_kernel_list[idx]}\nBVseg[i]:\n{BVseg[i]}\n"
            )
            ee += 1
            # returns evalue [1, 1, 1], evect[X, Y, Z] -> min, mid, max
            evalue, evect = compute_isoFAB()

        if isinstance(evalue, list):
            evalue = np.array(evalue)
        if isinstance(evect, list):
            evect = np.array(evect)

        _argsort = evalue.argsort()  # order eigenvalues 0=min, 1=mid, 2=max
        evalue = evalue[_argsort]
        evect = evect[:, _argsort]
        evalue = np.array([e.real for e in evalue])
        evect = np.array(evect)

        if projection == True:
            # Fabric projection for cortical bone
            # vectoronplane requires inputs evect max, mid, min, so evect[2], evect[1], evect[0]
            evect[:, 0], evect[:, 1], evect[:, 2] = vectoronplane(
                evect[:, 2], evect[:, 1], evect[:, 0], np.array([0.0, 0.0, 1.0])
            )
            if np.isnan(np.sum(np.array(evect))):
                # return 3x3 identity matrix
                _, evect = compute_isoFAB()
            evalue = cai_evalues()  # Cai et al, Acta Biomater. 2019

        _lim = float(2.5)
        if np.any(np.array(evalue) > _lim):
            # raise ValueError(f"evalue > {_lim} in element {i}")
            print(f"Exception n. {eee}:\nevalue > {_lim} in element {i}")
            eee += 1
            evalue, evect = compute_isoFAB()
            evalue = np.array([e.real for e in evalue])
            evect = np.array(evect)

        evals[i] = evalue
        evects[i] = evect
    logger.warning(f"MSL exception encountered {ee} times")
    logger.warning(f"Eigenvector exception encountered {eee} times")
    return evals, evects


def material_mapping_spline(
    bone: dict,
    cfg,
    filenames: FileConfig,
):
    """
    Material Mapping, including PSL ghost padding layers as copy of most distal and proximal layers
    For accurate PSL pipeline
    Additionaly, the optional BMC conversion can be specified in config.yaml.
    This function will conserve BMC from image to hFE model to ensure, no mass conservation.

    Included new MSL fabric evaluation with separation between cortex and trabecular bone
    Version adapted for spline meshing algorithm (POS, 08.2023)

    Parameters:
    # ? add parameters

    Returns:
    # ? add returns
    """
    # size of computePHI volume of interest, same as Schenk et al. 2022
    VOI_cort_mm = cfg.homogenization.ROI_BVTV_size_cort
    VOI_trab_mm = cfg.homogenization.ROI_BVTV_size_trab

    # get images
    BVTVscaled = bone["BVTVscaled"]
    BMD_array = bone["BMDscaled"]
    CORTMASK_array = bone["CORTMASK_array"]
    TRABMASK_array = bone["TRABMASK_array"]
    SEG_array = bone["SEG_array"]

    # get bone values
    FEelSize = bone["FEelSize"]
    spacing = bone["Spacing"]
    air_elements = cfg.mesher.air_elements
    ROI_BVTV_size_trab = cfg.homogenization.ROI_BVTV_size_cort
    ROI_BVTV_size_cort = cfg.homogenization.ROI_BVTV_size_trab
    fabric_type = cfg.homogenization.fabric_type

    if cfg.homogenization.fabric_type == "local":
        # Computed by compute_local_MSL and assign_MSL_triangulation
        MSL_kernel_list_cort = bone["MSL_kernel_list_cort"]
        MSL_kernel_list_trab = bone["MSL_kernel_list_trab"]

    # * Material mapping
    RHOc = {}
    # RHOc_corrected = {}  # RHO corrected by PBV (RHO * PHI)
    RHOt = {}
    # RHOt_corrected = {}  # RHO corrected by PBV (RHO * PHI)
    RHOc_FE = {}  # RHO of only FE element (ROI = FEelement)
    RHOt_FE = {}  # RHO of only FE element (ROI = FEelement)
    PHIc = {}
    PHIt = {}
    mm = {}
    m = {}
    # BVTVcortseg = {}
    # BVTVtrabseg = {}
    BVTVcortseg_elem = {}
    BVTVtrabseg_elem = {}
    cogs = {}
    DOA = {}

    cog_real_cort = bone["elms_centroids_cort"]
    cog_real_trab = bone["elms_centroids_trab"]

    # # * Cortical compartment
    #! As we are basing it on BMD and not SEG, SEG_correction=False in cortical compartment!
    #! (POS, 21.03.2024)
    phi_cort, rho_cort, rho_fe_cort = __material_mapping__(
        cog_real_cort,
        spacing,
        VOI_cort_mm,
        CORTMASK_array,
        BVTVscaled,
        SEG_correction=False,
        all_mask=cfg.old_cfg.all_mask,
        compartment_s="cort",
    )

    # * Trabecular compartment
    phi_trab, rho_trab, rho_fe_trab = __material_mapping__(
        cog_real_trab,
        spacing,
        VOI_trab_mm,
        TRABMASK_array,
        SEG_array,
        SEG_correction=cfg.image_processing.SEG_correction,
        all_mask=cfg.old_cfg.all_mask,
        compartment_s="trab",
    )

    # ? Not very elegant, but assures back compatibility with Denis's code
    PHIc = phi_cort
    PHIt = phi_trab
    RHOc = rho_cort
    RHOt = rho_trab
    RHOc_FE = rho_fe_cort
    RHOt_FE = rho_fe_trab
    RHOc_corrected = RHOc * PHIc
    RHOt_corrected = RHOt * PHIt  # ? in original function, it's RHOc * PHIt

    # Compute elemental BVTV from segmentation
    # Method computePHI can be used on segmentation instead of mask
    BVTVcortseg_elem = np.zeros(len(cog_real_cort))
    for i, cog_s in enumerate(cog_real_cort.values()):
        _, ROI_mask_s, _ = __get_masks__(
            cog_s, spacing, VOI_cort_mm, SEG_array, CORTMASK_array
        )
        BVTVcortseg_elem_s = __computePHI__(ROI_mask_s)
        BVTVcortseg_elem[i] = BVTVcortseg_elem_s

    BVTVtrabseg_elem = np.zeros(len(cog_real_trab))
    for i, cog_s in enumerate(cog_real_trab.values()):
        _, ROI_mask_s, _ = __get_masks__(
            cog_s, spacing, VOI_trab_mm, SEG_array, TRABMASK_array
        )
        BVTVtrabseg_elem_s = __computePHI__(ROI_mask_s)
        BVTVtrabseg_elem[i] = BVTVtrabseg_elem_s

    BVTVcortseg = np.divide(
        BVTVcortseg_elem,
        PHIc,
        out=np.zeros(BVTVcortseg_elem.shape, dtype=float),
        where=PHIc != 0,
    )

    BVTVtrabseg = np.divide(
        BVTVtrabseg_elem,
        PHIt,
        out=np.zeros(BVTVtrabseg_elem.shape, dtype=float),
        where=PHIt != 0,
    )

    BVcortseg = np.array(
        [
            BVTVcortseg_elem[i] * bone["elms_vol_cort"][i]
            for i, _ in enumerate(BVTVcortseg_elem)
        ]
    )
    BVtrabseg = np.array(
        [
            BVTVtrabseg_elem[i] * bone["elms_vol_trab"][i]
            for i, _ in enumerate(BVTVtrabseg_elem)
        ]
    )

    # Evaluate Fabric using MSL
    img_shape = SEG_array.shape
    MSL_centroids_mm, FEdimX, FEdimY, FEdimZ, CoarseFactor = __get_fe_dims__(
        img_shape, FEelSize[0], spacing[0]
    )

    # getClosestPhysPoint needs a np.ndarray, not dict_values
    cog_real_cort_np = np.array(list(cog_real_cort.values()))
    cog_real_trab_np = np.array(list(cog_real_trab.values()))

    if cfg.homogenization.orthotropic_cortex is True:
        closest_distances_cort, closest_phys_points_cort = getClosestPhysPoint(
            cog_real_cort_np, bone["evect_origin"]
        )

    else:
        closest_distances_cort, closest_phys_points_cort = getClosestPhysPoint(
            cog_real_cort_np, MSL_centroids_mm
        )

    closest_distances_trab, closest_phys_points_trab = getClosestPhysPoint(
        cog_real_trab_np, MSL_centroids_mm
    )

    correspondences_cort = correspondence_dict(cog_real_cort, closest_phys_points_cort)
    correspondences_trab = correspondence_dict(cog_real_trab, closest_phys_points_trab)

    elms_cort = correspondences_cort
    elms_trab = correspondences_trab

    if fabric_type == "local":
        if cfg.homogenization.orthotropic_cortex is True:
            m_cort, mm_cort = __compute_eval_evect_projection__(
                elms_cort,
                bone["cort_projection_evect"],
            )
        else:
            m_cort, mm_cort = __compute_eval_evect__(
                MSL_kernel_list_cort, elms_cort, BVcortseg, projection=True
            )

        m_trab, mm_trab = __compute_eval_evect__(
            MSL_kernel_list_trab, elms_trab, BVtrabseg, projection=False
        )
        # * checking that no mixed elements were forgotten
        assert len(m_cort) + len(m_trab) == len(cog_real_cort) + len(cog_real_trab)

        if air_elements == True:
            raise NotImplementedError("air elements not implemented yet")
            # ! added by Michael to produce a fullblock element
            # ! needed for longitudinal studies to have always the same amount of elements

        bone["nel_CORT"] = len(m_cort)
        bone["nel_TRAB"] = len(m_trab)

        logger.info(
            f"The following number of elements were mapped for each phase\n"
            f"  - Cortical:\t{len(m_cort)}\n"
            f"  - Trabecular:\t{len(m_trab)}\n"
        )

    (
        RHOc_comp,
        RHOt_comp,
        BMC_sim_comp,
        BMC_reco_tot,
        lambda_BMC_c,
        lambda_BMC_t,
    ) = _bmc_compensation(
        BMD_array,
        CORTMASK_array,
        TRABMASK_array,
        elms_cort,
        elms_trab,
        RHOc,
        RHOt,
        PHIc,
        PHIt,
        FEelSize,
        spacing,
        BMC_conservation=cfg.image_processing.BMC_conservation,
    )

    # Create abaqus input file
    # ------------------------------------------------------------------
    model_name = bone["sample"]
    centroids_cort = bone["elms_centroids_cort"]
    centroids_trab = bone["elms_centroids_trab"]
    nodes = bone["nodes"]
    elms = bone["elms"]
    botnodes = bone["bnds_bot"]
    topnodes = bone["bnds_top"]
    RP_coords_s = bone["reference_point_coord"]
    inp_filename = filenames["INPname"]
    save_dir = Path(inp_filename).parent

    abq = AbaqusWriter(
        cfg,
        save_dir,
        model_name,
        nodes,
        elms,
        centroids_cort,
        centroids_trab,
        m_cort,
        m_trab,
        RHOc,
        RHOt,
        PHIc,
        PHIt,
        mm_cort,
        mm_trab,
        botnodes,
        topnodes,
        RP_tag=10000000,
        RP_coords=RP_coords_s,
        STEP_INC=1000,
        NLGEOM=cfg.abaqus.nlgeom,
        PARAM_FLAG=2,
        DENSIFICATOR_FLAG=0,
        VISCOSITY_FLAG=0,
        POSTYIELD_FLAG=1,
    )
    _ver = cfg.version.current_version
    umat_name_s = Path(cfg.abaqus.umat).name
    abq_dictionary = abq.abq_dictionary(umat_name=umat_name_s)
    inp_path = abq.abaqus_writer(_ver)

    logger.info("Writing vtk maps of fabric for visualization:")
    utils.fab2vtk_fromdict(filenames.vtk_name, abq_dictionary)
    logger.info("Writing vtk maps of fabric for visualization: Done")
    # extend m with cort and trab
    m = np.append(m_cort, m_trab)
    mm = np.append(mm_cort, mm_trab, axis=0)
    cogs_arr = np.append(cog_real_cort, cog_real_trab)
    # convert cogs to dict
    cogs = {i: cogs_arr[i] for i in range(len(cogs_arr))}

    # store variables to bone dict
    bone["RHOc_array"] = RHOc
    bone["RHOt_array"] = RHOt
    bone["RHOc_orig_array"] = RHOc
    bone["RHOt_orig_array"] = RHOt
    bone["PHIc_array"] = PHIc
    bone["PHIt_array"] = PHIt
    bone["RHOc_FE_array"] = RHOc_FE
    bone["RHOt_FE_array"] = RHOt_FE
    bone["elems"] = elms
    bone["elems_bone"] = elms
    bone["nodes"] = nodes
    # bone["elsets"] = elsets
    bone["marray"] = m
    bone["mmarray1"] = mm[:, 0]
    bone["mmarray2"] = mm[:, 1]
    bone["mmarray3"] = mm[:, 2]
    bone["cogs"] = cogs
    bone["CoarseFactor"] = bone["FEelSize"][0] / bone["Spacing"][0]
    bone["m_dict"] = m
    bone["mm_dict"] = mm
    bone["cogs_dict"] = cogs
    bone["lambda_BMC_c"] = lambda_BMC_c
    bone["lambda_BMC_t"] = lambda_BMC_t
    return bone, abq_dictionary, inp_path


def get_bone_dict(basepath, meshpath, _mesh):
    bone = {}
    meshname = meshpath.name
    with open(basepath / f"{_mesh}_BVTVscaled.pkl", "rb") as a:
        BVTVscaled = pickle.load(a)
        bone["BVTVscaled"] = BVTVscaled

    with open(basepath / f"{_mesh}_BMDscaled.pkl", "rb") as b:
        BMDscaled = pickle.load(b)
        bone["BMDscaled"] = BMDscaled

    with open(basepath / f"{_mesh}_CORTMASK_array.pkl", "rb") as c:
        CORTMASK_array = pickle.load(c)
        bone["CORTMASK_array"] = CORTMASK_array

    with open(basepath / f"{_mesh}_TRABMASK_array.pkl", "rb") as d:
        TRABMASK_array = pickle.load(d)
        bone["TRABMASK_array"] = TRABMASK_array

    with open(basepath / f"{_mesh}_SEG_array.pkl", "rb") as e:
        SEG_array = pickle.load(e)
        bone["SEG_array"] = SEG_array

    with open(basepath / f"{_mesh}_FEelSize.pkl", "rb") as f:
        FEelSize = pickle.load(f)
        bone["FEelSize"] = FEelSize

    with open(basepath / f"{_mesh}_Spacing.pkl", "rb") as g:
        Spacing = pickle.load(g)
        bone["Spacing"] = Spacing

    with open(meshpath / f"{meshname}_spline_centroids_cort_dict.pickle", "rb") as h:
        elems = pickle.load(h)
        bone["elms_centroids_cort"] = elems
    with open(meshpath / f"{meshname}_spline_centroids_trab_dict.pickle", "rb") as h:
        elems = pickle.load(h)
        bone["elms_centroids_trab"] = elems

    with open(meshpath / f"{meshname}_spline_topnodes.pickle", "rb") as h:
        topnodes = pickle.load(h)
        bone["bnds_top"] = topnodes
    with open(meshpath / f"{meshname}_spline_botnodes.pickle", "rb") as h:
        botnodes = pickle.load(h)
        bone["bnds_bot"] = botnodes

    with open(meshpath / f"{meshname}_elms.pickle", "rb") as h:
        elems = pickle.load(h)
        bone["elms"] = elems

    with open(meshpath / f"{meshname}_nodes.pickle", "rb") as i:
        nodes = pickle.load(i)
        bone["nodes"] = nodes

    # with open(meshpath / f"{meshname}_elsets.pkl", "rb") as j:
    #     elsets = pickle.load(j)
    #     bone["elsets"] = elsets

    with open(basepath / f"{_mesh}_MSL_kernel_list_cort.pkl", "rb") as k:
        MSL_kernel_list_cort = pickle.load(k)
        bone["MSL_kernel_list_cort"] = MSL_kernel_list_cort

    with open(basepath / f"{_mesh}_MSL_kernel_list_trab.pkl", "rb") as l:
        MSL_kernel_list_trab = pickle.load(l)
        bone["MSL_kernel_list_trab"] = MSL_kernel_list_trab

    with open(basepath / f"{_mesh}_MESH.pkl", "rb") as m:
        MESH = pickle.load(m)
        bone["MESH"] = MESH

    bone["reference_point_coord"] = [16.5649162, 11.66050003, 35.0]

    elm_vol_cort_path = f"{meshpath}/{meshname}_{_mesh}_elm_vol_cort.npy"
    elm_vol_trab_path = f"{meshpath}/{meshname}_{_mesh}_elm_vol_trab.npy"
    bone["elms_vol_cort"] = np.load(elm_vol_cort_path)
    bone["elms_vol_trab"] = np.load(elm_vol_trab_path)
    bone["sample"] = meshpath.name

    return bone
