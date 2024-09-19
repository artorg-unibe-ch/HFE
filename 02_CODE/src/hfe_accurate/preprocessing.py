import gc
import logging

import numpy as np
import vtk  # type: ignore
from hfe_accurate.project_normals_cortex import clustered_point_normals
from hfe_accurate.struct_voxel_indices import map_isosurface  # type: ignore
from hfe_accurate.surface_nets import surface_nets
from hfe_utils.imutils import numpy2vtk
from hfe_utils.io_utils import timeit
from numba import njit  # type: ignore
from scipy.ndimage.filters import convolve  # type: ignore
from vtk.numpy_interface import dataset_adapter as dsa  # type: ignore
from vtk.util.numpy_support import vtk_to_numpy  # type: ignore

LOGGING_NAME = "HFE-ACCURATE"
logger = logging.getLogger(LOGGING_NAME)
logger.propagate = False


def calculate_bvtv(
    scaling,
    slope,
    intercept,
    BMD_array,
    CORTMASK_array,
    TRABMASK_array,
    cfg,
    IMTYPE: str,
):
    """
    Calculates BVTV and mask images.

    Args:
        scaling (float): Scaling factor for the image.
        slope (float): Slope used in the image.
        intercept (float): Intercept used in the image.
        BMD_array (numpy.ndarray): Array containing BMD values.
        CORTMASK_array (numpy.ndarray): Array containing cortical mask values.
        TRABMASK_array (numpy.ndarray): Array containing trabecular mask values.
        cfg (dict): Configuration object containing image processing settings.
        IMTYPE (str): String defining the type of image ("BMD" or "NATIVE").

    Returns:
        tuple: A tuple containing the following elements:
            - BVTVscaled (numpy.ndarray): Scaled BVTV values.
            - BMDscaled (numpy.ndarray): Scaled BMD values.
            - BVTVraw (numpy.ndarray): Raw BVTV values.

    This function performs the following operations:
    1. Calculates BVTVraw based on the image type (BMD or NATIVE).
    2. Applies BVTV scaling if specified in the configuration.
    3. Creates a mask by combining cortical and trabecular masks.
    4. Applies the mask to BVTVscaled, BMDscaled, and BVTVraw.
    """

    print("\n ... prepare mask and BVTV images")
    print("     -> Scaling   = ", scaling)
    print("     -> Slope     = ", slope)
    print("     -> Intercept = ", intercept)

    if IMTYPE.find("BMD") > -1:
        # if image is already in BMD units (e.g. Hosseini's data)
        BVTVraw = BMD_array / 1200.0
    elif IMTYPE.find("NATIVE") > -1:
        BMD_array = (BMD_array / scaling) * slope + intercept
        BVTVraw = BMD_array / 1200.0  # if image is in native units

    # BVTV scaling
    if cfg.image_processing.bvtv_scaling == 1:
        seg_scaling_slope = cfg.image_processing.bvtv_slope
        seg_scaling_intercept = cfg.image_processing.bvtv_intercept
        BVTVscaled = seg_scaling_slope * BVTVraw + seg_scaling_intercept
    else:
        BVTVscaled = BVTVraw

    # set bone values
    MASK = CORTMASK_array + TRABMASK_array
    MASK[MASK > 0] = 1
    BVTVscaled = BVTVscaled * MASK
    BMDscaled = BVTVscaled * 1200 * MASK
    BVTVraw = BVTVraw * MASK
    return BVTVscaled, BMDscaled, BVTVraw


@timeit
def input_sanity_check(SEG_array, trabmask, cortmask, spacing, tolerance, dimZ):
    """
    Ensures the input data is in the correct format and initializes necessary variables.

    Args:
        SEG_array (numpy.ndarray): Image array of segmentation.
        trabmask (numpy.ndarray): Binary trabecular mask image.
        cortmask (numpy.ndarray): Binary cortical mask image.
        spacing (list): List of spacing in 3D.
        tolerance (float): Tolerance value for the triangulation process.
        dimZ (float): Dimension in the Z direction.

    Returns:
        tuple: A tuple containing the formatted SEG array, trabmask, cortmask, spacing, tolerance, and dimZ.
    """
    if not isinstance(SEG_array, vtk.vtkImageData):
        SEGim_vtk = numpy2vtk(SEG_array, spacing)

    trabmask = fmt_sanity_check(trabmask)
    cortmask = fmt_sanity_check(cortmask)
    spacing = fmt_sanity_check(spacing)
    tolerance = fmt_sanity_check(tolerance)
    dimZ = fmt_sanity_check(dimZ)

    return SEGim_vtk, trabmask, cortmask, spacing, tolerance, dimZ


def fmt_sanity_check(in_file):
    # check if file is a numpy.ndarray, else convert it
    if not isinstance(in_file, np.ndarray):
        out_file = np.array(in_file)
    else:
        out_file = in_file
    return out_file


@njit()
def __mask_cogs__(
    COG_temp: np.ndarray,
    spacing: np.ndarray,
):
    """
    Using numpy broadcasting to calculate the mask_cog
    Speed increased by 25x (POS, 2023-07-06)

    Args:
        COG_temp (np.ndarray): Array of center of gravity points
        Spacing (np.ndarray): SCANCO image spacing

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Centers of gravity in x, y, z direction
    """

    spac_0 = spacing[0]
    spac_1 = spacing[1]
    spac_2 = spacing[2]

    mask_cog1 = (COG_temp[:, 0] - (COG_temp[:, 0] % spac_0)) / spac_0
    mask_cog2 = (COG_temp[:, 1] - (COG_temp[:, 1] % spac_1)) / spac_1
    mask_cog3 = (COG_temp[:, 2] - (COG_temp[:, 2] % spac_2)) / spac_2
    return mask_cog1, mask_cog2, mask_cog3


def __assign_to_mask__(
    cfg,
    COG_temp: np.ndarray,
    trabmask: np.ndarray,
    mask_cog: np.ndarray,
    dimZ_min_tolerance: float,
    tolerance: float,
):
    """
    Assigns each point in COG_temp to either trabecular or cortical mask
    based on whether it is inside trabecular mask or not.
    Returns the cog_points and indices for trabecular and cortical masks separately.

    Args:
        COG_temp (np.ndarray): Array of center of gravity points
        trabmask (np.ndarray): Trabecular mask
        mask_cog (np.ndarray): Centers of gravity in x, y, z direction
        dimZ_min_tolerance (float): dimZ - tolerance
        tolerance (float): Tolerance for z-coordinates

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            cog_points_trab, indices_trab, cog_points_cort, indices_cort
    """
    cog_points_temp = np.array(COG_temp)
    z_coords = cog_points_temp[:, 2]
    in_trab_mask = (
        trabmask[
            mask_cog[:, 0].astype(np.int32),
            mask_cog[:, 1].astype(np.int32),
            mask_cog[:, 2].astype(np.int32),
        ]
        > 0
    )
    in_trab_mask &= (0 + tolerance <= z_coords) & (z_coords <= dimZ_min_tolerance)
    cog_points_trab = cog_points_temp[in_trab_mask]
    indices_trab = np.where(in_trab_mask)[0]

    if cfg.homogenization.orthotropic_cortex is True:
        cog_points_cort = None
        indices_cort = None

    else:
        in_cort_mask = (
            ~in_trab_mask
            & (0 + tolerance <= z_coords)
            & (z_coords <= dimZ_min_tolerance)
        )
        cog_points_cort = cog_points_temp[in_cort_mask]
        indices_cort = np.where(in_cort_mask)[0]
    return cog_points_trab, indices_trab, cog_points_cort, indices_cort


@timeit
def compute_dyadic_product_einsum(cfg, PointNormalArray, indices_cort, indices_trab):
    """
        Computes the dyadic product of the normal vectors of the cells in the cortical and trabecular regions
        of the triangulated surface using the Einstein summation convention.

    Args:
        PointNormalArray (vtk.vtkDataArray): The vtkDataArray containing the normal vectors of the cells.
        indices_cort (np.ndarray): The indices of the cells in the cortical region.
        indices_trab (np.ndarray): The indices of the cells in the trabecular region.

    Returns:
        Tuple[list, list]: The dyadic product of the normal vectors of the cells
        in the cortical and trabecular regions, respectively.
    """

    def _compute_dyad(PointNormalArray, indices, batch_size=1000):
        dyads = []
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i : i + batch_size]
            batch_normals = [PointNormalArray.GetTuple(j) for j in batch_indices]
            batch_dyads = np.einsum("ij,ik->ijk", batch_normals, batch_normals)
            dyads.extend(batch_dyads)
        return dyads

    if cfg.homogenization.orthotropic_cortex is True:
        dyadic_cort_einsum = None
    else:
        dyadic_cort_einsum = _compute_dyad(
            PointNormalArray, indices_cort, batch_size=1000
        )
    dyadic_trab_einsum = _compute_dyad(PointNormalArray, indices_trab, batch_size=1000)
    logger.info("4b/6 Computation dyadic products finished")
    return dyadic_cort_einsum, dyadic_trab_einsum


@timeit
def compute_cell_area(cfg, vtkNormals, indices_cort, indices_trab):
    """
    Computes the area of each cell in the cortical and trabecular regions of the triangulated surface.

    Args:
        vtkNormals (vtk.vtkPolyDataNormals): The vtkPolyDataNormals object containing the triangulated surface normals.
        indices_cort (np.ndarray): The indices of the cells in the cortical region.
        indices_trab (np.ndarray): The indices of the cells in the trabecular region.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The area of each cell in the cortical and trabecular regions, respectively.
    """
    # Get Cell Area https://www.vtk.org/Wiki/VTK/Examples/Python/MeshLabelImage
    triangleCellAN = vtk.vtkMeshQuality()
    triangleCellAN.SetInputConnection(vtkNormals.GetOutputPort())
    triangleCellAN.SetTriangleQualityMeasureToArea()
    triangleCellAN.SaveCellQualityOn()  # default
    triangleCellAN.Update()  # creates vtkDataSet
    qualityArray = triangleCellAN.GetOutput().GetCellData().GetArray("Quality")

    qualityArray_np = vtk_to_numpy(qualityArray)

    if cfg.homogenization.orthotropic_cortex is True:
        area_cort = None
    else:
        area_cort = qualityArray_np[indices_cort]
    area_trab = qualityArray_np[indices_trab]

    logger.info("5/6 Computation cell area finished")
    return area_cort, area_trab


@timeit
def get_cell_centers(vtk_output):
    """
    Finds the center of gravity of each cell in a vtk output object.

    Args:
        vtk_output (vtk.vtkPolyData): The vtk output object.

    Returns:
        numpy.ndarray: Array containing the center of gravity of each cell.
    """
    # Find find Center of gravity of each cell
    filt = vtk.vtkCellCenters()
    filt.SetInputDataObject(vtk_output)
    filt.Update()
    cog_temp_s = dsa.WrapDataObject(filt.GetOutput()).Points
    cog_temp = fmt_sanity_check(cog_temp_s)
    logger.info("2/6 Calculation of cell centers finished")
    return cog_temp


@timeit
def assign_vtk2cell(cfg, cog_temp, spacing, dimZ, tolerance, trabmask):
    """
    Assigns each triangle of the mesh to either the trabecular or cortical mask based on its center of gravity.

    Args:
        cfg (dict): Configuration object containing homogenization settings.
        cog_temp (numpy.ndarray): Array containing the center of gravity coordinates for each triangle.
        spacing (numpy.ndarray): Array containing the voxel spacing in x, y, and z directions.
        dimZ (float): Maximum z-coordinate of the mesh.
        tolerance (float): Tolerance for the cortical compartment.
        trabmask (numpy.ndarray): Array containing the trabecular mask.

    Returns:
        tuple: A tuple containing the center of gravity coordinates and indices for the triangles assigned to
               the trabecular and cortical masks, respectively.
    """
    mask_cog1, mask_cog2, mask_cog3 = __mask_cogs__(cog_temp, spacing)
    mask_cog = np.array([mask_cog1, mask_cog2, mask_cog3], dtype=np.int32).transpose()

    dimZ_min_tolerance = dimZ - tolerance
    (
        cog_points_trab,
        indices_trab,
        cog_points_cort,
        indices_cort,
    ) = __assign_to_mask__(
        cfg, cog_temp, trabmask, mask_cog, dimZ_min_tolerance, tolerance
    )
    logger.info("3/6 Assignment of vtk cells to trabecular and cortical mask finished")
    return cog_points_trab, indices_trab, cog_points_cort, indices_cort


@timeit
def compute_cell_normals(STL):
    """
    Computes the normal vectors of the cells in a triangulated surface.

    Args:
        STL (vtk.vtkPolyData): The triangulated surface.

    Returns:
        Tuple[vtk.vtkDataArray, vtk.vtkPolyDataNormals]:
        The vtkDataArray containing the normal vectors of the cells and the vtkPolyDataNormals object.
    """
    # calc cell normals and dyadic product
    vtkNormals = vtk.vtkPolyDataNormals()
    vtkNormals.SetInputData(STL)
    vtkNormals.ComputeCellNormalsOn()
    vtkNormals.ComputePointNormalsOff()
    vtkNormals.ConsistencyOn()
    vtkNormals.AutoOrientNormalsOn()  # Only works with closed surface. All Normals will point outward.
    vtkNormals.Update()
    PointNormalArray = vtkNormals.GetOutput().GetCellData().GetNormals()
    logger.info("4a/6 Cell normals calculated")
    return PointNormalArray, vtkNormals


@timeit
def get_area_dyadic(
    cfg,
    area_cort: np.ndarray,
    area_trab: np.ndarray,
    dyadic_cort: list[np.ndarray],
    dyadic_trab: list[np.ndarray],
):
    """
    Computes the area dyadic, which represents the multiplication of the area
    with the cross-product of the normals of each triangle.

    Args:
        cfg (dict): Configuration object containing homogenization settings.
        area_cort (numpy.ndarray): An array of cortical area values.
        area_trab (numpy.ndarray): An array of trabecular area values.
        dyadic_cort (list[numpy.ndarray]): A list of cortical dyadic product values.
        dyadic_trab (list[numpy.ndarray]): A list of trabecular dyadic product values.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: A tuple containing the computed cortical
                                    and trabecular area dyadic values.
    """
    if cfg.homogenization.orthotropic_cortex is True:
        areadyadic_cort = None
    else:
        reshaped_area_cort = np.reshape(area_cort, (-1, 1, 1))
    reshaped_area_trab = np.reshape(area_trab, (-1, 1, 1))

    if cfg.homogenization.orthotropic_cortex is True:
        areadyadic_cort = None
    else:
        areadyadic_cort = np.multiply(reshaped_area_cort, dyadic_cort)
    areadyadic_trab = np.multiply(reshaped_area_trab, dyadic_trab)
    logger.info("6/6 Computation dyadic areas finished")
    return areadyadic_cort, areadyadic_trab


def smooth_kernel(MSL: np.ndarray, ROI_kernel_size: int) -> np.ndarray:
    """
    Applies a smoothing kernel to a 3D numpy array.

    Args:
        MSL (np.ndarray): A 3D numpy array representing the input data.
        ROI_kernel_size (int): The size of the smoothing kernel.

    Returns:
        np.ndarray: A 3D numpy array representing the smoothed data.

    Examples:
        >>> data = np.random.rand(10, 10, 10)
        >>> smoothed_data = smooth_kernel(data, 3)
    """
    kernel = np.ones([ROI_kernel_size, ROI_kernel_size, ROI_kernel_size])
    kernel = kernel[:, :, :, None, None]

    MSL_kernel = convolve(MSL, kernel, mode="constant", cval=0.0)

    return MSL_kernel


@timeit
def msl_triangulation(cfg, SEG_array, cortmask, trabmask, spacing, tolerance):
    """
    Evaluates MSL fabric tensors for cortical and trabecular regions by triangulating the surface.

    Args:
        cfg (dict): Configuration object containing homogenization settings.
        SEG_array (numpy.ndarray): Image array of segmentation.
        cortmask (numpy.ndarray): Binary cortical mask image.
        trabmask (numpy.ndarray): Binary trabecular mask image.
        spacing (list): List of spacing in 3D.
        tolerance (float): Tolerance value for the triangulation process.

    Returns:
        tuple: A tuple containing the following elements:
            - cog_points_cort (numpy.ndarray): Center of gravity of triangles in cortical phase.
            - cog_points_trab (numpy.ndarray): Center of gravity of triangles in trabecular phase.
            - areadyadic_cort (numpy.ndarray): Area-weighted dyadic product of cortical triangles.
            - areadyadic_trab (numpy.ndarray): Area-weighted dyadic product of trabecular triangles.
            - nfacet_range (numpy.ndarray): Number of triangles in specific phase.
            - indices_cort (numpy.ndarray): Indices of triangles in cortical phase.
            - indices_trab (numpy.ndarray): Indices of triangles in trabecular phase.

    Notes
    -----
    - The function is based on the original code by D. Schenk (2018-2022)
    - Adaptation of assign_MSL_triangulation function to account
        for the transformation (M. Indermaur, 2023)
    - Improved memory and CPU time performance (S. Poncioni, 2023)
    - Cortical compartment as transverse isotropic material, uses cortical mask to calculate (S. Poncioni, 2024)
    """
    # TODO: mask SEG_vtk with size of trabmask (we don't calculate everything also for cortex)
    ORTHOTROPIC_CORTEX = cfg.homogenization.orthotropic_cortex
    if ORTHOTROPIC_CORTEX is True:
        # mask SEG_vtk with trabmask with boolean
        # SEG_array = np.where(trabmask, SEG_array, 0)
        pass

    # * 0/6 Input sanity check
    try:
        dimZ = (np.shape(SEG_array) * spacing)[2]
    except TypeError:
        spacing = np.array([spacing[0], spacing[1], spacing[2]])
        dimZ = (np.shape(SEG_array) * spacing)[2]

    SEG_vtk, trabmask, cortmask, spacing, tolerance, dimZ = input_sanity_check(
        SEG_array, trabmask, cortmask, spacing, tolerance, dimZ
    )

    # * 1/6 STL file creation for trabecular compartment
    surfnet_output = surface_nets(
        SEG_vtk,
        output_mesh_type="tri",
        output_style="boundary",
        smoothing=True,
        decimate=True,
        smoothing_num_iterations=10,
    )

    del SEG_vtk
    gc.collect()

    # * 2/6 Calculation of Number of cells
    cog_temp = get_cell_centers(surfnet_output)
    nfacet = surfnet_output.GetNumberOfCells()

    # * 3/6 Computation COG
    (
        cog_points_trab,
        indices_trab,
        cog_points_cort,
        indices_cort,
    ) = assign_vtk2cell(
        cfg,
        cog_temp,
        spacing,
        dimZ,
        tolerance,
        trabmask,
    )

    # * 4/6 Computation cell normals and dyadic products
    PointNormalArray, vtkNormals = compute_cell_normals(surfnet_output)

    dyadic_cort, dyadic_trab = compute_dyadic_product_einsum(
        cfg, PointNormalArray, indices_cort, indices_trab
    )

    del (
        cog_temp,
        surfnet_output,
        PointNormalArray,
    )
    gc.collect()

    # * 5/6 Computation of the cell area
    area_cort, area_trab = compute_cell_area(
        cfg, vtkNormals, indices_cort, indices_trab
    )

    # * 6/6 Computation of the area dyadic
    areadyadic_cort, areadyadic_trab = get_area_dyadic(
        cfg, area_cort, area_trab, dyadic_cort, dyadic_trab
    )

    nfacet_range = np.arange(nfacet)

    return (
        cog_points_cort,
        cog_points_trab,
        areadyadic_cort,
        areadyadic_trab,
        nfacet_range,
        indices_cort,
        indices_trab,
    )


def compute_msl_spline(bone: dict, cfg: dict) -> dict:
    """
    Computes the mean surface length (MSL) for a given bone image and configuration.

    Args:
        bone (dict): A dictionary containing bone data, including spacing, segmentation arrays, and masks.
        cfg (dict): A configuration dictionary containing homogenization parameters.

    Returns:
        dict: The updated bone dictionary with computed MSL spline values.

    """

    # read config dict
    STL_tolerance = cfg.homogenization.STL_tolerance
    ROI_kernel_size_cort = cfg.homogenization.ROI_kernel_size_cort
    ROI_kernel_size_trab = cfg.homogenization.ROI_kernel_size_trab

    # read bone dict
    spacing = bone["Spacing"]
    SEG_array = bone["SEG_array"]
    TRABMASK_array = bone["TRABMASK_array"]
    CORTMASK_array = bone["CORTMASK_array"]

    (
        cog_points_cort,
        cog_points_trab,
        areadyadic_cort,
        areadyadic_trab,
        nfacet_range,
        indices_cort,
        indices_trab,
    ) = msl_triangulation(
        cfg, SEG_array, CORTMASK_array, TRABMASK_array, spacing, STL_tolerance
    )

    # ? maybe I won't need to copy these into the bone dict (POS, 10.07.2023)
    bone["cog_points_cort"] = cog_points_cort
    bone["cog_points_trab"] = cog_points_trab
    bone["areadyadic_cort"] = areadyadic_cort
    bone["areadyadic_trab"] = areadyadic_trab
    bone["nfacet"] = nfacet_range
    bone["indizes_cort"] = indices_cort
    bone["indizes_trab"] = indices_trab

    DIMS = np.floor(np.array(bone["BVTVscaled"].shape) / bone["CoarseFactor"])
    DIMS_int = DIMS.astype(int)

    # Assign areadyadic values
    # ----------------------------------------------------------------
    # Each areadyadic value of a triangle is added to the pool of FE element it's lying in
    if cfg.homogenization.orthotropic_cortex is True:
        MSL_values_cort = None
        MSL_kernel_list_cort = None
        # Calculate projected eigenvector from cortical mask
        (
            evect_origin,
            evect,
        ) = clustered_point_normals(cfg, CORTMASK_array, TRABMASK_array, spacing)
        bone["evect_origin"] = evect_origin
        bone["cort_projection_evect"] = evect
    else:
        MSL_values_cort = map_isosurface(
            cog_points_cort, areadyadic_cort, DIMS=DIMS_int
        )
        MSL_kernel_cort = smooth_kernel(MSL_values_cort, ROI_kernel_size_cort)
        MSL_kernel_cort = np.transpose(MSL_kernel_cort, (2, 1, 0, 3, 4))
        MSL_kernel_list_cort = np.reshape(
            MSL_kernel_cort, (int(np.size(MSL_kernel_cort) / 9), 3, 3)
        )

    MSL_values_trab = map_isosurface(cog_points_trab, areadyadic_trab, DIMS=DIMS_int)
    MSL_kernel_trab = smooth_kernel(MSL_values_trab, ROI_kernel_size_trab)
    # transpose to conventional orientation
    MSL_kernel_trab = np.transpose(MSL_kernel_trab, (2, 1, 0, 3, 4))

    MSL_kernel_list_trab = np.reshape(
        MSL_kernel_trab, (int(np.size(MSL_kernel_trab) / 9), 3, 3)
    )

    bone["MSL_kernel_list_trab"] = MSL_kernel_list_trab
    bone["MSL_kernel_list_cort"] = MSL_kernel_list_cort

    return bone


def set_summary_variables(bone):
    """
    Computes variables for summary file
    """

    # get bone values
    BMDscaled = bone["BMDscaled"]
    BVTVscaled = bone["BVTVscaled"]

    CORTMASK_array = bone["CORTMASK_array"]
    CORTMASK_array[CORTMASK_array > 0] = 1
    TRABMASK_array = bone["TRABMASK_array"]
    TRABMASK_array[TRABMASK_array > 0] = 1

    MASK_array = np.add(CORTMASK_array, TRABMASK_array)
    # MASK_array[MASK_array > 0] = 1

    FEelSize = bone["FEelSize"]
    Spacing = bone["Spacing"]

    RHOc_array = bone["RHOc_array"]
    RHOc_FE_array = bone["RHOc_FE_array"]
    PHIc_array = bone["PHIc_array"]

    RHOt_array = bone["RHOt_array"]
    RHOt_FE_array = bone["RHOt_FE_array"]
    PHIt_array = bone["PHIt_array"]

    RHOc_orig_array = bone["RHOc_orig_array"]
    RHOt_orig_array = bone["RHOt_orig_array"]

    # Computation of variables for summary file
    # ------------------------------------------------------
    # ------------------------------------------------------
    variables = {}

    # Mask volume [mm^3]
    # ------------------------------------------------------
    # Mask volume from MASK array
    variables["Mask_Volume_CORTMASK"] = np.sum(CORTMASK_array * Spacing[1] ** 3)
    variables["Mask_Volume_TRABMASK"] = np.sum(TRABMASK_array * Spacing[1] ** 3)
    variables["Mask_Volume_MASK"] = (
        variables["Mask_Volume_CORTMASK"] + variables["Mask_Volume_TRABMASK"]
    )

    # Mask volume from FE elememts
    variables["CORTMask_Volume_FE"] = np.sum(PHIc_array * FEelSize[1] ** 3)
    variables["TRABMask_Volume_FE"] = np.sum(PHIt_array * FEelSize[1] ** 3)
    # Ratio quality check mesh
    variables["CORTVolume_ratio"] = (
        variables["CORTMask_Volume_FE"] / variables["Mask_Volume_CORTMASK"]
    )
    variables["TRABVolume_ratio"] = (
        variables["TRABMask_Volume_FE"] / variables["Mask_Volume_TRABMASK"]
    )
    variables["TOTVolume_ratio"] = (
        variables["TRABMask_Volume_FE"] + variables["CORTMask_Volume_FE"]
    ) / (variables["Mask_Volume_TRABMASK"] + variables["Mask_Volume_CORTMASK"])

    # BMD computation [mgHA/ccm]
    # ------------------------------------------------------
    variables["TOT_mean_BMD_image"] = np.mean(BMDscaled[MASK_array > 0])
    variables["CORT_mean_BMD_image"] = np.mean(BMDscaled[CORTMASK_array > 0])
    variables["TRAB_mean_BMD_image"] = np.mean(BMDscaled[TRABMASK_array > 0])

    # BMC
    variables["TOT_mean_BMC_image"] = (
        np.mean(BMDscaled[MASK_array > 0]) * variables["Mask_Volume_MASK"] / 1000
    )
    variables["CORT_mean_BMC_image"] = (
        np.mean(BMDscaled[CORTMASK_array > 0])
        * variables["Mask_Volume_CORTMASK"]
        / 1000
    )
    variables["TRAB_mean_BMC_image"] = (
        np.mean(BMDscaled[TRABMASK_array > 0])
        * variables["Mask_Volume_TRABMASK"]
        / 1000
    )

    variables["CORT_simulation_BMC_FE_tissue_ROI"] = (
        np.sum(RHOc_array * PHIc_array * FEelSize[0] ** 3 * 1200) / 1000
    )
    variables["CORT_simulation_BMC_FE_tissue_orig_ROI"] = (
        np.sum(RHOc_orig_array * PHIc_array * FEelSize[0] ** 3 * 1200) / 1000
    )
    variables["TRAB_simulation_BMC_FE_tissue_ROI"] = (
        np.sum(RHOt_array * PHIt_array * FEelSize[0] ** 3 * 1200) / 1000
    )
    variables["TRAB_simulation_BMC_FE_tissue_orig_ROI"] = (
        np.sum(RHOt_orig_array * PHIt_array * FEelSize[0] ** 3 * 1200) / 1000
    )
    variables["TOT_simulation_BMC_FE_tissue_ROI"] = (
        variables["CORT_simulation_BMC_FE_tissue_ROI"]
        + variables["TRAB_simulation_BMC_FE_tissue_ROI"]
    )
    variables["TOT_simulation_BMC_FE_tissue_orig_ROI"] = (
        variables["CORT_simulation_BMC_FE_tissue_orig_ROI"]
        + variables["TRAB_simulation_BMC_FE_tissue_orig_ROI"]
    )
    # Quality check
    # corrected, with BMC conversion
    variables["TRAB_BMC_ratio_ROI"] = (
        variables["TRAB_simulation_BMC_FE_tissue_ROI"]
        / variables["TRAB_mean_BMC_image"]
    )
    variables["CORT_BMC_ratio_ROI"] = (
        variables["CORT_simulation_BMC_FE_tissue_ROI"]
        / variables["CORT_mean_BMC_image"]
    )
    variables["TOT_BMC_ratio_ROI"] = (
        variables["TOT_simulation_BMC_FE_tissue_ROI"] / variables["TOT_mean_BMC_image"]
    )
    # original, without correction
    variables["TRAB_BMC_ratio_orig_ROI"] = (
        variables["TRAB_simulation_BMC_FE_tissue_orig_ROI"]
        / variables["TRAB_mean_BMC_image"]
    )
    variables["CORT_BMC_ratio_orig_ROI"] = (
        variables["CORT_simulation_BMC_FE_tissue_orig_ROI"]
        / variables["CORT_mean_BMC_image"]
    )
    variables["TOT_BMC_ratio_orig_ROI"] = (
        variables["TOT_simulation_BMC_FE_tissue_orig_ROI"]
        / variables["TOT_mean_BMC_image"]
    )

    # BVTV computation [%]
    # ------------------------------------------------------
    # mean tissue BVTV
    variables["TOT_BVTV_tissue"] = np.mean(BVTVscaled[MASK_array == 1])
    variables["CORT_BVTV_tissue"] = np.mean(BVTVscaled[CORTMASK_array == 1])
    variables["TRAB_BVTV_tissue"] = np.mean(BVTVscaled[TRABMASK_array == 1])

    # BVTV from FE elements, but only in ROI
    variables["CORT_simulation_BVTV_FE_tissue_ROI"] = np.sum(
        RHOc_array * PHIc_array
    ) / np.sum(PHIc_array)
    variables["TRAB_simulation_BVTV_FE_tissue_ROI"] = np.sum(
        RHOt_array * PHIt_array
    ) / np.sum(PHIt_array)

    variables["CORT_simulation_BVTV_FE_tissue_orig_ROI"] = np.sum(
        RHOc_orig_array * PHIc_array
    ) / np.sum(PHIc_array)
    variables["TRAB_simulation_BVTV_FE_tissue_orig_ROI"] = np.sum(
        RHOt_orig_array * PHIt_array
    ) / np.sum(PHIt_array)

    variables["CORT_simulation_BVTV_FE_tissue_ELEM"] = np.sum(
        RHOc_FE_array * PHIc_array
    ) / np.sum(PHIc_array)
    variables["TRAB_simulation_BVTV_FE_tissue_ELEM"] = np.sum(
        RHOt_FE_array * PHIt_array
    ) / np.sum(PHIt_array)

    # BVTV from FE elements, including full element volume (as well volume of FE elements outside of mask)
    variables["CORT_simulation_BVTV_FE_elements_ROI"] = np.sum(
        RHOc_array * PHIc_array
    ) / len(PHIc_array)
    variables["TRAB_simulation_BVTV_FE_elements_ROI"] = np.sum(
        RHOt_array * PHIt_array
    ) / len(PHIt_array)

    variables["CORT_simulation_BVTV_FE_elements_orig_ROI"] = np.sum(
        RHOc_orig_array * PHIc_array
    ) / len(PHIc_array)
    variables["TRAB_simulation_BVTV_FE_elements_orig_ROI"] = np.sum(
        RHOt_orig_array * PHIt_array
    ) / len(PHIt_array)

    variables["CORT_simulation_BVTV_FE_elements_ELEM"] = np.sum(
        RHOc_array * PHIc_array
    ) / len(PHIc_array)
    variables["TRAB_simulation_BVTV_FE_elements_ELEM"] = np.sum(
        RHOt_array * PHIt_array
    ) / len(PHIt_array)
    variables["TOT_simulation_BVTV_FE_elements_ELEM"] = (
        np.sum(RHOt_array * PHIt_array) + np.sum(RHOc_array + PHIc_array)
    ) / (len(PHIt_array) + len(PHIc_array))

    variables["CORT_BVTV_ratio_ROI"] = (
        variables["CORT_simulation_BVTV_FE_tissue_ROI"] / variables["CORT_BVTV_tissue"]
    )
    variables["TRAB_BVTV_ratio_ROI"] = (
        variables["TRAB_simulation_BVTV_FE_tissue_ROI"] / variables["TRAB_BVTV_tissue"]
    )

    variables["CORT_BVTV_ratio_ELEM"] = (
        variables["CORT_simulation_BVTV_FE_tissue_ELEM"] / variables["CORT_BVTV_tissue"]
    )
    variables["TRAB_BVTV_ratio_ELEM"] = (
        variables["TRAB_simulation_BVTV_FE_tissue_ELEM"] / variables["TRAB_BVTV_tissue"]
    )

    # Bone volume BV [mm^3]
    variables["TOT_BV_tissue"] = (
        variables["TOT_BVTV_tissue"] * variables["Mask_Volume_MASK"]
    )
    variables["CORT_BV_tissue"] = (
        variables["CORT_BVTV_tissue"] * variables["Mask_Volume_CORTMASK"]
    )
    variables["TRAB_BV_tissue"] = (
        variables["TRAB_BVTV_tissue"] * variables["Mask_Volume_TRABMASK"]
    )

    # BMC [mgHA]
    # ------------------------------------------------------
    # tissue BMC from mask and BMD image
    variables["TOT_BMC_tissue"] = (
        variables["TOT_mean_BMD_image"] * variables["Mask_Volume_MASK"] / 1000
    )
    variables["CORT_BMC_tissue"] = (
        variables["CORT_mean_BMD_image"] * variables["Mask_Volume_CORTMASK"] / 1000
    )
    variables["TRAB_BMC_tissue"] = (
        variables["TRAB_mean_BMD_image"] * variables["Mask_Volume_TRABMASK"] / 1000
    )

    # BMC FE tissue (BVTV that FE simulation uses --> homogenized as ROI is bigger than FE element)
    variables["CORT_simulation_BMC_FE_tissue_ROI"] = (
        np.sum(RHOc_array * PHIc_array * FEelSize[0] ** 3 * 1200) / 1000
    )
    variables["TRAB_simulation_BMC_FE_tissue_ROI"] = (
        np.sum(RHOt_array * PHIt_array * FEelSize[0] ** 3 * 1200) / 1000
    )
    variables["TOT_simulation_BMC_FE_tissue_ROI"] = (
        variables["CORT_simulation_BMC_FE_tissue_ROI"]
        + variables["TRAB_simulation_BMC_FE_tissue_ROI"]
    )

    # BMC FE tissue that should be equal to total tissue BMC, as only RHO inside FE element is considered.
    variables["CORT_simulation_BMC_FE_tissue_ELEM"] = (
        np.sum(RHOc_FE_array * PHIc_array * FEelSize[0] ** 3 * 1200) / 1000
    )
    variables["TRAB_simulation_BMC_FE_tissue_ELEM"] = (
        np.sum(RHOt_FE_array * PHIt_array * FEelSize[0] ** 3 * 1200) / 1000
    )
    variables["TOT_simulation_BMC_FE_tissue_ELEM"] = (
        variables["CORT_simulation_BMC_FE_tissue_ELEM"]
        + variables["TRAB_simulation_BMC_FE_tissue_ELEM"]
    )

    # Ratio quality check mesh
    variables["TOT_BMC_ratio_ROI"] = (
        variables["TOT_simulation_BMC_FE_tissue_ROI"] / variables["TOT_BMC_tissue"]
    )
    variables["CORT_BMC_ratio_ROI"] = (
        variables["CORT_simulation_BMC_FE_tissue_ROI"] / variables["CORT_BMC_tissue"]
    )
    variables["TRAB_BMC_ratio_ROI"] = (
        variables["TRAB_simulation_BMC_FE_tissue_ROI"] / variables["TRAB_BMC_tissue"]
    )

    variables["TOT_BMC_ratio_ELEM"] = (
        variables["TOT_simulation_BMC_FE_tissue_ELEM"] / variables["TOT_BMC_tissue"]
    )
    variables["CORT_BMC_ratio_ELEM"] = (
        variables["CORT_simulation_BMC_FE_tissue_ELEM"] / variables["CORT_BMC_tissue"]
    )
    variables["TRAB_BMC_ratio_ELEM"] = (
        variables["TRAB_simulation_BMC_FE_tissue_ELEM"] / variables["TRAB_BMC_tissue"]
    )

    return variables
