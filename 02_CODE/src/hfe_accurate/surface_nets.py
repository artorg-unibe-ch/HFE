import logging
from typing import Literal

import fast_simplification as fs  # type: ignore
import pyvista as pv
import vtk  # type: ignore
from hfe_utils.io_utils import timeit

LOGGING_NAME = "HFE-ACCURATE"
logger = logging.getLogger(LOGGING_NAME)
logger.propagate = False


@timeit
def surface_nets(
    imvtk,
    output_mesh_type: Literal["quads", "tri"] = "tri",
    output_style: Literal["default", "boundary"] = "default",
    smoothing: bool = False,
    decimate: bool = False,
    smoothing_num_iterations: int = 50,
    target_reduction: float = 0.9,
):
    """
    Applies the Surface Nets algorithm to a given vtkImageData object to generate a mesh.

    Args:
        imvtk (vtk.vtkImageData): The input image data.
        output_mesh_type (str, optional): The type of mesh to output ("quads" or "tri"). Defaults to "tri".
        output_style (str, optional): The style of the output ("default" or "boundary"). Defaults to "default".
        smoothing (bool, optional): Whether to apply smoothing to the output. Defaults to False.
        decimate (bool, optional): Whether to apply decimation to the output. Defaults to False.
        smoothing_num_iterations (int, optional): The number of smoothing iterations. Defaults to 50.
        target_reduction (float, optional): The target reduction for decimation. Defaults to 0.9.

    Returns:
        pv.core.pointset.UnstructuredGrid: Mesh after applying Surface Nets.
    """
    TARGET_REDUCTION = target_reduction

    num_labels = int(imvtk.GetPointData().GetScalars().GetRange()[1])

    surfnets = vtk.vtkSurfaceNets3D()
    surfnets.SetInputData(imvtk)

    if num_labels is not None:
        surfnets.GenerateLabels(num_labels, 1, num_labels)

    if output_mesh_type not in ["quads", "tri"]:
        raise ValueError(
            f'Invalid output mesh type "{output_mesh_type}", use "quads" or "tri"'
        )

    if output_style not in ["default", "boundary"]:
        raise ValueError(
            f'Invalid output style "{output_style}", use "default" or "boundary"'
        )

    if output_mesh_type == "quads":
        surfnets.SetOutputMeshTypeToQuads()
    elif output_mesh_type == "tri":
        surfnets.SetOutputMeshTypeToTriangles()
    if output_style == "default":
        surfnets.SetOutputStyleToDefault()
    elif output_style == "boundary":
        surfnets.SetOutputStyleToBoundary()

    surfnets.Update()
    mesh = pv.wrap(surfnets.GetOutput())

    # Decimation
    if decimate:
        mesh = fs.simplify_mesh(
            mesh, target_reduction=TARGET_REDUCTION, agg=8, verbose=True
        )

    # Smoothing
    if smoothing:
        mesh = mesh.smooth_taubin(n_iter=smoothing_num_iterations, pass_band=0.5)

    # save mesh for debugging
    # import datetime # put at the top of the file
    # timenow = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # mesh.save(f"mesh_{timenow}.vtk")

    logger.info("1/6 STL file creation finished")
    return mesh
