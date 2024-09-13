import cProfile
from math import floor, sqrt

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import SimpleITK as sitk
from hfe_accurate.surface_nets import surface_nets
from scipy import ndimage  # type: ignore

# flake8: noqa: E501


def clustered_point_normals(
    cfg, CORTmask_array: np.ndarray, TRABmask_array: np.ndarray, spacing: list
):
    # peel off the first and last 10 slices
    CORTmask_array = CORTmask_array[:, :, 10:-10]
    TRABmask_array = TRABmask_array[:, :, 10:-10]

    def __fill_mask__() -> np.ndarray:
        """
        Fills the mask by combining CORTmask_array and TRABmask_array.

        Returns:
            np.ndarray: The filled mask.
        """
        mask = CORTmask_array + TRABmask_array
        cort_filled = np.where(mask > 0, 1, 0)
        return cort_filled

    def __get__normals__(surfnet_output: pv.PolyData):
        """
        Computes and normalizes the normals of a given PolyData object.

        Args:
            surfnet_output (pv.PolyData): The PolyData object for which to compute and normalize the normals.

        Returns:
            tuple: A tuple containing the points of the PolyData object and their corresponding normalized normals.
        """
        surfnet_output.compute_normals(
            inplace=True, consistent_normals=False, auto_orient_normals=True
        )

        point_normals = surfnet_output.point_normals
        points = surfnet_output.points
        norms = np.linalg.norm(point_normals, axis=1)
        point_normals_normalized = point_normals / norms[:, np.newaxis]

        return points, point_normals_normalized

    def __cluster_normals__(points: np.ndarray, point_normals_normalized: np.ndarray):
        """
        Clusters the normals of points in a 3D grid.
        It first normalizes and scales the points to the grid size.
        Then, for each point, it adds its corresponding normal to the sum of the normals
        at the grid position of the point and increments the count of vectors at that position.
        Finally, it computes the average of the vectors at each grid position and returns
        the flattened average vectors and grid points.

        Args:
            points (np.ndarray): The points for which to cluster the normals.
            point_normals_normalized (np.ndarray): The normalized normals of the points.

        Returns:
            tuple: A tuple containing the flattened average vectors and grid points.
        """
        _img_dim = cort_filled.shape[0]
        coarse_factor = cfg.mesher.element_size / spacing[0]
        _grid = floor(_img_dim / coarse_factor)
        grid_size = (_grid, _grid, _grid)

        min_vals = points.min(axis=0)
        max_vals = points.max(axis=0)

        points_normalized = (points - min_vals) / (max_vals - min_vals)
        points_scaled = points_normalized * (np.array(grid_size) - 1)

        vector_sum = np.zeros(grid_size + (3,))
        vector_count = np.zeros(grid_size)

        for point, vector in zip(points_scaled, point_normals_normalized):
            grid_pos = tuple(point.astype(int))
            vector_sum[grid_pos] += vector
            vector_count[grid_pos] += 1

        average_vectors = vector_sum / np.maximum(vector_count, 1)[:, :, :, np.newaxis]
        grid_x, grid_y, grid_z = np.mgrid[0:_grid, 0:_grid, 0:_grid]
        grid_points = (
            np.stack((grid_x, grid_y, grid_z), axis=-1)
            * (max_vals - min_vals)
            / (_grid - 1)
            + min_vals
        )
        grid_points_flat = grid_points.reshape(-1, 3)
        average_vectors_flat = average_vectors.reshape(-1, 3)

        # Calculate centroids of the grid points
        x, y, z = np.meshgrid(
            np.arange(0.5, grid_size[0] + 0.5),
            np.arange(0.5, grid_size[1] + 0.5),
            np.arange(0.5, grid_size[2] + 0.5),
            indexing="ij",
        )
        x = x.flatten() * coarse_factor
        y = y.flatten() * coarse_factor
        z = z.flatten() * coarse_factor
        MSL_centroids = np.column_stack((x, y, z))
        MSL_centroids_r = np.reshape(MSL_centroids, (-1, 3))
        MSL_centroids_mm = MSL_centroids_r * spacing

        return average_vectors_flat, grid_points_flat, MSL_centroids_mm

    def __ray_tracing_normals__(
        average_vectors_flat: np.ndarray,
        grid_points_flat: np.ndarray,
        surfnet_output: pv.PolyData,
    ):
        """
        This function performs ray tracing on the given points and vectors.
        For each point, it casts a ray in the direction of the corresponding vector.
        If the ray intersects the surface and the intersection is within a maximum distance, the point is moved to the intersection.
        If the ray does not intersect the surface or the intersection is too far, a ray is cast in the opposite direction.
        If neither ray intersects the surface or the intersections are too far, the point is left where it is.
        Finally, it plots the ray traced normals and returns the moved points and their corresponding vectors.

        Args:
            average_vectors_flat (np.ndarray): The vectors for which to perform ray tracing.
            grid_points_flat (np.ndarray): The points at which to perform ray tracing.
            surfnet_output (pv.PolyData): The PolyData object representing the surface to intersect.

        Returns:
            tuple: A tuple containing the moved points and their corresponding vectors.
        """
        # Define line segment
        direction = average_vectors_flat
        moved_grid_points_flat = np.empty_like(grid_points_flat)

        max_distance = 10 // spacing[0]
        scaling_factor = 1e2

        for i, point in enumerate(grid_points_flat):
            # Scale and cast a ray in the direction of the vector
            dir = average_vectors_flat[i]
            dir_vector = point + scaling_factor * dir

            points, _ = surfnet_output.ray_trace(point, dir_vector)
            # If the ray intersects the surface and the intersection is within the maximum distance
            if (
                points.shape[0] > 0
                and np.linalg.norm(points[0] - point) <= max_distance
            ):
                moved_grid_points_flat[i] = points[0]
            else:
                # If the ray does not intersect the surface or the intersection is too far, cast a ray in the opposite direction
                dir_vector = point - scaling_factor * direction[i]
                points, _ = surfnet_output.ray_trace(point, dir_vector)
                if (
                    points.shape[0] > 0
                    and np.linalg.norm(points[0] - point) <= max_distance
                ):
                    moved_grid_points_flat[i] = points[0]
                else:
                    # If neither ray intersects the surface or the intersections are too far, leave the point where it is
                    moved_grid_points_flat[i] = point

        # plot ray traced normals
        # p = pv.Plotter()
        # p.add_mesh(surfnet_output, color="tan", show_edges=False)
        # p.add_arrows(moved_grid_points_flat, average_vectors_flat, mag=50)
        # p.show_axes()
        # p.show()
        return moved_grid_points_flat, average_vectors_flat

    def __project_onto_plane__(x, n):
        """
        This function projects a vector onto a plane defined by a normal vector.
        It first computes the dot product and the norm of the input vectors, and normalizes the normal vector.
        Then, it computes the projection of the input vector onto the normal vector and subtracts
        this projection from the input vector to obtain the projection onto the plane.

        Args:
            x (list): The input vector to be projected onto the plane.
            n (list): The normal vector defining the plane.

        Returns:
            list: The projection of the input vector onto the plane defined by the normal vector.
        """

        def dot_product(x, y):
            return sum([x[i] * y[i] for i in range(len(x))])

        def norm(x):
            return sqrt(dot_product(x, x))

        def normalize(x):
            return [x[i] / norm(x) for i in range(len(x))]

        d = dot_product(x, n) / norm(n)
        p = [d * normalize(n)[i] for i in range(len(n))]
        return [x[i] - p[i] for i in range(len(x))]

    def __plane_projection__(average_vectors_flat, vertical_vector):
        """
        This function projects a set of vectors onto a plane perpendicular to a given vertical vector.
        It first removes any zero vectors from the input set, then for each remaining vector,
        it projects it onto the plane defined by the vertical vector.
        The function returns the set of projected vectors.

        Args:
            average_vectors_flat (np.ndarray): The set of vectors to be projected onto the plane.
            vertical_vector (np.ndarray): The vector defining the plane onto which the vectors will be projected.

        Returns:
            np.ndarray: The set of vectors projected onto the plane defined by the vertical vector.
        """
        # Since we have a grid, some vectors will be [0, 0, 0], hence removing them
        average_vectors_flat_nonzero = average_vectors_flat[
            ~np.all(average_vectors_flat == 0, axis=1)
        ]

        projected_vectors = np.zeros_like(average_vectors_flat_nonzero)
        for i in range(len(average_vectors_flat_nonzero)):
            projected_vectors[i] = __project_onto_plane__(
                vertical_vector, average_vectors_flat_nonzero[i]
            )
        return projected_vectors

    cort_filled = __fill_mask__()
    mesh = pv.wrap(cort_filled)
    surfnet_output = surface_nets(
        mesh,
        output_mesh_type="tri",
        output_style="default",
        smoothing=True,
        decimate=True,
        smoothing_num_iterations=10,
        target_reduction=0.8,
    )

    points, point_normals_normalized = __get__normals__(surfnet_output)
    average_vectors_flat, grid_points_flat, MSL_centroids_mm = __cluster_normals__(
        points, point_normals_normalized
    )
    moved_grid_points_flat, average_vectors_flat = __ray_tracing_normals__(
        average_vectors_flat, grid_points_flat, surfnet_output
    )

    # Since we have a grid, some vectors will be [0, 0, 0], hence removing the same indices from the points
    moved_grid_points_flat_nonzero = moved_grid_points_flat[
        ~np.all(average_vectors_flat == 0, axis=1)
    ]

    average_vectors_flat_nonzero = average_vectors_flat[
        ~np.all(average_vectors_flat == 0, axis=1)
    ]

    MSL_centroids_mm_nonzero = MSL_centroids_mm[
        ~np.all(average_vectors_flat == 0, axis=1)
    ]

    # plot projected vectors
    projected_vectors = __plane_projection__(
        average_vectors_flat_nonzero, vertical_vector=[0, 0, 1]
    )

    evect = np.zeros((len(moved_grid_points_flat_nonzero), 3, 3))
    evect[:, 0] = average_vectors_flat_nonzero  # min
    evect[:, 1] = np.cross(projected_vectors, average_vectors_flat_nonzero)  # mid
    evect[:, 2] = projected_vectors  # max

    # p = pv.Plotter()
    # p.add_mesh(surfnet_output, color="tan", show_edges=False)
    # p.add_arrows(moved_grid_points_flat_nonzero, evect[:, 0], mag=30, color="blue")
    # p.add_arrows(moved_grid_points_flat_nonzero, evect[:, 1], mag=30, color="green")
    # p.add_arrows(moved_grid_points_flat_nonzero, evect[:, 2], mag=30, color="red")
    # p.show_axes()
    # p.show()

    return (
        moved_grid_points_flat_nonzero,
        evect,
    )
