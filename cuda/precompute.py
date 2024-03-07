"""
==========
Precompute
==========

This file contains the functions to precompute the geometric data that is
used by the operators.

Copyright (C) 2024 Adeeb Arif Kor
"""

import numpy as np
import numpy.typing as npt
import numba


@numba.njit(fastmath=True)
def compute_boundary_facets_scaled_jacobian_determinant(
        detJ_f: npt.NDArray[np.floating],
        mesh: tuple[npt.NDArray[np.int32], npt.NDArray[np.floating]],
        boundary_data: npt.NDArray[np.int32],
        dphi_f: npt.NDArray[np.floating],
        weights: npt.NDArray[np.floating]
):
    """
    Compute the boundary facets Jacobian determinant and scaled it with the
    quadrature weights.

    detJ_f = w_{q} |J_f{q}|

    Parameters
    ----------
    detJ_f : array for the output
    mesh : mesh topology and geometry
    boundary_data : array containing the cells and local facets indices on the
        boundary.
    dphi_f : derivatives of the basis functions on the cell facets.
    weights : quadrature weights

    Note: Currently, this function only works for 3D mesh
    """

    dtype = detJ_f.dtype
    x_dofs, x_g = mesh

    nq = weights.size  # Number of quadrature points

    # Map of the hexahedron reference facet Jacobian
    hex_reference_facet_jacobian = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
            [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]],
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        ],
        dtype=dtype,
    )

    for i, (cell, local_facet) in enumerate(boundary_data):
        coord_dofs = x_g[x_dofs[cell]]

        for q in range(nq):
            dphi = dphi_f[local_facet]

            J_cell = dphi[:, q, :] @ coord_dofs[:, :]

            J_facet = J_cell.T @ hex_reference_facet_jacobian[local_facet]

            detJ = np.linalg.norm(np.cross(J_facet[:, 0], J_facet[:, 1]))

            detJ_f[i, q] = detJ * weights[q]


@numba.njit(fastmath=True)
def compute_scaled_jacobian_determinant(
        detJ: npt.NDArray[np.floating],
        mesh: tuple[npt.NDArray[np.int32], npt.NDArray[np.floating]],
        num_cell: int,
        dphi: npt.NDArray[np.floating],
        weights: npt.NDArray[np.floating],
):
    """
    Compute the determinant of the Jacobian and scaled it with the
    quadrature weights.

    detJ = w_{q} |J_{q}|

    Parameters
    ----------
    detJ : array for the output
    mesh : mesh topology and geometry
    num_cell : number of cells in the mesh
    dphi : derivatives of the basis functions
    weights : quadrature weights

    Note: Currently, this function only works for 3D mesh
    """

    x_dofs, x_g = mesh

    nq = weights.size  # Number of quadrature points

    # Compute the scaled Jacobian determinant
    for cell in range(num_cell):
        coord_dofs = x_g[x_dofs[cell]]

        for q in range(nq):
            J_ = dphi[:, q, :] @ coord_dofs[:, :]

            detJ[cell, q] = np.fabs(np.linalg.det(J_)) * weights[q]


@numba.njit(fastmath=True)
def compute_scaled_geometrical_factor(
        G: npt.NDArray[np.floating],
        mesh: tuple[npt.NDArray[np.int32], npt.NDArray[np.floating]],
        num_cell: int,
        dphi: npt.NDArray[np.floating],
        weights: npt.NDArray[np.floating],
):
    """
    Compute the scaled geometrical factor given by

    G = w_{q} J_{q}^{-T}J_{q}^{-1} |J_{q}|

    Parameters
    ----------
    G : array for the output
    mesh : mesh topology and geometry
    num_cell : number of cells in the mesh
    dphi : derivatives of the basis functions
    weights : quadrature weights

    Note: Currently, this function only works for 3D mesh
    """

    x_dofs, x_g = mesh

    nq = weights.size  # Number of quadrature points

    # Compute the scaled geometrical factor
    for cell in range(num_cell):
        coord_dofs = x_g[x_dofs[cell]]

        for q in range(nq):
            # Compute the Jacobian
            J_ = dphi[:, q, :] @ coord_dofs[:, :]

            # Compute the geometrical factor
            G_ = np.linalg.inv(J_).T @ np.linalg.inv(J_)

            # Compute the scaled Jacobian determinant
            sdetJ = np.fabs(np.linalg.det(J_)) * weights[q]

            # Only store the upper triangular values since G is symmetric
            G[cell, q, 0] = sdetJ * G_[0, 0]
            G[cell, q, 1] = sdetJ * G_[0, 1]
            G[cell, q, 2] = sdetJ * G_[0, 2]
            G[cell, q, 3] = sdetJ * G_[1, 1]
            G[cell, q, 4] = sdetJ * G_[1, 2]
            G[cell, q, 5] = sdetJ * G_[2, 2]
