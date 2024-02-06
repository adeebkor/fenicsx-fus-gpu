from time import perf_counter_ns

import numpy as np
import numba
from mpi4py import MPI

import basix
from dolfinx.mesh import create_unit_cube, CellType


@numba.njit(fastmath=True)
def compute_scaled_jacobian_determinant(detJ, mesh, dim, num_cell, dphi,
                                        weights):
    """
    Compute the determinant of the Jacobian and scaled it with the
    quadrature weights.

    detJ = w_{q} |J_{q}|
    """
    x_dofs, x_g = mesh

    tdim, gdim = dim

    nc = num_cell  # Number of cells
    nq = weights.size  # Number of quadrature points

    J_ = np.zeros((tdim, gdim), dtype=np.float32)

    # Compute the scaled Jacobian determinant
    for c in range(nc):
        coord_dofs = x_g[x_dofs[c]]

        for q in range(nq):
            J_ = dphi[:, q, :] @ coord_dofs[:, :gdim]

            detJ[c, q] = np.fabs(np.linalg.det(J_)) * weights[q]


@numba.njit(fastmath=True)
def compute_scaled_geometrical_factor(G, mesh, dim, num_cell, dphi, weights):
    """
    Compute the geometrical factor given by
    G = w_{q} J_{q}^{-T}J_{q}^{-1} |J_{q}|
    """

    x_dofs, x_g = mesh

    tdim, gdim = dim

    nc = num_cell  # Number of cells
    nq = weights.size  # Number of quadrature points

    J_ = np.zeros((tdim, gdim), dtype=np.float32)
    G_ = np.zeros((tdim, gdim), dtype=np.float32)

    # Compute the scaled geometrical factor
    for c in range(nc):
        coord_dofs = x_g[x_dofs[c]]

        for q in range(nq):
            # Compute the Jacobian
            J_ = dphi[:, q, :] @ coord_dofs[:, :gdim]

            # Compute the geometrical factor
            G_ = np.linalg.inv(J_) @ np.linalg.inv(J_).T

            # Compute the scaled Jacobian determinant
            sdetJ = np.fabs(np.linalg.det(J_)) * weights[q]

            # Only store the upper triangular values since G is symmetric
            if gdim == 2:
                G[c, q, 0] = sdetJ * G_[0, 0]
                G[c, q, 1] = sdetJ * G_[0, 1]
                G[c, q, 2] = sdetJ * G_[1, 1]
            elif gdim == 3:
                G[c, q, 0] = sdetJ * G_[0, 0]
                G[c, q, 1] = sdetJ * G_[0, 1]
                G[c, q, 2] = sdetJ * G_[0, 2]
                G[c, q, 3] = sdetJ * G_[1, 1]
                G[c, q, 4] = sdetJ * G_[1, 2]
                G[c, q, 5] = sdetJ * G_[2, 2]


if __name__ == "__main__":
    P = 3  # Basis function order
    Q = {
        2: 3,
        3: 4,
        4: 6,
        5: 8,
        6: 10,
        7: 12,
        8: 14,
        9: 16,
        10: 18,
    }  # Quadrature degree

    pts, wts = basix.quadrature.make_quadrature(
        basix.CellType.hexahedron, Q[P], basix.QuadratureType.gll
    )

    mesh = create_unit_cube(
        MPI.COMM_WORLD, 8, 8, 8, cell_type=CellType.hexahedron)

    x_dofs = mesh.geometry.dofmap
    x_g = mesh.geometry.x
    cell_type = mesh.basix_cell()

    tdim = mesh.topology.dim
    gdim = mesh.geometry.dim
    num_cell = mesh.topology.index_map(tdim).size_local

    element = basix.create_element(basix.ElementFamily.P, mesh.basix_cell(), 1)
    table = element.tabulate(1, pts)
    dphi = table[1:, :, :, 0]

    detJ = np.zeros((num_cell, wts.size), dtype=np.float32)

    # Time the implementations

    # Initial called to JIT compile function
    compute_scaled_jacobian_determinant(
        detJ, (x_dofs, x_g), (tdim, gdim), num_cell, dphi, wts
    )

    # Time scaled Jacobian determinant function
    timing_jacobian_det = np.empty(10)
    for i in range(timing_jacobian_det.size):
        tic = perf_counter_ns()
        compute_scaled_jacobian_determinant(
            detJ, (x_dofs, x_g), (tdim, gdim), num_cell, dphi, wts
        )
        toc = perf_counter_ns()
        timing_jacobian_det[i] = toc - tic

    timing_jacobian_det *= 1e-3

    print(
        f"Elapsed time (scaled Jacobian determinant): "
        f"{timing_jacobian_det.mean():.0f} ± "
        f"{timing_jacobian_det.std():.0f} μs"
    )

    G = np.zeros((num_cell, wts.size, 3 * (gdim - 1)), dtype=np.float32)

    # Initial called to JIT compile function
    compute_scaled_geometrical_factor(
        G, (x_dofs, x_g), (tdim, gdim), num_cell, dphi, wts
    )

    # Time scaled Jacobian determinant function
    timing_geometrical_fac = np.empty(10)
    for i in range(timing_geometrical_fac.size):
        tic = perf_counter_ns()
        compute_scaled_geometrical_factor(
            G, (x_dofs, x_g), (tdim, gdim), num_cell, dphi, wts
        )
        toc = perf_counter_ns()
        timing_geometrical_fac[i] = toc - tic

    timing_geometrical_fac *= 1e-3

    print(
        f"Elapsed time (scaled geometrical factor): "
        f"{timing_geometrical_fac.mean():.0f} ± "
        f"{timing_geometrical_fac.std():.0f} μs"
    )
