from time import perf_counter_ns

import numpy as np
import numba
from mpi4py import MPI

import basix
import basix.ufl
from dolfinx.fem import functionspace, Function
from dolfinx.mesh import create_box, CellType

from precompute import compute_scaled_jacobian_determinant


@numba.njit(fastmath=True)
def mass_operator(x, coeffs, y, detJ, dofmap, tp_order):
    nc = coeffs.size

    for c in range(nc):
        # Pack coefficients
        x_ = x[dofmap[c][tp_order]]

        # Apply transform
        x_ *= detJ[c] * coeffs[c]

        # Add contributions
        y[dofmap[c][tp_order]] += x_


if __name__ == "__main__":

    P = 5  # Basis function order
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

    mesh = create_box(
        MPI.COMM_WORLD, ((0., 0., 0.), (1., 1., 1.)),
        (2, 2, 2), cell_type=CellType.hexahedron)

    # Tensor product representation
    element = basix.ufl.element(
        basix.ElementFamily.P, mesh.basix_cell(), P,
        basix.LagrangeVariant.gll_warped
    )
    tp_order = np.array(element.get_tensor_product_representation()[0][1])

    # Create function space
    V = functionspace(mesh, element)
    dofmap = V.dofmap.list

    # Create function
    u0 = Function(V)  # Input function
    u = u0.x.array.astype(np.float32)
    u[:] = 1.0
    b0 = Function(V)  # Output function
    b = b0.x.array.astype(np.float32)
    b[:] = 0.0

    # Prepare input data to kernels
    x_dofs = mesh.geometry.dofmap
    x_g = mesh.geometry.x
    cell_type = mesh.basix_cell()

    tdim = mesh.topology.dim
    gdim = mesh.geometry.dim
    num_cells = mesh.topology.index_map(tdim).size_local
    coeffs = np.ones(num_cells)

    pts, wts = basix.quadrature.make_quadrature(
        basix.CellType.hexahedron, Q[P], basix.QuadratureType.gll
    )

    gelement = basix.create_element(
        basix.ElementFamily.P, mesh.basix_cell(), 1)
    gtable = gelement.tabulate(1, pts)
    dphi = gtable[1:, :, :, 0]

    nq = wts.size
    detJ = np.zeros((num_cells, nq), dtype=np.float32)

    compute_scaled_jacobian_determinant(
        detJ, (x_dofs, x_g), (tdim, gdim), num_cells, dphi, wts)

    # Initial called to JIT compile function
    mass_operator(u, coeffs, b, detJ, dofmap, tp_order)

    # Timing mass operator function
    timing_mass_operator = np.empty(10)
    for i in range(timing_mass_operator.size):
        b[:] = 0.0
        tic = perf_counter_ns()
        mass_operator(u, coeffs, b, detJ, dofmap, tp_order)
        toc = perf_counter_ns()
        timing_mass_operator[i] = toc - tic

    timing_mass_operator *= 1e-3

    print(
        f"Elapsed time (mass operator): "
        f"{timing_mass_operator.mean():.0f} ± "
        f"{timing_mass_operator.std():.0f} μs")
