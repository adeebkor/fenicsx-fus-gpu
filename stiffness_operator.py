from time import perf_counter_ns

import numpy as np
import numba
from mpi4py import MPI

import basix
import basix.ufl
from dolfinx.fem import functionspace, Function
from dolfinx.mesh import create_box, CellType

from precompute import compute_scaled_geometrical_factor


def stiffness_transform(Gc, coeff, fw0, fw1, fw2, nd):
    for iq in range(nd*nd*nd):
        G_ = Gc[iq]
        w0 = fw0[iq]
        w1 = fw1[iq]
        w2 = fw2[iq]

        fw0[iq] = coeff * (G_[0] * w0 + G_[1] * w1 + G_[2] * w2)
        fw1[iq] = coeff * (G_[1] * w0 + G_[3] * w1 + G_[4] * w2)
        fw2[iq] = coeff * (G_[2] * w0 + G_[4] * w1 + G_[5] * w2)


def stiffness_operator(x, coeffs, y, G, dofmap, dphi, tp_order):
    nc = coeffs.size
    nd = dphi.shape[1]

    for c in range(nc):
        # Pack coefficients
        x_ = x[dofmap[c][tp_order]].reshape(nd, nd, nd)

        # Apply contraction in the x-direction
        T1 = np.einsum("qi,ijk->qjk", dphi_1D, x_)

        # Apply contraction in the y-direction
        T2 = np.einsum("qj,ijk->iqk", dphi_1D, x_)

        # Apply contraction in the z-direction
        T3 = np.einsum("qk,ijk->ijq", dphi_1D, x_)

        # Apply transform"
        stiffness_transform(G[c], coeffs[c],
                            T1.reshape(nd*nd*nd), 
                            T2.reshape(nd*nd*nd), 
                            T3.reshape(nd*nd*nd), nd)

        # Apply contraction in the x-direction
        T1 = T1.reshape(nd, nd, nd)
        y0_ = np.einsum("qi,qjk->ijk", dphi_1D, T1)

        # Apply contraction in the y-direction
        T2 = T2.reshape(nd, nd, nd)
        y1_ = np.einsum("qj,iqk->ijk", dphi_1D, T2)

        # Apply contraction in the z-direction
        T3 = T3.reshape(nd, nd, nd)
        y2_ = np.einsum("qk,ijq->ijk", dphi_1D, T3)

        # Add contributions
        y_ = y0_ + y1_ + y2_
        y[dofmap[c][tp_order]] += y_.reshape(nd*nd*nd)


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

    N = 2
    mesh = create_box(
        MPI.COMM_WORLD, ((0., 0., 0.), (1., 1., 1.)),
        (N, N, N), cell_type=CellType.hexahedron)

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
    u0 = Function(V)
    u0.interpolate(lambda x: np.sin(x[0]) * np.cos(np.pi * x[1]))
    u = u0.x.array.astype(np.float32)
    b0 = Function(V)
    b = b0.x.array.astype(np.float32)
    b[:] = 0.0

    # Prepare input data to kernels
    x_dofs = mesh.geometry.dofmap
    x_g = mesh.geometry.x
    cell_type = mesh.basix_cell()

    tdim = mesh.topology.dim
    gdim = mesh.geometry.dim
    num_cells = mesh.topology.index_map(tdim).size_local
    coeffs = 1.0 * np.ones(num_cells)

    pts, wts = basix.quadrature.make_quadrature(
        basix.CellType.hexahedron, Q[P], basix.QuadratureType.gll)

    gelement = basix.create_element(
        basix.ElementFamily.P, mesh.basix_cell(), 1)
    gtable = gelement.tabulate(1, pts)
    dphi = gtable[1:, :, :, 0]

    nq = wts.size
    G = np.empty((num_cells, nq, (3*(gdim-1))), dtype=np.float32)

    compute_scaled_geometrical_factor(
        G, (x_dofs, x_g), (tdim, gdim), num_cells, dphi, wts)
    
    # Create 1D element for sum factorisation
    element_1D = basix.create_element(
        basix.ElementFamily.P, basix.CellType.interval, P,
        basix.LagrangeVariant.gll_warped)
    pts_1D, wts_1D = basix.quadrature.make_quadrature(
        basix.CellType.interval, Q[P], basix.QuadratureType.gll
    )
    table_1D = element_1D.tabulate(1, pts_1D)
    dphi_1D = table_1D[1, :, :, 0]

    # Initial called to JIT compile function
    stiffness_operator(u, coeffs, b, G, dofmap, dphi_1D, tp_order)

    # Timing stiffness operator function
    timing_stiffness_operator = np.empty(10)
    for i in range(timing_stiffness_operator.size):
        b[:] = 0.0
        tic = perf_counter_ns()
        stiffness_operator(u, coeffs, b, G, dofmap, dphi_1D, tp_order)
        toc = perf_counter_ns()
        timing_stiffness_operator[i] = toc - tic

    timing_stiffness_operator *= 1e-3

    print(
        f"Elapsed time (mass operator): "
        f"{timing_stiffness_operator.mean():.0f} ± "
        f"{timing_stiffness_operator.std():.0f} μs")

