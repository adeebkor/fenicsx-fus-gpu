from time import perf_counter_ns

import numpy as np
from mpi4py import MPI

import basix
import basix.ufl
from dolfinx.fem import assemble_vector, functionspace, form, Function
from dolfinx.mesh import create_box, CellType
from dolfinx.io import XDMFFile
from ufl import inner, grad, dx, TestFunction

from precompute import compute_scaled_geometrical_factor
from operators import stiffness_operator_einsum, stiffness_operator

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

N = 16
mesh = create_box(
    MPI.COMM_WORLD, ((0., 0., 0.), (1., 1., 1.)),
    (N, N, N), cell_type=CellType.hexahedron)

# # Read mesh
# with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "r") as fmesh:
#     mesh_name = "hex"
#     mesh = fmesh.read_mesh(name=f"{mesh_name}")
#     mt_cell = fmesh.read_meshtags(mesh, name=f"{mesh_name}_cells")
#     mesh.topology.create_connectivity(
#         mesh.topology.dim-1, mesh.topology.dim)

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
u0.interpolate(lambda x: 100 * np.sin(2*np.pi*x[0]) * np.cos(3*np.pi*x[1])
               * np.sin(4*np.pi*x[2]))
u_0 = u0.x.array.astype(np.float32)

# Output for stiffness operator
b0 = Function(V)
b_0 = b0.x.array.astype(np.float32)
b_0[:] = 0.0

# Output for stiffness operator (einsum)
b1 = Function(V)
b_1 = b1.x.array.astype(np.float32)
b_1[:] = 0.0

# Prepare input data to kernels
x_dofs = mesh.geometry.dofmap
x_g = mesh.geometry.x
cell_type = mesh.basix_cell()

tdim = mesh.topology.dim
gdim = mesh.geometry.dim
num_cells = mesh.topology.index_map(tdim).size_local
coeffs = - 1.0 * np.ones(num_cells)

pts, wts = basix.quadrature.make_quadrature(
    basix.CellType.hexahedron, Q[P], basix.QuadratureType.gll)

gelement = basix.create_element(
    basix.ElementFamily.P, mesh.basix_cell(), 1)
gtable = gelement.tabulate(1, pts)
dphi = gtable[1:, :, :, 0]

nq = wts.size
G = np.zeros((num_cells, nq, (3*(gdim-1))), dtype=np.float32)

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
nd = dphi_1D.shape[1]

stiffness_operator(u_0, coeffs, b_0, G, dofmap, tp_order, dphi_1D.flatten(),
                   nd)
stiffness_operator_einsum(u_0, coeffs, b_1, G, dofmap, tp_order, dphi_1D, nd)

# b_0[abs(b_0) < 1e-6] = 0.0
# b_1[abs(b_1) < 1e-6] = 0.0

np.testing.assert_allclose(b_0[:], b_1[:], atol=1e-5)

# Use DOLFINx assembler for comparison
md = {"quadrature_rule": "GLL", "quadrature_degree": Q[P]}

v = TestFunction(V)
a_dolfinx = form(- inner(grad(u0), grad(v)) * dx(metadata=md))

b_dolfinx = assemble_vector(a_dolfinx)

# b_dolfinx.array[abs(b_dolfinx.array) < 1e-6] = 0.0

np.testing.assert_allclose(b_0[:], b_dolfinx.array[:], atol=1e-5)
np.testing.assert_allclose(b_1[:], b_dolfinx.array[:], atol=1e-5)

# Timing stiffness operator function
timing_stiffness_operator = np.zeros(10)
for i in range(timing_stiffness_operator.size):
    b_0[:] = 0.0
    tic = perf_counter_ns()
    stiffness_operator(u_0, coeffs, b_0, G, dofmap, tp_order,
                       dphi_1D.flatten(), nd)
    toc = perf_counter_ns()
    timing_stiffness_operator[i] = toc - tic

timing_stiffness_operator *= 1e-3

print(
    f"Elapsed time (stiffness operator): "
    f"{timing_stiffness_operator.mean():.0f} ± "
    f"{timing_stiffness_operator.std():.0f} μs")

timing_stiffness_operator_einsum = np.zeros(10)
for i in range(timing_stiffness_operator_einsum.size):
    b_1[:] = 0.0
    tic = perf_counter_ns()
    stiffness_operator_einsum(u_0, coeffs, b_1, G, dofmap, tp_order,
                              dphi_1D, nd)
    toc = perf_counter_ns()
    timing_stiffness_operator_einsum[i] = toc - tic

timing_stiffness_operator_einsum *= 1e-3

print(
    f"Elapsed time (stiffness operator (einsum)): "
    f"{timing_stiffness_operator_einsum.mean():.0f} ± "
    f"{timing_stiffness_operator_einsum.std():.0f} μs")
