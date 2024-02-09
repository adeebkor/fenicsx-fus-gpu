from time import perf_counter_ns

import numpy as np
from mpi4py import MPI

import basix
import basix.ufl
from dolfinx.fem import assemble_vector, functionspace, form, Function
from dolfinx.mesh import create_box, CellType
from dolfinx.io import XDMFFile
from ufl import inner, dx, TestFunction

from precompute import compute_scaled_jacobian_determinant
from operators import mass_operator

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
    (N, N, N), cell_type=CellType.hexahedron, dtype=np.float64)

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
u0 = Function(V, dtype=np.float64)  # Input function
u = u0.x.array
u[:] = 1.0
b0 = Function(V, dtype=np.float64)  # Output function
b = b0.x.array
b[:] = 0.0

# Prepare input data to kernels
x_dofs = mesh.geometry.dofmap
x_g = mesh.geometry.x
cell_type = mesh.basix_cell()

tdim = mesh.topology.dim
gdim = mesh.geometry.dim
num_cells = mesh.topology.index_map(tdim).size_local
coeffs = np.ones(num_cells, dtype=np.float64)

pts, wts = basix.quadrature.make_quadrature(
    basix.CellType.hexahedron, Q[P], basix.QuadratureType.gll)

gelement = basix.create_element(
    basix.ElementFamily.P, mesh.basix_cell(), 1, dtype=np.float64)
gtable = gelement.tabulate(1, pts)
dphi = gtable[1:, :, :, 0]

nq = wts.size
detJ = np.zeros((num_cells, nq), dtype=np.float64)

compute_scaled_jacobian_determinant(
    detJ, (x_dofs, x_g), (tdim, gdim), num_cells, dphi, wts)

# Initial called to JIT compile function
mass_operator(u, coeffs, b, detJ, dofmap, tp_order)

# Use DOLFINx assembler for comparison
md = {"quadrature_rule": "GLL", "quadrature_degree": Q[P]}

v = TestFunction(V)
u0.x.array[:] = 1.0
a_dolfinx = form(inner(u0, v) * dx(metadata=md), dtype=np.float64)

b_dolfinx = assemble_vector(a_dolfinx)

np.testing.assert_allclose(b[:], b_dolfinx.array[:], atol=1e-9)

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
