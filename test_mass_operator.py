#
# .. _test_mass_operator:
#
# Test whether the mass operator is working correctly by comparing the output
# with DOLFINx.
# ===========================================================================
# Copyright (C) 2024 Adeeb Arif Kor

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

float_type = np.float64

if isinstance(float_type, np.float64):
    tol = 1e-12
else:
    tol = 1e-6

P = 4  # Basis function order
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
    (N, N, N), cell_type=CellType.hexahedron, dtype=float_type)

# Mesh geometry data
x_dofs = mesh.geometry.dofmap
x_g = mesh.geometry.x
cell_type = mesh.basix_cell()

# Uncomment below if we would like to test unstructured mesh
mesh.geometry.x[:, :] += np.random.uniform(
    -0.01, 0.01, (mesh.geometry.x.shape[0], 3))

# Tensor product element
basix_element = basix.create_tp_element(
    basix.ElementFamily.P, mesh.basix_cell(), P,
    basix.LagrangeVariant.gll_warped
)
element = basix.ufl._BasixElement(basix_element)

# Create function space
V = functionspace(mesh, element)
dofmap = V.dofmap.list

# Create function
u0 = Function(V, dtype=float_type)  # Input function
u = u0.x.array
u[:] = 1.0
b0 = Function(V, dtype=float_type)  # Output function
b = b0.x.array
b[:] = 0.0

tdim = mesh.topology.dim
gdim = mesh.geometry.dim
num_cells = mesh.topology.index_map(tdim).size_local
coeffs = np.ones(num_cells, dtype=float_type)

pts, wts = basix.quadrature.make_quadrature(
    basix.CellType.hexahedron, Q[P], basix.QuadratureType.gll)

gelement = basix.create_element(
    basix.ElementFamily.P, mesh.basix_cell(), 1, dtype=float_type)
gtable = gelement.tabulate(1, pts)
dphi = gtable[1:, :, :, 0]

nq = wts.size
detJ = np.zeros((num_cells, nq), dtype=float_type)

compute_scaled_jacobian_determinant(detJ, (x_dofs, x_g), num_cells, dphi, wts)

# Initial called to JIT compile function
mass_operator(u, coeffs, b, detJ, dofmap)

# Use DOLFINx assembler for comparison
md = {"quadrature_rule": "GLL", "quadrature_degree": Q[P]}

v = TestFunction(V)
u0.x.array[:] = 1.0
a_dolfinx = form(inner(u0, v) * dx(metadata=md), dtype=float_type)

b_dolfinx = assemble_vector(a_dolfinx)

# Check the difference between the vectors
print("Euclidean difference: ", 
      np.linalg.norm(b - b_dolfinx.array) / np.linalg.norm(b_dolfinx.array))

# Test the closeness between the vectors
np.testing.assert_allclose(b[:], b_dolfinx.array[:], atol=tol)

# Timing mass operator function
timing_mass_operator = np.empty(10)
for i in range(timing_mass_operator.size):
    b[:] = 0.0
    tic = perf_counter_ns()
    mass_operator(u, coeffs, b, detJ, dofmap)
    toc = perf_counter_ns()
    timing_mass_operator[i] = toc - tic

timing_mass_operator *= 1e-3

print(
    f"Elapsed time (mass operator): "
    f"{timing_mass_operator.mean():.0f} ± "
    f"{timing_mass_operator.std():.0f} μs")
