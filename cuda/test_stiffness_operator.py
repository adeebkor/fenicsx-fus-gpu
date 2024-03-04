#
# .. _test_stiffness_operator:
#
# Test whether the stiffness operator is working correctly by comparing the
# output with DOLFINx.
# =========================================================================
# Copyright (C) 2024 Adeeb Arif Kor

import numpy as np
from mpi4py import MPI

import numba
import numba.cuda as cuda

import basix
import basix.ufl
from dolfinx.fem import assemble_vector, functionspace, form, Function
from dolfinx.mesh import create_box, CellType, GhostMode
from ufl import inner, grad, dx, TestFunction

from precompute import compute_scaled_geometrical_factor
from operators import stiffness_operator

float_type = np.float64

if isinstance(float_type, np.float64):
  tol = 1e-12
else:
  tol = 1e-6

# Check if CUDA is available
if cuda.is_available():
    print("CUDA is available")

cuda.detect()


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
  (N, N, N), cell_type=CellType.hexahedron, 
  ghost_mode=GhostMode.none,
  dtype=float_type
)

# Mesh geometry data
x_dofs = mesh.geometry.dofmap
x_g = mesh.geometry.x
cell_type = mesh.basix_cell()

# Uncomment below if we would like to test unstructured mesh
mesh.geometry.x[:, :] += np.random.uniform(
  -0.01, 0.01, (mesh.geometry.x.shape[0], 3))

# Tensor product element
basix_element = basix.create_tp_element(
  basix.ElementFamily.P, mesh.basix_cell(), P, basix.LagrangeVariant.gll_warped
)
element = basix.ufl._BasixElement(basix_element)

# Create function space
V = functionspace(mesh, element)
dofmap = V.dofmap.list

# Create function
u0 = Function(V, dtype=float_type)
u0.interpolate(lambda x: 100 * np.sin(2*np.pi*x[0]) * np.cos(3*np.pi*x[1])
               * np.sin(4*np.pi*x[2]))
u = u0.x.array

# Output for stiffness operator
b0 = Function(V, dtype=float_type)
b = b0.x.array
b[:] = 0.0

tdim = mesh.topology.dim
gdim = mesh.geometry.dim
num_cells = mesh.topology.index_map(tdim).size_local
coeffs = - 1.0 * np.ones(num_cells, dtype=float_type)

pts, wts = basix.quadrature.make_quadrature(
  basix.CellType.hexahedron, Q[P], basix.QuadratureType.gll
)

gelement = basix.create_element(
  basix.ElementFamily.P, mesh.basix_cell(), 1, dtype=float_type
)
gtable = gelement.tabulate(1, pts)
dphi = gtable[1:, :, :, 0]

nq = wts.size
G = np.zeros((num_cells, nq, (3*(gdim-1))), dtype=float_type)

compute_scaled_geometrical_factor(G, (x_dofs, x_g), num_cells, dphi, wts)

# Create 1D element for sum factorisation
element_1D = basix.create_element(
  basix.ElementFamily.P, basix.CellType.interval, P,
  basix.LagrangeVariant.gll_warped, dtype=float_type)
pts_1D, wts_1D = basix.quadrature.make_quadrature(
    basix.CellType.interval, Q[P], basix.QuadratureType.gll)
pts_1D, wts_1D = pts_1D.astype(float_type), wts_1D.astype(float_type)

table_1D = element_1D.tabulate(1, pts_1D)
dphi_1D = table_1D[1, :, :, 0]
nd = dphi_1D.shape[1]

# set the number of threads in a block
threadsperblock = (nd, nd, nd)
num_blocks = num_cells

# Allocate memory on the device
G_d = cuda.to_device(G)
coeffs_d = cuda.to_device(coeffs)
dofmap_d = cuda.to_device(dofmap)
u_d = cuda.to_device(u)
b_d = cuda.to_device(b)
dphi_1D_d = cuda.to_device(dphi_1D)
nd_d = cuda.to_device(nd)

# Call the stiffness operator function
print("Running operator!")
stiffness_operator[num_blocks, threadsperblock](u_d, coeffs_d, b_d, G_d, dofmap_d, dphi_1D_d)
print("Done!")

# Copy the result back to the host
b_d.copy_to_host(b)

# Use DOLFINx assembler for comparison
md = {"quadrature_rule": "GLL", "quadrature_degree": Q[P]}

v = TestFunction(V)
a_dolfinx = form(- inner(grad(u0), grad(v)) * dx(metadata=md), dtype=float_type)

b_dolfinx = assemble_vector(a_dolfinx)

# Check the difference between the vectors
print(
    "Euclidean difference: ",
    np.linalg.norm(b - b_dolfinx.array) / np.linalg.norm(b_dolfinx.array),
)

# Test the closeness between the vectors
np.testing.assert_allclose(b[:], b_dolfinx.array[:], atol=tol)