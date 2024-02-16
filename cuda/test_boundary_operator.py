#
# .. _test_boundary_operator:
#
# Test whether the boundary operator is working correctly by comparing the
# output with DOLFINx.
# ===========================================================================
# Copyright (C) 2024 Igor A. Baratta and Adeeb Arif Kor

from time import perf_counter_ns

import numpy as np
from mpi4py import MPI

import basix
import basix.ufl
from dolfinx.fem import assemble_vector, functionspace, form, Function
from dolfinx.mesh import create_box, locate_entities_boundary, CellType
from ufl import inner, ds, TestFunction

import dolfinx.io as io

from precompute import compute_boundary_facets_scaled_jacobian_determinant
from operators import mass_operator
from utils import facet_integration_domain

import numba
import numba.cuda as cuda

# Check if CUDA is available
if cuda.is_available():
    print("CUDA is available")

cuda.detect()


float_type = np.float32

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

N = 10
mesh = create_box(
    MPI.COMM_WORLD,
    ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    (N, N, N),
    cell_type=CellType.hexahedron,
    dtype=float_type,
)

topology = mesh.topology
tdim = topology.dim

# Mesh geometry data
x_dofs = mesh.geometry.dofmap
x_g = mesh.geometry.x
cell_type = mesh.basix_cell()

# Uncomment below if we would like to test unstructured mesh
mesh.geometry.x[:, :] += np.random.uniform(-0.01, 0.01, (mesh.geometry.x.shape[0], 3))

with io.XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "w") as file:
    file.write_mesh(mesh)

family = basix.ElementFamily.P
variant = basix.LagrangeVariant.gll_warped
# Tensor product element
basix_element = basix.create_tp_element(family, mesh.basix_cell(), P, variant)
element = basix.ufl._BasixElement(basix_element)  # basix ufl element

# Create function space
V = functionspace(mesh, element)
dofmap = V.dofmap.list

# Find degrees-of-freedom on the boundary
boundary_facets = locate_entities_boundary(
    mesh, tdim - 1, lambda x: np.full(x.shape[1], True, dtype=bool)
)

# Create function
u0 = Function(V, dtype=float_type)
u = u0.x.array
u[:] = 1.0
b0 = Function(V, dtype=float_type)
b = b0.x.array
b[:] = 0.0

tdim = mesh.topology.dim
gdim = mesh.geometry.dim
num_cells = mesh.topology.index_map(tdim).size_local

gelement = basix.create_element(family, mesh.basix_cell(), 1, dtype=float_type)

# Find the cells that contains the boundary facets
np.set_printoptions(threshold=np.inf)


boundary_data = facet_integration_domain(boundary_facets, mesh)

# Get the local DOF on the facets
local_facet_dof = np.array(basix_element.entity_closure_dofs[2], dtype=np.int32)

# Compute the determinant of the Jacobian on the boundary facets
# num_boundary_cells = np.unique(boundary_data[:, 1]).size
# num_boundary_facets = boundary_facets.size

# Get the quadrature points and weights on the reference facet
pts_f, wts_f = basix.quadrature.make_quadrature(
    basix.CellType.quadrilateral, Q[P], basix.QuadratureType.gll
)

nq_f = wts_f.size

# Create the evaluation points on the facets of the reference hexahedron
pts_0 = pts_f[:, 0]
pts_1 = pts_f[:, 1]

pts_f = np.zeros((6, nq_f, 3), dtype=float_type)
pts_f[0, :, :] = np.c_[pts_0, pts_1, np.zeros(nq_f, dtype=float_type)]  # z = 0
pts_f[1, :, :] = np.c_[pts_0, np.zeros(nq_f, dtype=float_type), pts_1]  # y = 0
pts_f[2, :, :] = np.c_[np.zeros(nq_f, dtype=float_type), pts_0, pts_1]  # x = 0
pts_f[3, :, :] = np.c_[np.ones(nq_f, dtype=float_type), pts_0, pts_1]  # x = 1
pts_f[4, :, :] = np.c_[pts_0, np.ones(nq_f, dtype=float_type), pts_1]  # y = 1
pts_f[5, :, :] = np.c_[pts_0, pts_1, np.ones(nq_f, dtype=float_type)]  # z = 1

# Evaluate the derivatives on the facets of the reference hexahedron
dphi_f = np.zeros((6, 3, nq_f, 8), dtype=float_type)

for f in range(6):
    gtable_f = gelement.tabulate(1, pts_f[f, :, :]).astype(float_type)
    dphi_f[f, :, :, :] = gtable_f[1:, :, :, 0]

# Compute the determinant of the Jacobian on the boundary facets
detJ_f = np.zeros((boundary_data.shape[0], nq_f), dtype=float_type)

compute_boundary_facets_scaled_jacobian_determinant(
    detJ_f, (x_dofs, x_g), boundary_data, dphi_f, wts_f
)

facet_dofmap = np.zeros(
    (boundary_data.shape[0], local_facet_dof.shape[1]), dtype=np.int32
)
for i, (cell, local_facet) in enumerate(boundary_data):
    facet_dofmap[i, :] = dofmap[cell][local_facet_dof[local_facet]]

# Compute the boundary operator
# boundary_facet_operator(u, coeffs, b, detJ_f, dofmap, boundary_data, local_facet_dof)

coeffs = np.ones(facet_dofmap.shape[0], dtype=float_type)  # material coefficients

# Set the number of threads in a block
threadsperblock = 128
numb_blocks = (facet_dofmap.size + (threadsperblock - 1)) // threadsperblock

# Allocate memory on the device
x_device = cuda.to_device(u)
facet_constants_device = cuda.to_device(coeffs)
y_device = cuda.to_device(b)
detJ_f_device = cuda.to_device(detJ_f)
facet_dofmap_device = cuda.to_device(facet_dofmap)

# Call the kernel
mass_operator[numb_blocks, threadsperblock](
    x_device, facet_constants_device, y_device, detJ_f_device, facet_dofmap_device
)

cuda.synchronize()

# Copy the result back to the host
y_device.copy_to_host(b)

# Use DOLFINx assembler for comparison
md = {"quadrature_rule": "GLL", "quadrature_degree": Q[P]}

v = TestFunction(V)
u0.x.array[:] = 1.0
a_dolfinx = form(inner(u0, v) * ds(metadata=md), dtype=float_type)

b_dolfinx = assemble_vector(a_dolfinx)

# Check the difference between the vectors
print(
    "Euclidean difference: ",
    np.linalg.norm(b - b_dolfinx.array) / np.linalg.norm(b_dolfinx.array),
)

# Test the closeness between the vectors
np.testing.assert_allclose(b, b_dolfinx.array, rtol=tol, atol=tol)
