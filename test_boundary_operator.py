from time import perf_counter_ns

import numpy as np
from mpi4py import MPI

import basix
import basix.ufl
from dolfinx.fem import assemble_vector, functionspace, form, locate_dofs_topological, Function
from dolfinx.mesh import create_box, locate_entities_boundary, CellType
from dolfinx.io import XDMFFile
from ufl import inner, ds, TestFunction

from precompute import compute_boundary_facets_scaled_jacobian_determinant
from operators import boundary_facet_operator

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

# Find degrees-of-freedom on the boundary
boundary_facets = locate_entities_boundary(
    mesh, mesh.topology.dim-1, lambda x: np.full(x.shape[1], True, dtype=bool))

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
coeffs = np.ones(num_cells, dtype=float_type)

gelement = basix.create_element(
    basix.ElementFamily.P, mesh.basix_cell(), 1, dtype=float_type)

# Find the cells that contains the boundary facets
cell_to_facet_map = mesh.topology.connectivity(
    tdim, tdim-1)
facet_to_cell_map = mesh.topology.connectivity(
    tdim-1, tdim)

boundary_facet_cell = {}

for boundary_facet in boundary_facets:
    boundary_facet_cell[boundary_facet] = facet_to_cell_map.links(boundary_facet)[0]

# Create an array that contains all the boundary facet data
boundary_data = np.zeros((boundary_facets.size, 2), dtype=np.int32)

for i, (facet, cell) in enumerate(boundary_facet_cell.items()):
    facets = cell_to_facet_map.links(cell) 
    local_facet = np.where(facet == facets)
    boundary_data[i, 0] = cell
    boundary_data[i, 1] = local_facet[0][0]

# Get the local DOF on the facets
local_facet_dof = np.array(basix_element.entity_closure_dofs[2], dtype=np.int32)

# Compute the determinant of the Jacobian on the boundary facets
num_boundary_cells = np.unique(boundary_data[:, 1]).size
num_boundary_facets = boundary_facets.size

# Get the quadrature points and weights on the reference facet
pts_f, wts_f = basix.quadrature.make_quadrature(
    basix.CellType.quadrilateral, Q[P], basix.QuadratureType.gll)

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
detJ_f = np.zeros((boundary_data.shape[0], nq_f), 
                  dtype=float_type)

compute_boundary_facets_scaled_jacobian_determinant(
    detJ_f, (x_dofs, x_g), boundary_data, dphi_f, wts_f)

# Compute the boundary operator
boundary_facet_operator(
    u, coeffs, b, detJ_f, dofmap, boundary_data, local_facet_dof)

# Use DOLFINx assembler for comparison
md = {"quadrature_rule": "GLL", "quadrature_degree": Q[P]}

v = TestFunction(V)
u0.x.array[:] = 1.0
a_dolfinx = form(inner(u0, v) * ds(metadata=md), dtype=float_type)

b_dolfinx = assemble_vector(a_dolfinx)

# Check the difference between the vectors
print("Euclidean difference: ", 
      np.linalg.norm(b - b_dolfinx.array) / np.linalg.norm(b_dolfinx.array))

# Test the closeness between the vectors
np.testing.assert_allclose(b, b_dolfinx.array, rtol=tol, atol=tol)
