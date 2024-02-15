from time import perf_counter_ns

import numpy as np
from mpi4py import MPI

import basix
import basix.ufl
from dolfinx.fem import assemble_vector, functionspace, form, locate_dofs_topological, Function
from dolfinx.mesh import create_box, locate_entities_boundary, CellType
from dolfinx.io import XDMFFile
from ufl import inner, ds, TestFunction

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

float_type = np.float64

N = 8
mesh = create_box(
    MPI.COMM_WORLD, ((0., 0., 0.), (1., 1., 1.)),
    (N, N, N), cell_type=CellType.hexahedron, dtype=float_type)

# Mesh geometry data
x_dofs = mesh.geometry.dofmap
x_g = mesh.geometry.x
cell_type = mesh.basix_cell()

# If we would like to test unstructured mesh, uncomment below
# mesh.geometry.x[:, :] += np.random.uniform(
#     -0.01, 0.01, (mesh.geometry.x.shape[0], 3))

# Tensor product representation
element = basix.ufl.element(
    basix.ElementFamily.P, mesh.basix_cell(), P,
    basix.LagrangeVariant.gll_warped
)
tp_order = np.array(element.get_tensor_product_representation()[0][1])

element_q = basix.ufl.element(
    basix.ElementFamily.P, basix.CellType.quadrilateral, P,
    basix.LagrangeVariant.gll_warped
)
tp_order_q = np.array(element_q.get_tensor_product_representation()[0][1])

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
    basix.ElementFamily.P, mesh.basix_cell(), 1)

# Find the cells that contains the boundary facets
cell_to_facet_map = mesh.topology.connectivity(
    tdim, tdim-1)
facet_to_cell_map = mesh.topology.connectivity(
    tdim-1, tdim)

boundary_facet_cell = {}

for boundary_facet in boundary_facets:
    boundary_facet_cell[boundary_facet] = facet_to_cell_map.links(boundary_facet)[0]

# Create an array that contains all the boundary facet data
facet_data = np.zeros((boundary_facets.size, 3), dtype=np.int32)

for i, (facet, cell) in enumerate(boundary_facet_cell.items()):
    facets = cell_to_facet_map.links(cell) 
    local_facet = np.where(facet == facets)
    facet_data[i, 0] = local_facet[0][0]
    facet_data[i, 1] = cell
    facet_data[i, 2] = facet

# Get the local DOF on the facets
element = V.element.basix_element
local_facet_dof = np.array(element.entity_closure_dofs[2], dtype=np.int32)

# Map of the hexahedron reference facet Jacobian
hexahedron_reference_facet_jacobian = np.array(
    [[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
     [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]],
     [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
     [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
     [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]],
     [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]])

# Compute the determinant of the Jacobian on the boundary facets
num_boundary_cells = np.unique(facet_data[:, 1]).size
num_boundary_facets = boundary_facets.size

# Get the quadrature points and weights on the reference facet
pts_f, wts_f = basix.quadrature.make_quadrature(
    basix.CellType.quadrilateral, Q[P], basix.QuadratureType.gll)

nq_f = wts_f.size

# Create the evaluation points on the facets of the reference hexahedron
pts_0 = pts_f[:, 0]
pts_1 = pts_f[:, 1]

pts_f = np.zeros((6, nq_f, 3), dtype=float_type)
pts_f[0, :, :] = np.c_[pts_0, pts_1, np.zeros(nq_f, dtype=float_type)]
pts_f[1, :, :] = np.c_[pts_0, np.zeros(nq_f, dtype=float_type), pts_1]
pts_f[2, :, :] = np.c_[np.zeros(nq_f, dtype=float_type), pts_0, pts_1]
pts_f[3, :, :] = np.c_[np.ones(nq_f, dtype=float_type), pts_0, pts_1]
pts_f[4, :, :] = np.c_[pts_0, np.ones(nq_f, dtype=float_type), pts_1]
pts_f[5, :, :] = np.c_[pts_0, pts_1, np.ones(nq_f, dtype=float_type)]

# Evaluate the derivatives on the facets of the reference hexahedron
dphi_f = np.zeros((6, 3, nq_f, 8), dtype=float_type)

for f in range(6):
    gtable_f = gelement.tabulate(1, pts_f[f, :, :]).astype(float_type)
    dphi_f[f, :, :, :] = gtable_f[1:, :, :, 0]

dphi_f[abs(dphi_f) < 1e-10] = 0.0

# Compute the determinant of the Jacobian on the boundary facets
detJ_f = np.zeros((facet_data.shape[0], nq_f), 
                  dtype=float_type)

for i, (local_facet, cell, facet) in enumerate(facet_data):
    coord_dofs = x_g[x_dofs[cell]]

    for q in range(nq_f):
        dphi_ = dphi_f[local_facet]

        J_cell = dphi_[:, q, :] @ coord_dofs[:, :gdim]

        J_facet = J_cell @ hexahedron_reference_facet_jacobian[local_facet]

        detJ_f[i, q] = np.linalg.norm(
            np.cross(J_facet[:, 0], J_facet[:, 1])) * wts_f[q]

# Compute the boundary operator
for i, (local_facet, cell, facet) in enumerate(facet_data):
    x_ = u[dofmap[cell][local_facet_dof[local_facet]][tp_order_q]]

    x_ *= detJ_f[i, :] * coeffs[cell]

    b[dofmap[cell][local_facet_dof[local_facet]][tp_order_q]] += x_

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
np.testing.assert_allclose(b, b_dolfinx.array)
