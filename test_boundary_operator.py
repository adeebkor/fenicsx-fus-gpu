from time import perf_counter_ns

import numpy as np
from mpi4py import MPI

import basix
import basix.ufl
from dolfinx.fem import assemble_vector, functionspace, form, locate_dofs_topological, Function
from dolfinx.mesh import create_box, locate_entities_boundary, CellType
from dolfinx.io import XDMFFile
from ufl import inner, ds, TestFunction

from precompute import compute_scaled_jacobian_determinant

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

# Tensor product representation
element = basix.ufl.element(
    basix.ElementFamily.P, mesh.basix_cell(), P,
    basix.LagrangeVariant.gll_warped
)
tp_order = np.array(element.get_tensor_product_representation()[0][1])

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

# Prepare input data to kernels
x_dofs = mesh.geometry.dofmap
x_g = mesh.geometry.x
cell_type = mesh.basix_cell()

tdim = mesh.topology.dim
gdim = mesh.geometry.dim
num_cells = mesh.topology.index_map(tdim).size_local
coeffs = np.ones(num_cells, dtype=float_type)

pts, wts = basix.quadrature.make_quadrature(
    basix.CellType.hexahedron, Q[P], basix.QuadratureType.gll)

gelement = basix.create_element(
    basix.ElementFamily.P, mesh.basix_cell(), 1)
gtable = gelement.tabulate(1, pts).astype(float_type)
dphi = gtable[1:, :, :, 0]

nq = wts.size
detJ = np.zeros((num_cells, nq), dtype=float_type)

compute_scaled_jacobian_determinant(
    detJ, (x_dofs, x_g), (tdim, gdim), num_cells, dphi, wts)

# Find the cells that contains the boundary facets
cell_to_facet_map = mesh.topology.connectivity(
    tdim, tdim-1)
facet_to_cell_map = mesh.topology.connectivity(
    tdim-1, tdim)

boundary_facet_cell = {}

for facet in boundary_facets:
    boundary_facet_cell[facet] = facet_to_cell_map.links(facet)[0]

local_facets = np.zeros((boundary_facets.size, 3), dtype=np.int32)

i = 0
for facet, cell in boundary_facet_cell.items():
    facets = cell_to_facet_map.links(cell) 
    local_facet = np.where(facet == facets)
    local_facets[i, 0] = local_facet[0][0]
    local_facets[i, 1] = cell
    local_facets[i, 2] = facet
    i += 1

element = V.element.basix_element

local_facet_dof = element.entity_closure_dofs[2]

hexahedron_reference_facet_jacobian = np.array(
    [[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
     [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]],
     [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
     [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
     [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]],
     [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]])


# Compute the determinant of the Jacobian on the boundary facets
num_boundary_cells = np.unique(local_facets[:, 1]).size
num_boundary_facets = boundary_facets.size

detJ_f = np.zeros((local_facets.shape[0], nq), 
                  dtype=float_type)

for i, (local_facet, cell, facet) in enumerate(local_facets):
    coord_dofs = x_g[x_dofs[cell]]

    for q in range(nq):
        J_cell = dphi[:, q, :] @ coord_dofs[:, :gdim]

        J_facet = J_cell @ hexahedron_reference_facet_jacobian[local_facet]

        detJ_f[i, q] = np.fabs(np.linalg.det(J_facet.T.dot(J_facet))) * wts[q]

for i, (local_facet, cell, facet) in enumerate(local_facets):
    x_ = u[dofmap[cell][local_facet_dof[local_facet]]]

    x_ *= detJ_f[i, local_facet_dof[local_facet]] * coeffs[cell]

    b[dofmap[cell][local_facet_dof[local_facet]]] += x_

print(b)

# Use DOLFINx assembler for comparison
md = {"quadrature_rule": "GLL", "quadrature_degree": Q[P]}

v = TestFunction(V)
u0.x.array[:] = 1.0
a_dolfinx = form(inner(u0, v) * ds(metadata=md), dtype=float_type)

b_dolfinx = assemble_vector(a_dolfinx)

print(b_dolfinx.array)

print(b_dolfinx.array.nonzero()[0].size)

print(np.allclose(b.nonzero()[0], b_dolfinx.array.nonzero()[0]))
