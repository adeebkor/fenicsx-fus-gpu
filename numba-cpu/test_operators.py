#
# .. _test_operators:
#
# Test whether the operators are working correctly by comparing the output with
# DOLFINx.
# =============================================================================
# Copyright (C) 2024 Adeeb Arif Kor

import numpy as np
from mpi4py import MPI

import basix
import basix.ufl
from dolfinx.fem import assemble_vector, functionspace, form, Function
from dolfinx.la import InsertMode
from dolfinx.mesh import create_box, locate_entities_boundary, CellType, GhostMode
from ufl import inner, grad, ds, dx, TestFunction

from precompute import (
    compute_scaled_jacobian_determinant,
    compute_scaled_geometrical_factor,
    compute_boundary_facets_scaled_jacobian_determinant,
)
from operators import mass_operator, stiffness_operator
from scatterer import scatter_forward, scatter_reverse
from utils import facet_integration_domain

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

nd = P + 1
Nd = nd * nd * nd
Nf = nd * nd

comm = MPI.COMM_WORLD
N = 16
mesh = create_box(
    comm,
    ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    (N, N, N),
    cell_type=CellType.hexahedron,
    ghost_mode=GhostMode.none,
    dtype=float_type,
)

# Mesh geometry data
x_dofs = mesh.geometry.dofmap
x_g = mesh.geometry.x
cell_type = mesh.basix_cell()

# Uncomment below if we would like to test unstructured mesh
mesh.geometry.x[:, :] += np.random.uniform(-0.01, 0.01, (mesh.geometry.x.shape[0], 3))

# Tensor product element
family = basix.ElementFamily.P
variant = basix.LagrangeVariant.gll_warped
cell_type = mesh.basix_cell()

basix_element = basix.create_tp_element(family, cell_type, P, variant)
element = basix.ufl._BasixElement(basix_element)  # basix ufl element

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

# Compute geometric data of cell entities
pts, wts = basix.quadrature.make_quadrature(
    basix.CellType.hexahedron, Q[P], basix.QuadratureType.gll
)
nq = wts.size

gelement = basix.create_element(
    basix.ElementFamily.P, mesh.basix_cell(), 1, dtype=float_type
)
gtable = gelement.tabulate(1, pts)
dphi = gtable[1:, :, :, 0]

# Compute scaled Jacobian determinant (cell)
detJ = np.zeros((num_cells, nq), dtype=float_type)
compute_scaled_jacobian_determinant(detJ, (x_dofs, x_g), num_cells, dphi, wts)

cell_constants = np.ones(dofmap.shape[0], dtype=float_type)

# Compute scaled geometrical factor (J^{-T}J_{-1})
G = np.zeros((num_cells, nq, (3 * (gdim - 1))), dtype=float_type)
compute_scaled_geometrical_factor(G, (x_dofs, x_g), num_cells, dphi, wts)

# Compute geometric data of boundary facet entities
boundary_facets = locate_entities_boundary(
    mesh, mesh.topology.dim - 1, lambda x: np.full(x.shape[1], True, dtype=bool)
)

boundary_data = facet_integration_domain(
    boundary_facets, mesh
)  # cells with boundary facets
local_facet_dof = np.array(
    basix_element.entity_closure_dofs[2], dtype=np.int32
)  # local DOF on facets

pts_f, wts_f = basix.quadrature.make_quadrature(
    basix.CellType.quadrilateral, Q[P], basix.QuadratureType.gll
)
nq_f = wts_f.size

# Evaluation points on the facets of the reference hexahedron
pts_0 = pts_f[:, 0]
pts_1 = pts_f[:, 1]

pts_f = np.zeros((6, nq_f, 3), dtype=float_type)
pts_f[0, :, :] = np.c_[pts_0, pts_1, np.zeros(nq_f, dtype=float_type)]  # z = 0
pts_f[1, :, :] = np.c_[pts_0, np.zeros(nq_f, dtype=float_type), pts_1]  # y = 0
pts_f[2, :, :] = np.c_[np.zeros(nq_f, dtype=float_type), pts_0, pts_1]  # x = 0
pts_f[3, :, :] = np.c_[np.ones(nq_f, dtype=float_type), pts_0, pts_1]  # x = 1
pts_f[4, :, :] = np.c_[pts_0, np.ones(nq_f, dtype=float_type), pts_1]  # y = 1
pts_f[5, :, :] = np.c_[pts_0, pts_1, np.ones(nq_f, dtype=float_type)]  # z = 1

# Derivatives on the facets of the reference hexahedron
dphi_f = np.zeros((6, 3, nq_f, 8), dtype=float_type)

for f in range(6):
    gtable_f = gelement.tabulate(1, pts_f[f, :, :]).astype(float_type)
    dphi_f[f, :, :, :] = gtable_f[1:, :, :, 0]

# Compute scaled Jacobian determinant (boundary facets)
detJ_f = np.zeros((boundary_data.shape[0], nq_f), dtype=float_type)
compute_boundary_facets_scaled_jacobian_determinant(
    detJ_f, (x_dofs, x_g), boundary_data, dphi_f, wts_f
)

# Create boundary facets dofmap
bfacet_dofmap = np.zeros(
    (boundary_data.shape[0], local_facet_dof.shape[1]), dtype=np.int32
)

for i, (cell, local_facet) in enumerate(boundary_data):
    bfacet_dofmap[i, :] = dofmap[cell][local_facet_dof[local_facet]]

bfacet_constants = np.ones(bfacet_dofmap.shape[0], dtype=float_type)

# -------------- #
# Scatterer data #
# -------------- #

imap = V.dofmap.index_map

# Compute ghosts data in this process that are owned by other processes
nlocal = imap.size_local
nghost = imap.num_ghosts
owners = imap.owners
unique_owners, owners_size = np.unique(owners, return_counts=True)
owners_idx = np.argsort(owners)

owners_offsets = np.cumsum(owners_size)
owners_offsets = np.insert(owners_offsets, 0, 0)

# Compute owned data by this process that are ghosts data in other process
shared_dofs = imap.index_to_dest_ranks()
shared_ranks = np.unique(shared_dofs.array)

ghosts = []
for shared_rank in shared_ranks:
    for dof in range(nlocal):
        if shared_rank in shared_dofs.links(dof):
            ghosts.append(shared_rank)

ghosts = np.array(ghosts)
unique_ghosts, ghosts_size = np.unique(ghosts, return_counts=True)
ghosts_offsets = np.cumsum(ghosts_size)
ghosts_offsets = np.insert(ghosts_offsets, 0, 0)

# Communicate the ghost indices
all_requests = []

# Send
send_buff_idx = np.zeros(np.sum(owners_size), dtype=np.int64)
send_buff_idx[:] = imap.ghosts[owners_idx]
for i, owner in enumerate(unique_owners):  # send to destination
    begin = owners_offsets[i]
    end = owners_offsets[i + 1]
    reqs = comm.Isend(send_buff_idx[begin:end], dest=owner)
    all_requests.append(reqs)

# Receive
recv_buff_idx = np.zeros(np.sum(ghosts_size), dtype=np.int64)
for i, ghost in enumerate(unique_ghosts):  # receive from source
    begin = ghosts_offsets[i]
    end = ghosts_offsets[i + 1]
    reqr = comm.Irecv(recv_buff_idx[begin:end], source=ghost)
    all_requests.append(reqr)

MPI.Request.Waitall(all_requests)

ghosts_idx = recv_buff_idx - imap.local_range[0]

owners_data = [owners_idx, owners_size, owners_offsets, unique_owners]
ghosts_data = [ghosts_idx, ghosts_size, ghosts_offsets, unique_ghosts]

scatter_rev = scatter_reverse(comm, owners_data, ghosts_data, nlocal, float_type)
scatter_fwd = scatter_forward(comm, owners_data, ghosts_data, nlocal, float_type)

# DOLFINx assembler for comparison
md = {"quadrature_rule": "GLL", "quadrature_degree": Q[P]}
v = TestFunction(V)

# ------------- #
# Mass operator #
# ------------- #

b[:] = 0.0
mass_operator_cell = mass_operator(Nd, float_type)
mass_operator_cell(u, cell_constants, b, detJ, dofmap)
scatter_rev(b)

u0.x.array[:] = 1.0
a0_dolfinx = form(inner(u0, v) * dx(metadata=md), dtype=float_type)
b0_dolfinx = assemble_vector(a0_dolfinx)
b0_dolfinx.scatter_reverse(InsertMode.add)

# Check the difference between the vectors
mass_difference = np.linalg.norm(b - b0_dolfinx.array) / np.linalg.norm(
    b0_dolfinx.array
)
print(f"Euclidean difference (mass operator): {mass_difference}", flush=True)

assert mass_difference < tol

# ------------------ #
# Stiffness operator #
# ------------------ #

# Create 1D element for sum factorisation
element_1D = basix.create_element(
    family, basix.CellType.interval, P, variant, dtype=float_type
)
pts_1D, wts_1D = basix.quadrature.make_quadrature(
    basix.CellType.interval, Q[P], basix.QuadratureType.gll
)
pts_1D, wts_1D = pts_1D.astype(float_type), wts_1D.astype(float_type)

table_1D = element_1D.tabulate(1, pts_1D)
dphi_1D = table_1D[1, :, :, 0]
nd = dphi_1D.shape[1]
dphi_1D = dphi_1D.flatten()

u0.interpolate(
    lambda x: 100
    * np.sin(2 * np.pi * x[0])
    * np.cos(3 * np.pi * x[1])
    * np.sin(4 * np.pi * x[2])
)
b[:] = 0.0
stiff_operator_cell = stiffness_operator(P, dphi_1D, float_type)
stiff_operator_cell(u, cell_constants, b, G, dofmap)
scatter_rev(b)

a1_dolfinx = form(inner(grad(u0), grad(v)) * dx(metadata=md), dtype=float_type)
b1_dolfinx = assemble_vector(a1_dolfinx)
b1_dolfinx.scatter_reverse(InsertMode.add)

# Check the difference between the vectors
stiffness_difference = np.linalg.norm(b - b1_dolfinx.array) / np.linalg.norm(
    b1_dolfinx.array
)
print(f"Euclidean difference (stiffness operator): {stiffness_difference}", flush=True)

assert stiffness_difference < tol

# ------------------ #
# Boundary operators #
# ------------------ #

b[:] = 0.0
u[:] = 1.0
mass_operator_bfacet = mass_operator(Nf, float_type)
mass_operator_bfacet(u, bfacet_constants, b, detJ_f, bfacet_dofmap)
scatter_rev(b)

a3_dolfinx = form(inner(u0, v) * ds(metadata=md), dtype=float_type)
b3_dolfinx = assemble_vector(a3_dolfinx)
b3_dolfinx.scatter_reverse(InsertMode.add)

# Check the difference between the vectors
bfacet_difference = np.linalg.norm(b - b3_dolfinx.array) / np.linalg.norm(
    b3_dolfinx.array
)
print(f"Euclidean difference (boundary operator): {bfacet_difference}", flush=True)

# Test the closeness between the vectors
assert bfacet_difference < tol
