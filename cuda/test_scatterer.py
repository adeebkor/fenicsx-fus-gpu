#
# .. _test_scatterer:
#
# Test whether the scatterers are working correctly by comparing the output
# with DOLFINx.
# =============================================================================
# Copyright (C) 2024 Adeeb Arif Kor

import numpy as np
from mpi4py import MPI

import numba.cuda as cuda

import basix
import basix.ufl
from dolfinx.fem import functionspace, Function
from dolfinx.la import InsertMode
from dolfinx.mesh import create_box, CellType, GhostMode

from scatterer import scatter_reverse, scatter_forward
from utils import compute_scatterer_data

# MPI
comm = MPI.COMM_WORLD
rank = comm.rank

# Check if CUDA is available
if cuda.is_available():
    print("CUDA is available")

cuda.detect()
cuda.select_device(rank)

print(f"{rank} : {cuda.get_current_device()}")

# Set float type
float_type = np.float64

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

N = 4
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

# Tensor product element
family = basix.ElementFamily.P
variant = basix.LagrangeVariant.gll_warped
cell_type = mesh.basix_cell()

basix_element = basix.create_tp_element(family, cell_type, P, variant)
element = basix.ufl._BasixElement(basix_element)  # basix ufl element

# Create functions space
V = functionspace(mesh, element)
dofmap = V.dofmap.list
imap = V.dofmap.index_map
nlocal = imap.size_local

if rank == 0:
    print(f"Number of degrees-of-freedom: {imap.size_global}")

# ------------ #
# Scatter data #
# ------------ #

owners_data, ghosts_data = compute_scatterer_data(imap)

owners_idx_d = [cuda.to_device(owner_idx) for owner_idx in owners_data[0]]
owners_size = owners_data[1]
unique_owners = owners_data[2]

ghosts_idx_d = [cuda.to_device(ghost_idx) for ghost_idx in ghosts_data[0]]
ghosts_size = ghosts_data[1]
unique_ghosts = ghosts_data[2]

owners_data_d = [owners_idx_d, owners_size, unique_owners]
ghosts_data_d = [ghosts_idx_d, ghosts_size, unique_ghosts]

# Define function for testing
u0 = Function(V, dtype=float_type)
u0.interpolate(
    lambda x: 100
    * np.sin(2 * np.pi * x[0])
    * np.cos(3 * np.pi * x[1])
    * np.sin(4 * np.pi * x[2])
)
u_ = u0.x.array.copy()

# -------------------- #
# Test scatter reverse #
# -------------------- #

scatter_rev = scatter_reverse(comm, owners_data_d, ghosts_data_d, nlocal, float_type)

# Allocate memory on the device
u_d = cuda.to_device(u_)

# Scatter
scatter_rev(u_d)

# Copy to host
u_d.copy_to_host(u_)

# Do scatter reverse using DOLFINx
u0.x.scatter_reverse(InsertMode.add)

# Check the difference between the vectors
print(f"REVERSE: {rank}: {np.allclose(u0.x.array, u_)}", flush=True)

# -------------------- #
# Test scatter forward #
# -------------------- #

scatter_fwd = scatter_forward(comm, owners_data_d, ghosts_data_d, nlocal, float_type)

# Allocate memory on the device
u_d = cuda.to_device(u_)

# Scatter forward
scatter_fwd(u_d)

# Copy to host
u_d.copy_to_host(u_)

# Do scatter forward using DOLFINx
u0.x.scatter_forward()

# Check the difference between the vectors
print(f"FORWARD: {rank}: {np.allclose(u0.x.array, u_)}", flush=True)
