#
# .. _test_scatterer:
#
# Test whether the scatterers are working correctly by comparing the output
# with DOLFINx.
# =============================================================================
# Copyright (C) 2024 Adeeb Arif Kor

import numpy as np
from mpi4py import MPI

import basix
import basix.ufl
from dolfinx.fem import functionspace, Function
from dolfinx.la import InsertMode
from dolfinx.mesh import create_box, CellType, GhostMode

from scatterer import scatter_forward, scatter_reverse

# MPI
comm = MPI.COMM_WORLD
rank = comm.rank

# Set float type
float_type = np.float64

P = 4  # Basis function order

N = 4
mesh = create_box(
    comm,
    ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    (N, N, N),
    cell_type=CellType.hexahedron,
    ghost_mode=GhostMode.none,
    dtype=float_type,
)

# Tensor product element
family = basix.ElementFamily.P
variant = basix.LagrangeVariant.gll_warped
cell_type = mesh.basix_cell()

basix_element = basix.create_tp_element(family, cell_type, P, variant)
element = basix.ufl._BasixElement(basix_element)  # basix ufl element

# Prepare data for scatterer
V = functionspace(mesh, element)
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

scatter_rev = scatter_reverse(comm, owners_data, ghosts_data, nlocal, float_type)

scatter_rev(u_)
u0.x.scatter_reverse(InsertMode.add)
print(f"REVERSE: {rank}: {np.allclose(u0.x.array, u_)}", flush=True)

# -------------------- #
# Test scatter forward #
# -------------------- #

scatter_fwd = scatter_forward(comm, owners_data, ghosts_data, nlocal, float_type)

scatter_fwd(u_)
u0.x.scatter_forward()
print(f"FORWARD: {rank}: {np.allclose(u0.x.array, u_)}", flush=True)
