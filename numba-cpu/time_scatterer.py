#
# .. _time_scatterer:
#
# Time the scatterers
# =============================================================================
# Copyright (C) 2024 Adeeb Arif Kor

from time import perf_counter_ns

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

basix_element = basix.create_tp_element(
    family, cell_type, P, variant
)  # doesn't work with tp element, why?
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
# Time scatter reverse #
# -------------------- #

# Numba
scatter_rev = scatter_reverse(comm, owners_data, ghosts_data, nlocal, float_type)

# Call once to jit compile function
scatter_rev(u_)

timing_scatter_rev = np.empty(50)
for i in range(timing_scatter_rev.size):
    tic = perf_counter_ns()
    scatter_rev(u_)
    toc = perf_counter_ns()
    timing_scatter_rev[i] = toc - tic

timing_scatter_rev *= 1e-9

print(
    f"Elapsed time (scatter reverse (Numba)): "
    f"{timing_scatter_rev.mean():.7f} ± "
    f"{timing_scatter_rev.std():.7f} s"
)

# DOLFINx
timing_scatter_rev_dolfinx = np.empty(10)
for i in range(timing_scatter_rev_dolfinx.size):
    tic = perf_counter_ns()
    u0.x.scatter_reverse(InsertMode.add)
    toc = perf_counter_ns()
    timing_scatter_rev_dolfinx[i] = toc - tic

timing_scatter_rev_dolfinx *= 1e-9

print(
    f"Elapsed time (scatter reverse (DOLFINx)): "
    f"{timing_scatter_rev_dolfinx.mean():.7f} ± "
    f"{timing_scatter_rev_dolfinx.std():.7f} s"
)

# Differences between the two implementations
scatter_rev_diff = timing_scatter_rev.mean() / timing_scatter_rev_dolfinx.mean()
print(
    f"The DOLFINx implementation is {scatter_rev_diff} times faster",
    "(scatter reverse)",
)

# -------------------- #
# Time scatter forward #
# -------------------- #

# Numba
scatter_fwd = scatter_forward(comm, owners_data, ghosts_data, nlocal, float_type)

# Call once to jit compile function
scatter_fwd(u_)

timing_scatter_fwd = np.empty(10)
for i in range(timing_scatter_fwd.size):
    tic = perf_counter_ns()
    scatter_fwd(u_)
    toc = perf_counter_ns()
    timing_scatter_fwd[i] = toc - tic

timing_scatter_fwd *= 1e-9

print(
    f"Elapsed time (scatter forward (Numba)): "
    f"{timing_scatter_fwd.mean():.7f} ± "
    f"{timing_scatter_fwd.std():.7f} s"
)

# DOLFINx
timing_scatter_fwd_dolfinx = np.empty(10)
for i in range(timing_scatter_fwd_dolfinx.size):
    tic = perf_counter_ns()
    u0.x.scatter_forward()
    toc = perf_counter_ns()
    timing_scatter_fwd_dolfinx[i] = toc - tic

timing_scatter_fwd_dolfinx *= 1e-9

print(
    f"Elapsed time (scatter forward (DOLFINx)): "
    f"{timing_scatter_fwd_dolfinx.mean():.7f} ± "
    f"{timing_scatter_fwd_dolfinx.std():.7f} s"
)

# Differences between the two implementations
scatter_fwd_diff = timing_scatter_fwd.mean() / timing_scatter_fwd_dolfinx.mean()
print(
    f"The DOLFINx implementation is {scatter_fwd_diff} times faster",
    "(scatter forward)",
)
