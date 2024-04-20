"""
ADD DOC!
"""

import numpy as np
from mpi4py import MPI

import numba.cuda as cuda

import basix
import basix.ufl
from dolfinx.fem import functionspace, Function
from dolfinx.la import InsertMode
from dolfinx.mesh import create_box, CellType, GhostMode

from scatterer import scatter_reverse, scatter_forward

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
    comm, ((0., 0., 0.), (1., 1., 1.)),
    (N, N, N), cell_type=CellType.hexahedron, 
    ghost_mode=GhostMode.none,
    dtype=float_type
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

if rank == 0:
    print(f"Number of degrees-of-freedom: {imap.size_global}")

# Compute ghosts data in this process that are owned by other processes
nlocal = imap.size_local
nghost = imap.num_ghosts
owners = imap.owners
unique_owners, owners_size = np.unique(owners, return_counts=True)
owners_argsorted = np.argsort(owners)

owners_offsets = np.cumsum(owners_size)
owners_offsets = np.insert(owners_offsets, 0, 0)

owners_idx = [np.zeros(size, dtype=np.int64) for size in owners_size]
for i, owner in enumerate(unique_owners):
    begin = owners_offsets[i]
    end = owners_offsets[i + 1]
    owners_idx[i] = owners_argsorted[begin:end]

owners_idx_d = [cuda.to_device(owner_idx) for owner_idx in owners_idx]

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

all_requests = []

# Send
send_buff_idx = [np.zeros(size, dtype=np.int64) for size in owners_size]
for i, owner in enumerate(unique_owners):
    begin = owners_offsets[i]
    end = owners_offsets[i + 1]
    send_buff_idx[i] = imap.ghosts[owners_argsorted[begin:end]]

send_buff_idx_d = [cuda.to_device(send_buff) for send_buff in send_buff_idx]
for i, owner in enumerate(unique_owners):
    reqs = comm.Isend(send_buff_idx_d[i], dest=owner)
    all_requests.append(reqs)

# Receive
recv_buff_idx = [np.zeros(size, dtype=np.int64) for size in ghosts_size]
recv_buff_idx_d = [cuda.to_device(recv_buff) for recv_buff in recv_buff_idx]
for i, ghost in enumerate(unique_ghosts):
    reqr = comm.Irecv(recv_buff_idx_d[i], source=ghost)
    all_requests.append(reqr)

MPI.Request.Waitall(all_requests)

for i, ghosts in enumerate(unique_ghosts):
    recv_buff_idx[i] = recv_buff_idx_d[i].copy_to_host()

ghosts_idx = [recv_buff - imap.local_range[0] for recv_buff in recv_buff_idx]

ghosts_idx_d = [cuda.to_device(ghost_buff) for ghost_buff in ghosts_idx]

owners_data_d = [owners_idx_d, owners_size, owners_offsets, unique_owners]
ghosts_data_d = [ghosts_idx_d, ghosts_size, ghosts_offsets, unique_ghosts]

# Define function for testing
u0 = Function(V, dtype=float_type)
u0.interpolate(lambda x: 100 * np.sin(2*np.pi*x[0]) * np.cos(3*np.pi*x[1])
               * np.sin(4*np.pi*x[2]))
u_ = u0.x.array.copy()

# -------------------- #
# Test scatter reverse #
# -------------------- #

scatter_rev = scatter_reverse(
    comm, owners_data_d, ghosts_data_d, [nlocal, nghost], float_type
)

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

scatter_fwd = scatter_forward(
    comm, owners_data_d, ghosts_data_d, [nlocal, nghost], float_type
)

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
