import numpy as np
from mpi4py import MPI

import basix
import basix.ufl
from dolfinx.fem import assemble_vector, functionspace, form, Function
from dolfinx.la import InsertMode
from dolfinx.mesh import create_box, locate_entities_boundary, CellType, GhostMode

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

nd = P + 1
Nd = nd * nd * nd

N = 4
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

# Tensor product element
family = basix.ElementFamily.P
variant = basix.LagrangeVariant.gll_warped
cell_type = mesh.basix_cell()

basix_element = basix.create_element(family, cell_type, P, variant)  # doesn't work with tp element
element = basix.ufl._BasixElement(basix_element)  # basix ufl element

# Create function space
V = functionspace(mesh, element)
dofmap = V.dofmap.list
imap = V.dofmap.index_map

# MPI
rank = MPI.COMM_WORLD.rank
barrier = MPI.COMM_WORLD.Barrier

# Send setup
nlocal = imap.size_local
nghost = imap.num_ghosts
owners = imap.owners
unique_owners, ghost_size = np.unique(owners, return_counts=True)
print(f"{rank} : {unique_owners} : {ghost_size}")
sort_idx = np.argsort(owners)

send_offsets = np.cumsum(ghost_size)
send_offsets = np.insert(send_offsets, 0, 0)

# Receive setup
shared_dofs = imap.index_to_dest_ranks()
shared_ranks = np.unique(shared_dofs.array)

sources = []
for shared_rank in shared_ranks:
    for dof in range(nlocal):
        if shared_rank in shared_dofs.links(dof):
            sources.append(shared_rank)

sources = np.array(sources)
unique_sources, recv_size = np.unique(sources, return_counts=True)
recv_offsets = np.cumsum(recv_size)
recv_offsets = np.insert(recv_offsets, 0, 0)

all_requests = []

# Send
send_buff = np.zeros(np.sum(send_size), dtype=np.int64)
send_buff[:] = imap.ghosts[sort_idx]
for i, owner in enumerate(unique_owners):  # send to destination
    begin = send_offsets[i]
    end = send_offsets[i + 1]
    reqs = MPI.COMM_WORLD.Isend(send_buff[begin:end], dest=owner)
    all_requests.append(reqs)

# Receive
recv_buff = np.zeros(np.sum(recv_size), dtype=np.int64)
for i, source in enumerate(unique_sources):  # receive from source
    begin = recv_offsets[i]
    end = recv_offsets[i + 1]
    reqr = MPI.COMM_WORLD.Irecv(recv_buff[begin:end], source=source)
    all_requests.append(reqr)

MPI.Request.Waitall(all_requests)

recv_idx = recv_buff - imap.local_range[0]

# -------------------- #
# Test scatter reverse #
# -------------------- #

u0 = Function(V, dtype=float_type)
u0.interpolate(lambda x: 100 * np.sin(2*np.pi*x[0]) * np.cos(3*np.pi*x[1])
               * np.sin(4*np.pi*x[2]))
u = u0.x.array
u_ = u.copy()

all_requests = []

# Send
send_buff = np.zeros(np.sum(send_size), dtype=float_type)
u_ghosts = u[-nghost:]
send_buff[:] = u_ghosts[sort_idx]
for i, owner in enumerate(unique_owners):  # send to destination
    begin = send_offsets[i]
    end = send_offsets[i + 1]
    reqs = MPI.COMM_WORLD.Isend(send_buff[begin:end], dest=owner)
    all_requests.append(reqs)

# Receive
recv_buff = np.zeros(np.sum(recv_size), dtype=float_type)
for i, source in enumerate(unique_sources):  # receive from source
    begin = recv_offsets[i]
    end = recv_offsets[i + 1]
    reqr = MPI.COMM_WORLD.Irecv(recv_buff[begin:end], source=source)
    all_requests.append(reqr)

MPI.Request.Waitall(all_requests)

# Do scatter reverse
u_[recv_idx] += recv_buff
u0.x.scatter_reverse(InsertMode.add)
diff_idx = np.where(~np.isclose(u_, u))
print(f"REVERSE: {rank}: {np.allclose(u, u_)}", flush=True)

# -------------------- #
# Test scatter forward #
# -------------------- #

all_requests = []

# Send
send_buff = np.zeros(np.sum(recv_size), dtype=float_type)
u_owners = u[recv_idx]
send_buff[:] = u_owners
for i, dest in enumerate(unique_sources):
    begin = recv_offsets[i]
    end = recv_offsets[i + 1]
    reqs = MPI.COMM_WORLD.Isend(send_buff[begin:end], dest=dest)
    all_requests.append(reqs)

# Receive
recv_buff = np.zeros(np.sum(send_size), dtype=float_type)
for i, src in enumerate(unique_owners):
    begin = send_offsets[i]
    end = send_offsets[i + 1]
    reqr = MPI.COMM_WORLD.Irecv(recv_buff[begin:end], source=src)
    all_requests.append(reqr)

MPI.Request.Waitall(all_requests)

# Do scatter forward
u_[-nghost:][sort_idx] = recv_buff
u0.x.scatter_forward()
diff_idx = np.where(~np.isclose(u_, u))
print(f"FORWARD: {rank}: {np.allclose(u, u_)}", flush=True)

# vec[recv_idx] += vec_ghost_recv  # unpack
# send_buffer = vec[recv_idx]  # pack
