import numpy as np
from mpi4py import MPI

import numba.cuda as cuda

import basix
import basix.ufl
from dolfinx.fem import assemble_vector, functionspace, form, Function
from dolfinx.la import InsertMode
from dolfinx.mesh import create_box, locate_entities_boundary, CellType, GhostMode
from ufl import inner, grad, ds, dx, TestFunction

# Check if CUDA is available
if cuda.is_available():
    print("CUDA is available")

cuda.detect()

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

basix_element = basix.create_tp_element(family, cell_type, P, variant)
element = basix.ufl._BasixElement(basix_element)  # basix ufl element

# Create function space
V = functionspace(mesh, element)
dofmap = V.dofmap.list
imap = V.dofmap.index_map

# MPI
rank = MPI.COMM_WORLD.rank
barrier = MPI.COMM_WORLD.Barrier

# Extract data for communication
nlocal = imap.size_local
nghost = imap.num_ghosts
owners = imap.owners
unique_owners, send_size = np.unique(owners, return_counts=True)
send_ind = np.argsort(owners)

# print(f"{rank} : {unique_owners}")

shared_dofs = imap.index_to_dest_ranks()
shared_rank = np.unique(shared_dofs.array)

source = []
for r in shared_rank:
    for dof in range(nlocal):
        if r in shared_dofs.links(dof):
            source.append(r)

source = np.array(source)
src, recv_size = np.unique(source, return_counts=True)
# print(f"{rank} : {src}")

vec = np.ones(nlocal + nghost, dtype=float_type)
recv_offsets = np.cumsum(recv_size)
recv_offsets = np.insert(recv_offsets, 0, 0)

# Communicate indices data
all_reqs = []
send_buff = np.zeros(send_ind.size, dtype=np.int64)
send_offsets = np.cumsum(send_size)
send_offsets = np.insert(send_offsets, 0, 0)
send_buff[:] = imap.ghosts[send_ind]
for i, owner in enumerate(unique_owners):  # send to destination
    begin = send_offsets[i]
    end = send_offsets[i + 1]
    reqs = MPI.COMM_WORLD.Isend(send_buff[begin:end], dest=owner)
    all_reqs.append(reqs)

vec_ghost_recv = np.zeros(sum(recv_size), dtype=np.int64)
for i, s in enumerate(src):  # receive from source
    begin = recv_offsets[i]
    end = recv_offsets[i + 1]
    reqr = MPI.COMM_WORLD.Irecv(vec_ghost_recv[begin:end], source=s)
    all_reqs.append(reqr)

MPI.Request.Waitall(all_reqs)

print(f"{rank} : {vec_ghost_recv - imap.local_range[0]}")
recv_idx = vec_ghost_recv - imap.local_range[0]

vec[recv_idx] += vec_ghost_recv  # unpack
send_buffer = vec[recv_idx]  # pack

# Correct size on both send and receive
# Putting it into local array
