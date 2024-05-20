#
# Nonlinear wave
# - Plane wave
# - Homogenous media
# =================================
# Copyright (C) 2024 Adeeb Arif Kor


import numpy as np
import numba.cuda as cuda
from mpi4py import MPI

import basix
import basix.ufl
from dolfinx import cpp, la
from dolfinx.common import Timer
from dolfinx.fem import functionspace, Function
from dolfinx.io import VTXWriter
from dolfinx.mesh import create_box, locate_entities_boundary, CellType, GhostMode

from precompute import (
    compute_scaled_jacobian_determinant,
    compute_scaled_geometrical_factor,
    compute_boundary_facets_scaled_jacobian_determinant,
)
from operators import (
    mass_operator,
    stiffness_operator,
    axpy,
    copy,
    fill,
    pointwise_divide,
    square,
)
from scatterer import (
    scatter_forward,
    scatter_reverse,
)
from utils import (
    facet_integration_domain,
    compute_eval_params,
    compute_diffusivity_of_sound,
)

# MPI
comm = MPI.COMM_WORLD
rank = comm.rank

# Check if CUDA is available
if cuda.is_available():
    print("CUDA is available", flush=True)

cuda.detect()
cuda.select_device(rank)

print(f"{rank} : {cuda.get_current_device()}")

float_type = np.float64

# Source parameters
source_frequency = 0.5e6  # Hz
source_amplitude = 60000.0  # Pa
period = 1.0 / source_frequency  # s
angular_frequency = 2.0 * np.pi * source_frequency  # rad/s

# Material parameters
speed_of_sound = 1500.0  # m/s
density = 1000  # kg/m^3
nonlinear_coefficient = 100.0
attenuation_coefficient_dB = 50
attenuation_coefficient_Np = attenuation_coefficient_dB / 20 * np.log(10)
diffusivity_of_sound = compute_diffusivity_of_sound(
    angular_frequency, speed_of_sound, attenuation_coefficient_dB
)

# Domain parameters
domain_length = 0.12  # m

# FE parameters
basis_degree = 6
quadrature_degree = {
    2: 3,
    3: 4,
    4: 6,
    5: 8,
    6: 10,
    7: 12,
    8: 14,
    9: 16,
    10: 18,
}

nd = basis_degree + 1

# Mesh parameters
wave_length = speed_of_sound / source_frequency
num_of_waves = domain_length / wave_length
num_element = int(1 * num_of_waves)

# Create mesh
mesh = create_box(
    comm,
    ((0.0, 0.0, 0.0), (domain_length, domain_length, domain_length)),
    (num_element, num_element, num_element),
    cell_type=CellType.hexahedron,
    ghost_mode=GhostMode.none,
    dtype=float_type,
)

# Mesh data
tdim = mesh.topology.dim
gdim = mesh.geometry.dim
num_cells = mesh.topology.index_map(tdim).size_local
hmin = np.array(
    [cpp.mesh.h(mesh._cpp_object, tdim, np.arange(num_cells, dtype=np.int32)).min()],
    dtype=float_type,
)
mesh_size = np.zeros(1, dtype=float_type)
comm.Allreduce(hmin, mesh_size, op=MPI.MIN)

# Mesh geometry data
x_dofs = mesh.geometry.dofmap
x_g = mesh.geometry.x
cell_type = mesh.basix_cell()

# Temporal parameters
CFL = 0.70
time_step_size = CFL * mesh_size / (speed_of_sound * basis_degree**2)
step_per_period = int(period / time_step_size) + 1
time_step_size = period / step_per_period
start_time = 0.0
final_time = domain_length / speed_of_sound + 2.0 / source_frequency
number_of_step = int((final_time - start_time) / time_step_size) + 1

if rank == 0:
    print(f"Number of steps: {number_of_step}", flush=True)

# -----------------------------------------------------------------------------
# Evaluation parameters
npts_x = 100
npts_y = 100

x_p = np.linspace(0, domain_length, npts_x, dtype=float_type)
y_p = np.linspace(0, domain_length, npts_y, dtype=float_type)

X_p, Y_p = np.meshgrid(x_p, y_p)

points = np.zeros((3, npts_x*npts_y), dtype=float_type)
points[0] = X_p.flatten()
points[1] = Y_p.flatten()

x_eval, cell_eval = compute_eval_params(mesh, points, float_type)

data = np.zeros_like(x_eval, dtype=float_type)

try:
    data[:, 0] = x_eval[:, 0]
    data[:, 1] = x_eval[:, 1]
except:
    pass
# -----------------------------------------------------------------------------

# Define a DG function space for the material parameters
V_DG = functionspace(mesh, ("DG", 0))

c0 = Function(V_DG, dtype=float_type)
c0.x.array[:] = speed_of_sound
c0_ = c0.x.array

rho0 = Function(V_DG, dtype=float_type)
rho0.x.array[:] = density
rho0_ = rho0.x.array

delta0 = Function(V_DG, dtype=float_type)
delta0.x.array[:] = diffusivity_of_sound
delta0_ = delta0.x.array

beta0 = Function(V_DG, dtype=float_type)
beta0.x.array[:] = nonlinear_coefficient
beta0_ = beta0.x.array

# Tensor product element
family = basix.ElementFamily.P
variant = basix.LagrangeVariant.gll_warped

basix_element_tp = basix.create_tp_element(family, cell_type, basis_degree, variant, dtype=float_type)
perm = np.argsort(np.array(basix_element_tp.dof_ordering, dtype=np.int32))

# Basix element
basix_element = basix.create_element(family, cell_type, basis_degree, variant, dtype=float_type)
element = basix.ufl._BasixElement(basix_element)  # basix ufl element

# Define function space and functions
V = functionspace(mesh, element)
dofmap = V.dofmap.list[:, perm]
imap = V.dofmap.index_map

if rank == 0:
    print(f"Number of degrees-of-freedom: {imap.size_global}")

# ------------ #
# Scatter data #
# ------------ #

if rank == 0:
    print("Computing scatterer data", flush=True)

# Compute ghosts data in this process that are owned by other processes
nlocal = imap.size_local
nghost = imap.num_ghosts
ndofs = nlocal + nghost
owners = imap.owners
unique_owners, owners_size = np.unique(owners, return_counts=True)
owners_argsorted = np.argsort(owners)

print(f"*** {rank}: Computing owners", flush=True)

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

print(f"*** {rank}: Computing ghosts", flush=True)
# Bottleneck - is there a way to make this faster?
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

print(f"*** {rank} : Sending!", flush=True)

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

print(f"*** {rank} : Receiving!", flush=True)

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

owners_data_d = [owners_idx_d, owners_size, unique_owners]
ghosts_data_d = [ghosts_idx_d, ghosts_size, unique_ghosts]

print(f"*** {rank} : Instantiate scatterer!", flush=True)

# Instantiate scatterer
scatter_rev = scatter_reverse(comm, owners_data_d, ghosts_data_d, nlocal, float_type)
scatter_fwd = scatter_forward(comm, owners_data_d, ghosts_data_d, nlocal, float_type)

# Define functions
u_t_ = Function(V, dtype=float_type)
u_n_ = Function(V, dtype=float_type)
v_n_ = Function(V, dtype=float_type)

# Get the numpy arrays
u_t = u_t_.x.array
g = u_t.copy()
dg = u_t.copy()
u_n = u_n_.x.array
v_n = v_n_.x.array
w_n = v_n.copy()

u_n[:] = 0.0
v_n[:] = 0.0

# Create LHS and RHS vector
m_ = la.vector(V.dofmap.index_map, dtype=float_type)
b_ = la.vector(V.dofmap.index_map, dtype=float_type)

# Get array for LHS and RHS vector
m = m_.array
m0 = m.copy()
b = b_.array

# Compute geometric data of cell entities
pts, wts = basix.quadrature.make_quadrature(
    basix.CellType.hexahedron, quadrature_degree[basis_degree], basix.QuadratureType.gll
)
nq = wts.size

gelement = basix.create_element(family, cell_type, 1, dtype=float_type)
gtable = gelement.tabulate(1, pts)
dphi = gtable[1:, :, :, 0]

# Compute scaled Jacobian determinant (cell)
if rank == 0:
    print("Computing scaled Jacobian determinant (cell)", flush=True)

detJ = np.zeros((num_cells, nq), dtype=float_type)
compute_scaled_jacobian_determinant(detJ, (x_dofs, x_g), num_cells, dphi, wts)

# Compute scaled geometrical factor (J^{-T}J_{-1})
if rank == 0:
    print("Computing scaled geometrical factor", flush=True)

G = np.zeros((num_cells, nq, (3 * (gdim - 1))), dtype=float_type)
compute_scaled_geometrical_factor(G, (x_dofs, x_g), num_cells, dphi, wts)

# Boundary facet (source)
boundary_facets1 = locate_entities_boundary(
    mesh, mesh.topology.dim - 1, lambda x: np.isclose(x[0], np.finfo(float).eps)
)

# Boundary facet (absorbing)
boundary_facets2 = locate_entities_boundary(
    mesh, mesh.topology.dim - 1, lambda x: np.isclose(x[0], domain_length)
)

boundary_data1 = facet_integration_domain(
    boundary_facets1, mesh
)  # cells with boundary facets (source)
boundary_data2 = facet_integration_domain(
    boundary_facets2, mesh
)  # cells with boundary facets (absorbing)
local_facet_dof = np.array(
    basix_element_tp.entity_closure_dofs[2], dtype=np.int32
)  # local DOF on facets

pts_f, wts_f = basix.quadrature.make_quadrature(
    basix.CellType.quadrilateral,
    quadrature_degree[basis_degree],
    basix.QuadratureType.gll,
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

# Compute scaled Jacobian determinant (source facets)
if rank == 0:
    print("Computing scaled Jacobian determinant (source facets)", flush=True)

detJ_f1 = np.zeros((boundary_data1.shape[0], nq_f), dtype=float_type)
compute_boundary_facets_scaled_jacobian_determinant(
    detJ_f1, (x_dofs, x_g), boundary_data1, dphi_f, wts_f
)

# Compute scaled Jacobian determinant (absorbing facets)
if rank == 0:
    print("Computing scaled Jacobian determinant (absorbing facets)", flush=True)

detJ_f2 = np.zeros((boundary_data2.shape[0], nq_f), dtype=float_type)
compute_boundary_facets_scaled_jacobian_determinant(
    detJ_f2, (x_dofs, x_g), boundary_data2, dphi_f, wts_f
)

# Create boundary facets dofmap (source)
bfacet_dofmap1 = np.zeros(
    (boundary_data1.shape[0], local_facet_dof.shape[1]), dtype=np.int32
)

for i, (cell, local_facet) in enumerate(boundary_data1):
    bfacet_dofmap1[i, :] = dofmap[cell][local_facet_dof[local_facet]]

# Create boundary facets dofmap (absorbing)
bfacet_dofmap2 = np.zeros(
    (boundary_data2.shape[0], local_facet_dof.shape[1]), dtype=np.int32
)

for i, (cell, local_facet) in enumerate(boundary_data2):
    bfacet_dofmap2[i, :] = dofmap[cell][local_facet_dof[local_facet]]

# Define material coefficients
cell_coeff1 = 1.0 / rho0_ / c0_ / c0_
cell_coeff2 = - 2.0 * beta0_ / rho0_ / rho0_ / c0_ / c0_ / c0_ / c0_
cell_coeff3 = - 1.0 / rho0_ 
cell_coeff4 = - delta0_ / rho0_ / c0_ / c0_
cell_coeff5 = 2.0 * beta0_ / rho0_ / rho0_ / c0_ / c0_ / c0_ / c0_

facet_coeff1_1 = np.zeros((bfacet_dofmap1.shape[0]), dtype=float_type)
facet_coeff2_1 = np.zeros((bfacet_dofmap1.shape[0]), dtype=float_type) 
for i, (cell, local_facet) in enumerate(boundary_data1):
    facet_coeff1_1[i] = 1.0 / rho0_[cell]
    facet_coeff2_1[i] = delta0_[cell] / rho0_[cell] / c0_[cell] / c0_[cell]

facet_coeff1_2 = np.zeros((bfacet_dofmap2.shape[0]), dtype=float_type)
facet_coeff2_2 = np.zeros((bfacet_dofmap2.shape[0]), dtype=float_type)
for i, (cell, local_facet) in enumerate(boundary_data2):
    facet_coeff1_2[i] = delta0_[cell] / rho0_[cell] / c0_[cell] / c0_[cell] / c0_[cell]
    facet_coeff2_2[i] = - 1.0 / rho0_[cell] / c0_[cell]

# Create 1D element for sum factorisation
element_1D = basix.create_element(
    family, basix.CellType.interval, basis_degree, variant, dtype=float_type
)
pts_1D, wts_1D = basix.quadrature.make_quadrature(
    basix.CellType.interval, quadrature_degree[basis_degree], basix.QuadratureType.gll
)
pts_1D, wts_1D = pts_1D.astype(float_type), wts_1D.astype(float_type)

table_1D = element_1D.tabulate(1, pts_1D)
dphi_1D = table_1D[1, :, :, 0]

# ---------------------------- #
# Host to device data transfer #
# ---------------------------- #

# Cell operator
cell_coeff1_d = cuda.to_device(cell_coeff1)
cell_coeff2_d = cuda.to_device(cell_coeff2)
cell_coeff3_d = cuda.to_device(cell_coeff3)
cell_coeff4_d = cuda.to_device(cell_coeff4)
cell_coeff5_d = cuda.to_device(cell_coeff5)
dofmap_d = cuda.to_device(dofmap)
detJ_d = cuda.to_device(detJ)
G_d = cuda.to_device(G)
dphi_1D_d = cuda.to_device(dphi_1D)

# Boundary facet operator
facet_coeff1_1_d = cuda.to_device(facet_coeff1_1)
facet_coeff2_1_d = cuda.to_device(facet_coeff2_1)
facet_coeff1_2_d = cuda.to_device(facet_coeff1_2)
facet_coeff2_2_d = cuda.to_device(facet_coeff2_2)
bfacet_dofmap1_d = cuda.to_device(bfacet_dofmap1)
bfacet_dofmap2_d = cuda.to_device(bfacet_dofmap2)
detJ_f1_d = cuda.to_device(detJ_f1)
detJ_f2_d = cuda.to_device(detJ_f2)

# Arrays
u_t_d = cuda.to_device(u_t)
g_d = cuda.to_device(g)
dg_d = cuda.to_device(dg)
u_n_d = cuda.to_device(u_n)
v_n_d = cuda.to_device(v_n)
w_n_d = cuda.to_device(w_n)
m_d = cuda.to_device(m)
m0_d = cuda.to_device(m0)
b_d = cuda.to_device(b)

# --------------------- #
# JIT compile operators #
# --------------------- #

# Set the number of threads in a block
threadsperblock_m = 128

num_blocks_m = (dofmap.size + (threadsperblock_m - 1)) // threadsperblock_m
num_blocks_f1 = (bfacet_dofmap1.size + (threadsperblock_m - 1)) // threadsperblock_m
num_blocks_f2 = (bfacet_dofmap2.size + (threadsperblock_m - 1)) // threadsperblock_m

threadsperblock_s = (nd, nd, nd)
num_blocks_s = num_cells

mass_operator[num_blocks_m, threadsperblock_m](
    u_t_d, cell_coeff1_d, m_d, detJ_d, dofmap_d
)
stiff_operator_cell = stiffness_operator(basis_degree, float_type)
stiff_operator_cell[num_blocks_s, threadsperblock_s](
    u_t_d, cell_coeff2_d, b_d, G_d, dofmap_d, dphi_1D_d
)

threadsperblock_dofs = 1024
num_blocks_dofs = (ndofs + (threadsperblock_dofs - 1)) // threadsperblock_dofs

axpy[num_blocks_dofs, threadsperblock_dofs](0.0, u_t_d, m_d)
copy[num_blocks_dofs, threadsperblock_dofs](m_d, b_d)
fill[num_blocks_dofs, threadsperblock_dofs](0.0, m_d)
square[num_blocks_dofs, threadsperblock_dofs](u_t_d, m_d)
pointwise_divide[num_blocks_dofs, threadsperblock_dofs](m_d, b_d, u_t_d)

# ------------------------- #
# Assemble the LHS (Steady) #
# ------------------------- #

fill[num_blocks_dofs, threadsperblock_dofs](1.0, u_t_d)

fill[num_blocks_dofs, threadsperblock_dofs](0.0, m0_d)
mass_operator[num_blocks_m, threadsperblock_m](
    u_t_d, cell_coeff1_d, m0_d, detJ_d, dofmap_d
)
if bfacet_dofmap2.any():
    mass_operator[num_blocks_f2, threadsperblock_m](
        u_t_d, facet_coeff1_2_d, m0_d, detJ_f2_d, bfacet_dofmap2_d
    )
scatter_rev(m0_d)

# ---------------------- #
# Set initial conditions #
# ---------------------- #

fill[num_blocks_dofs, threadsperblock_dofs](0.0, u_n_d)
fill[num_blocks_dofs, threadsperblock_dofs](0.0, v_n_d)
fill[num_blocks_dofs, threadsperblock_dofs](0.0, w_n_d)

# --------------- #
# Solve using RK4 #
# --------------- #

# Runge-Kutta data
n_rk = 4
a_runge = np.array([0.0, 0.5, 0.5, 1.0])
b_runge = np.array([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0])
c_runge = np.array([0.0, 0.5, 0.5, 1.0])

# Solution vector at time step
u_ = u_n.copy()
v_ = v_n.copy()

# Solution vectors at intermediate time step
un = u_.copy()
vn = v_.copy()

# Solution vectors at start of time step
u0 = un.copy()
v0 = vn.copy()

# Slope vectors at intermediate time step
ku = u0.copy()
kv = v0.copy()

# Move data from host to device
u_d = cuda.to_device(u_)
v_d = cuda.to_device(v_)
un_d = cuda.to_device(un)
vn_d = cuda.to_device(vn)
u0_d = cuda.to_device(u0)
v0_d = cuda.to_device(v0)
ku_d = cuda.to_device(ku)
kv_d = cuda.to_device(kv)

# Temporal data
ts = start_time
tf = final_time
dt = time_step_size
step = 0
nstep = int((tf - ts) / dt) + 1

t = start_time

if rank == 0:
    print("Solve!", flush=True)

t_solve = Timer("Solve!")
t_solve.start()
while t < tf:
    dt = min(dt, tf - t)

    # Store solution at start of time step
    copy[num_blocks_dofs, threadsperblock_dofs](u_d, u0_d)
    copy[num_blocks_dofs, threadsperblock_dofs](v_d, v0_d)

    # Runge-Kutta step
    for i in range(n_rk):
        copy[num_blocks_dofs, threadsperblock_dofs](u0_d, un_d)
        copy[num_blocks_dofs, threadsperblock_dofs](v0_d, vn_d)

        axpy[num_blocks_dofs, threadsperblock_dofs](a_runge[i] * dt, ku_d, un_d)
        axpy[num_blocks_dofs, threadsperblock_dofs](a_runge[i] * dt, kv_d, vn_d)

        tn = t + c_runge[i] * dt

        # ----------- #
        # Evaluate f0 #
        # ----------- #

        copy[num_blocks_dofs, threadsperblock_dofs](vn_d, ku_d)

        # ----------- #
        # Evaluate f1 #
        # ----------- #

        # Compute window function
        T = 1 / source_frequency
        alpha = 4.0

        if t < T * alpha:
            window = 0.5 * (1.0 - np.cos(source_frequency * np.pi * t / alpha))
            dwindow = (
                0.5
                * np.pi
                * source_frequency
                / alpha
                * np.sin(source_frequency * np.pi * t / alpha)
            )
        else:
            window = 1.0
            dwindow = 0.0

        # Update source function
        g_vals = (
            window
            * source_amplitude
            * angular_frequency
            / speed_of_sound
            * np.cos(angular_frequency * t)
        )
        dg_vals = (
            dwindow
            * source_amplitude
            * angular_frequency
            / speed_of_sound
            * np.cos(angular_frequency * t)
            - window
            * source_amplitude
            * angular_frequency**2
            / speed_of_sound
            * np.sin(angular_frequency * t)
        )

        fill[num_blocks_dofs, threadsperblock_dofs](g_vals, g_d)
        fill[num_blocks_dofs, threadsperblock_dofs](dg_vals, dg_d)

        # Update fields
        copy[num_blocks_dofs, threadsperblock_dofs](un_d, u_n_d)
        copy[num_blocks_dofs, threadsperblock_dofs](vn_d, v_n_d)
        square[num_blocks_dofs, threadsperblock_dofs](vn_d, w_n_d)
        scatter_fwd(u_n_d)
        scatter_fwd(v_n_d)
        scatter_fwd(w_n_d)

        # Assemble LHS (Unsteady)
        fill[num_blocks_dofs, threadsperblock_dofs](0.0, m_d)
        mass_operator[num_blocks_m, threadsperblock_m](
            u_n_d, cell_coeff2_d, m_d, detJ_d, dofmap_d
        )
        scatter_rev(m_d)

        axpy[num_blocks_dofs, threadsperblock_dofs](1.0, m0_d, m_d)

        # Assemble RHS
        fill[num_blocks_dofs, threadsperblock_dofs](0.0, b_d)

        stiff_operator_cell[num_blocks_s, threadsperblock_s](
            u_n_d, cell_coeff3_d, b_d, G_d, dofmap_d, dphi_1D_d
        )
        stiff_operator_cell[num_blocks_s, threadsperblock_s](
            v_n_d, cell_coeff4_d, b_d, G_d, dofmap_d, dphi_1D_d
        )
        mass_operator[num_blocks_m, threadsperblock_m](
            w_n_d, cell_coeff5_d, b_d, detJ_d, dofmap_d
        )
        if bfacet_dofmap1.any():
            mass_operator[num_blocks_f1, threadsperblock_m](
                g_d, facet_coeff1_1_d, b_d, detJ_f1_d, bfacet_dofmap1_d
            )
            mass_operator[num_blocks_f1, threadsperblock_m](
                dg_d, facet_coeff2_1_d, b_d, detJ_f1_d, bfacet_dofmap1_d
            )
        if bfacet_dofmap2.any():
            mass_operator[num_blocks_f2, threadsperblock_m](
                v_n_d, facet_coeff2_2_d, b_d, detJ_f2_d, bfacet_dofmap2_d
            )
        scatter_rev(b_d)

        # Solve
        pointwise_divide[num_blocks_dofs, threadsperblock_dofs](b_d, m_d, kv_d)

        # --------------- #
        # Update solution #
        # --------------- #

        axpy[num_blocks_dofs, threadsperblock_dofs](b_runge[i] * dt, ku_d, u_d)
        axpy[num_blocks_dofs, threadsperblock_dofs](b_runge[i] * dt, kv_d, v_d)

    # Update time
    t += dt
    step += 1

    if step % 100 == 0 and rank == 0:
        print(f"t: {t:5.5},\t Steps: {step}/{nstep}, \t u[0] = {u_n[0]}", flush=True)

cuda.synchronize()
scatter_fwd(u_n_d)
u_n_d.copy_to_host(u_n)
v_n_d.copy_to_host(v_n)
t_solve.stop()

print(f"{rank}: {u_n}")
if rank == 0:
    print(f"Solve time: {t_solve.elapsed()[0]}")
    print(f"Solve time per step: {t_solve.elapsed()[0]/nstep}")

# --------------- #
# Output solution #
# --------------- #

# Evaluate function
u_n_eval = u_n_.eval(x_eval, cell_eval)

try:
    data[:, 2] = u_n_eval.flatten()
except:
    pass

# Write evaluation from each process into a single file
comm.Barrier()

for i in range(comm.size):
    if rank == i:
        fname = f"/home/user/adeeb/data/pressure_field_nonlinear_nproc{comm.size}.txt"
        f_data = open(fname, "a")
        np.savetxt(f_data, data, fmt='%.8f', delimiter=",")
        f_data.close()

    comm.Barrier()

# --------------------------------------------------------------
