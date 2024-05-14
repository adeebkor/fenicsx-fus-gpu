#
# Linear wave
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
from dolfinx.common import list_timings, Reduction, Timer, TimingType
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
)
from utils import facet_integration_domain

float_type = np.float64

# Source parameters
source_frequency = 0.5e6
source_amplitude = 60000.0
period = 1.0 / source_frequency
angular_frequency = 2.0 * np.pi * source_frequency

# Material parameters
speed_of_sound = 1500.0
density = 1000.0

# Domain parameters
domain_length = 0.12

# FE parameters
basis_degree = 4
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
num_element = int(2 * num_of_waves)

# Create mesh
mesh = create_box(
    MPI.COMM_WORLD,
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
MPI.COMM_WORLD.Allreduce(hmin, mesh_size, op=MPI.MIN)

# Mesh geometry data
x_dofs = mesh.geometry.dofmap
x_g = mesh.geometry.x
cell_type = mesh.basix_cell()

# Temporal parameters
CFL = 0.65
time_step_size = CFL * mesh_size / (speed_of_sound * basis_degree**2)
step_per_period = int(period / time_step_size) + 1
time_step_size = period / step_per_period
start_time = 0.0
final_time = domain_length / speed_of_sound + 2.0 / source_frequency
number_of_step = (final_time - start_time) / time_step_size + 1

if MPI.COMM_WORLD.rank == 0:
    print(f"Number of steps: {int(number_of_step)}", flush=True)

# Define a DG function space for the material parameters
V_DG = functionspace(mesh, ("DG", 0))

c0 = Function(V_DG, dtype=float_type)
c0.x.array[:] = speed_of_sound
c0_ = c0.x.array

rho0 = Function(V_DG, dtype=float_type)
rho0.x.array[:] = density
rho0_ = rho0.x.array

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
ndofs = V.dofmap.index_map.size_local

if MPI.COMM_WORLD.rank == 0:
    print(f"Number of degrees-of-freedom: {ndofs}")

# Define functions
u_t_ = Function(V, dtype=float_type)
u_n_ = Function(V, dtype=float_type)
v_n_ = Function(V, dtype=float_type)

# Get the numpy array
u_t = u_t_.x.array
g = u_t.copy()
u_n = u_n_.x.array
v_n = v_n_.x.array

u_n[:] = 0.0
v_n[:] = 0.0

# Create LHS and RHS vector
m_ = la.vector(V.dofmap.index_map, dtype=float_type)
b_ = la.vector(V.dofmap.index_map, dtype=float_type)

# Get array for LHS and RHS vector
m = m_.array
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
if MPI.COMM_WORLD.rank == 0:
    print("Computing scaled Jacobian determinant (cell)", flush=True)

detJ = np.zeros((num_cells, nq), dtype=float_type)
compute_scaled_jacobian_determinant(detJ, (x_dofs, x_g), num_cells, dphi, wts)

# Compute scaled geometrical factor (J^{-T}J_{-1})
if MPI.COMM_WORLD.rank == 0:
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
if MPI.COMM_WORLD.rank == 0:
    print("Computing scaled Jacobian determinant (source facets)", flush=True)

detJ_f1 = np.zeros((boundary_data1.shape[0], nq_f), dtype=float_type)
compute_boundary_facets_scaled_jacobian_determinant(
    detJ_f1, (x_dofs, x_g), boundary_data1, dphi_f, wts_f
)

# Compute scaled Jacobian determinant (absorbing facets)
if MPI.COMM_WORLD.rank == 0:
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
cell_coeff2 = - 1.0 / rho0_

facet_coeff1 = np.zeros((bfacet_dofmap1.shape[0]), dtype=float_type)
for i, (cell, local_facet) in enumerate(boundary_data1):
    facet_coeff1[i] = 1.0 / rho0_[cell]

facet_coeff2 = np.zeros((bfacet_dofmap2.shape[0]), dtype=float_type)
for i, (cell, local_facet) in enumerate(boundary_data2):
    facet_coeff2[i] = - 1.0 / rho0_[cell] / c0_[cell]

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
dofmap_d = cuda.to_device(dofmap)
detJ_d = cuda.to_device(detJ)
G_d = cuda.to_device(G)
dphi_1D_d = cuda.to_device(dphi_1D)

# Boundary facet operator
facet_coeff1_d = cuda.to_device(facet_coeff1)
facet_coeff2_d = cuda.to_device(facet_coeff2)
bfacet_dofmap1_d = cuda.to_device(bfacet_dofmap1)
bfacet_dofmap2_d = cuda.to_device(bfacet_dofmap2)
detJ_f1_d = cuda.to_device(detJ_f1)
detJ_f2_d = cuda.to_device(detJ_f2)

# Arrays
u_t_d = cuda.to_device(u_t)
g_d = cuda.to_device(g)
u_n_d = cuda.to_device(u_n)
v_n_d = cuda.to_device(v_n)
m_d = cuda.to_device(m)
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
pointwise_divide[num_blocks_dofs, threadsperblock_dofs](m_d, b_d, u_t_d)

# ------------ #
# Assemble LHS #
# ------------ #

fill[num_blocks_dofs, threadsperblock_dofs](1.0, u_t_d)

cuda.synchronize()
with Timer("~ assemble lhs"):
    fill[num_blocks_dofs, threadsperblock_dofs](0.0, m_d)

    with Timer("~ m0 assembly"):
        mass_operator[num_blocks_m, threadsperblock_m](
            u_t_d, cell_coeff1_d, m_d, detJ_d, dofmap_d
        )
        cuda.synchronize()

    cuda.synchronize()

# ---------------------- #
# Set initial conditions #
# ---------------------- #

fill[num_blocks_dofs, threadsperblock_dofs](0.0, u_n_d)
fill[num_blocks_dofs, threadsperblock_dofs](0.0, v_n_d)

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

if MPI.COMM_WORLD.rank == 0:
    print("Solve!", flush=True)

t_solve = Timer("Solve!")
t_solve.start()
while t < tf:
    dt = min(dt, tf - t)

    # Store solution at start of time step
    cuda.synchronize()
    with Timer("~ RK (copy ext)"):
        copy[num_blocks_dofs, threadsperblock_dofs](u_d, u0_d)
        copy[num_blocks_dofs, threadsperblock_dofs](v_d, v0_d)
        cuda.synchronize()

    # Runge-Kutta step
    for i in range(n_rk):
        with Timer("~ RK (copy int)"):
            copy[num_blocks_dofs, threadsperblock_dofs](u0_d, un_d)
            copy[num_blocks_dofs, threadsperblock_dofs](v0_d, vn_d)
            cuda.synchronize()

        with Timer("~ RK (axpy a)"):
            axpy[num_blocks_dofs, threadsperblock_dofs](a_runge[i] * dt, ku_d, un_d)
            axpy[num_blocks_dofs, threadsperblock_dofs](a_runge[i] * dt, kv_d, vn_d)
            cuda.synchronize()

        tn = t + c_runge[i] * dt

        # ----------- #
        # Evaluate f0 #
        # ----------- #

        with Timer("~ RK (f0)"):
            copy[num_blocks_dofs, threadsperblock_dofs](vn_d, ku_d)
            cuda.synchronize()

        # ----------- #
        # Evaluate f1 #
        # ----------- #

        with Timer("~ RK (f1)"):
            # Compute window function
            T = 1 / source_frequency
            alpha = 4

            if t < T * alpha:
                window = 0.5 * (1 - np.cos(source_frequency * np.pi * t / alpha))
            else:
                window = 1.0

            # Update source function
            with Timer("~ F1 (update source)"):
                g_vals = (
                    window
                    * source_amplitude
                    * angular_frequency
                    / speed_of_sound
                    * np.cos(angular_frequency * t)
                )
                fill[num_blocks_dofs, threadsperblock_dofs](g_vals, g_d)
                cuda.synchronize()

            # Update fields
            with Timer("~ F1 (update field)"):
                copy[num_blocks_dofs, threadsperblock_dofs](un_d, u_n_d)
                copy[num_blocks_dofs, threadsperblock_dofs](vn_d, v_n_d)
                cuda.synchronize()

            # Assemble RHS
            with Timer("~ F1 (assemble rhs)"):
                fill[num_blocks_dofs, threadsperblock_dofs](0.0, b_d)

                with Timer("~ b0 assembly"):
                    stiff_operator_cell[num_blocks_s, threadsperblock_s](
                        u_n_d, cell_coeff2_d, b_d, G_d, dofmap_d, dphi_1D_d
                    )
                    cuda.synchronize()

                with Timer("~ b facet assembly"):
                    mass_operator[num_blocks_f1, threadsperblock_m](
                        g_d, facet_coeff1_d, b_d, detJ_f1_d, bfacet_dofmap1_d
                    )
                    mass_operator[num_blocks_f2, threadsperblock_m](
                        v_n_d, facet_coeff2_d, b_d, detJ_f2_d, bfacet_dofmap2_d
                    )
                    cuda.synchronize()

            # Solve
            with Timer("~ F1 (solve)"):
                pointwise_divide[num_blocks_dofs, threadsperblock_dofs](b_d, m_d, kv_d)
                cuda.synchronize()

        # --------------- #
        # Update solution #
        # --------------- #

        with Timer("~ RK (axpy b)"):
            axpy[num_blocks_dofs, threadsperblock_dofs](b_runge[i] * dt, ku_d, u_d)
            axpy[num_blocks_dofs, threadsperblock_dofs](b_runge[i] * dt, kv_d, v_d)

    # Update time
    t += dt
    step += 1

    if step % 100 == 0 and MPI.COMM_WORLD.rank == 0:
        print(f"t: {t:5.5},\t Steps: {step}/{nstep}, \t u[0] = {u_[0]}", flush=True)

u_n_d.copy_to_host(u_n)
v_n_d.copy_to_host(v_n)
t_solve.stop()

if MPI.COMM_WORLD.rank == 0:
    print(f"Solve time: {t_solve.elapsed()[0]}")
    print(f"Solve time per step: {t_solve.elapsed()[0]/nstep}")

# --------------------- #
# Output final solution #
# --------------------- #

with VTXWriter(MPI.COMM_WORLD, "output_final.bp", u_n_, "bp4") as f:
    f.write(0.0)

# ------------ #
# List timings #
# ------------ #

list_timings(MPI.COMM_WORLD, [TimingType.wall], Reduction.average)
