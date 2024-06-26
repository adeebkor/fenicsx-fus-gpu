#
# Linear wave
# - Plane wave
# - Homogenous media
# =================================
# Copyright (C) 2024 Adeeb Arif Kor


import numpy as np
import numpy.typing as npt
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
from operators import mass_operator, stiffness_operator
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
domain_length = 0.03

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
Nf = nd * nd
Nd = nd * nd * nd

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
    [cpp.mesh.h(mesh._cpp_object, tdim, np.arange(num_cells, dtype=np.int32)).min()]
)
mesh_size = np.zeros(1)
MPI.COMM_WORLD.Reduce(hmin, mesh_size, op=MPI.MIN, root=0)
MPI.COMM_WORLD.Bcast(mesh_size, root=0)

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

basix_element_tp = basix.create_tp_element(family, cell_type, basis_degree, variant)
perm = np.argsort(np.array(basix_element_tp.dof_ordering, dtype=np.int32))

# Basix element
basix_element = basix.create_element(family, cell_type, basis_degree, variant)
element = basix.ufl._BasixElement(basix_element)  # basix ufl element

# Define function space and functions
V = functionspace(mesh, element)
dofmap = V.dofmap.list[:, perm]

if MPI.COMM_WORLD.rank == 0:
    print(f"Number of degrees-of-freedom: {V.dofmap.index_map.size_global}")

# Define functions
u0 = Function(V, dtype=float_type)
u_n_ = Function(V, dtype=float_type)
v_n_ = Function(V, dtype=float_type)

# Get the numpy array
u = u0.x.array
g = u.copy()
u_n = u_n_.x.array
v_n = v_n_.x.array

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
cell_coeff2 = -1.0 / rho0_

facet_coeff1 = np.zeros((bfacet_dofmap1.shape[0]), dtype=float_type)
for i, (cell, local_facet) in enumerate(boundary_data1):
    facet_coeff1[i] = 1.0 / rho0_[cell]

facet_coeff2 = np.zeros((bfacet_dofmap2.shape[0]), dtype=float_type)
for i, (cell, local_facet) in enumerate(boundary_data2):
    facet_coeff2[i] = -1.0 / rho0_[cell] / c0_[cell]

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
nd = dphi_1D.shape[1]
dphi_1D = dphi_1D.flatten()

# ---------------- #
# Define operators #
# ---------------- #

mass_operator_cell = mass_operator(Nd, float_type)
stiff_operator_cell = stiffness_operator(basis_degree, dphi_1D, float_type)
mass_operator_bfacet = mass_operator(Nf, float_type)

# Run once to jit compile operators
mass_operator_cell(u, cell_coeff1, m, detJ, dofmap)
stiff_operator_cell(u_n, cell_coeff2, b, G, dofmap)
mass_operator_bfacet(g, facet_coeff1, b, detJ_f1, bfacet_dofmap1)

# ------------ #
# Assemble LHS #
# ------------ #

u[:] = 1.0

with Timer("~ assemble lhs"):
    m[:] = 0.0

    with Timer("~ m0 assembly"):
        mass_operator_cell(u, cell_coeff1, m, detJ, dofmap)

    m_.scatter_reverse(la.InsertMode.add)

# ---------------------- #
# Set initial conditions #
# ---------------------- #

u_n[:] = 0.0
v_n[:] = 0.0

# ------------------ #
# RK slope functions #
# ------------------ #


def f(
    t: float,
    u: npt.NDArray[np.floating],
    v: npt.NDArray[np.floating],
    result: npt.NDArray[np.floating],
):
    """
    Evaluate dv/dt = f1(t, u, v)

    Parameters
    ----------
    t : Current time, i.e. tn
    u : Current u, i.e. un
    v : Current v, i.e. vn

    Return
    ------
    result : Result, i.e. k^{v}
    """

    T = 1 / source_frequency
    alpha = 4

    if t < T * alpha:
        window = 0.5 * (1 - np.cos(source_frequency * np.pi * t / alpha))
    else:
        window = 1.0

    # Update boundary condition
    with Timer("~ F1 (update source)"):
        g[:] = (
            window
            * source_amplitude
            * angular_frequency
            / speed_of_sound
            * np.cos(angular_frequency * t)
        )

    # Update fields
    with Timer("~ F1 (update field)"):
        u_n[:] = u[:]
        u_n_.x.scatter_forward()
        v_n[:] = v[:]
        v_n_.x.scatter_forward()

    # Assemble RHS
    with Timer("~ F1 (assemble rhs)"):
        b[:] = 0.0

        with Timer("~ b0 assembly"):
            stiff_operator_cell(u_n, cell_coeff2, b, G, dofmap)

        with Timer("~ b facet assembly"):
            mass_operator_bfacet(g, facet_coeff1, b, detJ_f1, bfacet_dofmap1)
            mass_operator_bfacet(v_n, facet_coeff2, b, detJ_f2, bfacet_dofmap2)

        b_.scatter_reverse(la.InsertMode.add)

    # Solve
    with Timer("~ F1 (solve)"):
        result[:] = b[:] / m[:]


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
    with Timer("~ RK (copy ext)"):
        u0[:] = u_[:]
        v0[:] = v_[:]

    # Runge-Kutta step
    for i in range(n_rk):
        with Timer("~ RK (copy int)"):
            un[:] = u0[:]
            vn[:] = v0[:]

        with Timer("~ RK (axpy a)"):
            un += a_runge[i] * dt * ku
            vn += a_runge[i] * dt * kv

        tn = t + c_runge[i] * dt

        # Evaluate slopes
        with Timer("~ RK (f0)"):
            ku[:] = vn[:]

        with Timer("~ RK (f1)"):
            f(tn, un, vn, kv)

        # Update solution
        with Timer("~ RK (axpy b)"):
            u_ += b_runge[i] * dt * ku
            v_ += b_runge[i] * dt * kv

    # Update time
    t += dt
    step += 1

    if step % 100 == 0 and MPI.COMM_WORLD.rank == 0:
        print(f"t: {t:5.5},\t Steps: {step}/{nstep}, \t u[0] = {u_[0]}", flush=True)

u_n_.x.scatter_forward()
v_n_.x.scatter_forward()
u_n[:] = u_[:]
v_n[:] = v_[:]
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
