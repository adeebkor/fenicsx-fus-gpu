#
# Linear wave
# - Plane wave
# - Homogenous media
# =================================
# Copyright (C) 2024 Adeeb Arif Kor

import time

import numpy as np
import numpy.typing as npt
import numba
from mpi4py import MPI

import basix
import basix.ufl
from dolfinx import cpp, la
from dolfinx.fem import assemble_scalar, assemble_vector, form, functionspace, Function
from dolfinx.io import VTXWriter
from dolfinx.mesh import create_box, locate_entities_boundary, CellType, GhostMode, meshtags
from ufl import dx, grad, inner, Measure, TestFunction

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
domain_length = 0.015

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

# Mesh parameters
lmbda = speed_of_sound / source_frequency
num_of_waves = domain_length / lmbda
num_element = int(2 * num_of_waves)

# Create mesh
mesh = create_box(
    MPI.COMM_WORLD,
    ((0., 0., 0.), (domain_length, domain_length, domain_length)),
    (num_element, num_element, num_element),
    cell_type=CellType.hexahedron,
    ghost_mode=GhostMode.none,
    dtype=float_type
)

# Mesh data
tdim = mesh.topology.dim
gdim = mesh.geometry.dim
num_cells = mesh.topology.index_map(tdim).size_local
hmin = np.array([cpp.mesh.h(
    mesh._cpp_object, tdim, np.arange(num_cells, dtype=np.int32)).min()])
mesh_size = np.zeros(1)
MPI.COMM_WORLD.Reduce(hmin, mesh_size, op=MPI.MIN, root=0)
MPI.COMM_WORLD.Bcast(mesh_size, root=0)

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

# Tag boundary facets
boundary_facets1 = locate_entities_boundary(
    mesh, mesh.topology.dim-1, lambda x: np.isclose(x[0], np.finfo(float).eps)
)

boundary_facets2 = locate_entities_boundary(
    mesh, mesh.topology.dim-1, lambda x: np.isclose(x[0], domain_length)
)

marked_facets = np.hstack([boundary_facets1, boundary_facets2])
marked_values = np.hstack([np.full_like(boundary_facets1, 1),
                           np.full_like(boundary_facets2, 2)])
sorted_facets = np.argsort(marked_facets)
facet_tag = meshtags(mesh, tdim-1, marked_facets[sorted_facets],
                     marked_values[sorted_facets])

ds = Measure('ds', subdomain_data=facet_tag, domain=mesh)

# Define finite element and function space
family = basix.ElementFamily.P
variant = basix.LagrangeVariant.gll_warped
cell_type = mesh.basix_cell()

element = basix.ufl.element(family, cell_type, basis_degree, variant)
V = functionspace(mesh, element)

if MPI.COMM_WORLD.rank == 0:
    print(f"Number of degrees-of-freedom: {V.dofmap.index_map.size_global}")

# Define functions
v = TestFunction(V)
u = Function(V)
g = Function(V)
u_n = Function(V)
v_n = Function(V)

# Quadrature parameters
md = {"quadrature_rule": "GLL",
      "quadrature_degree": quadrature_degree[basis_degree]}

# Define forms
u.x.array[:] = 1.0
a = form(
    inner(u/rho0/c0/c0, v) * dx(metadata=md)
)
m = assemble_vector(a)
m.scatter_reverse(la.InsertMode.add)

L = form(
    - inner(1.0/rho0*grad(u_n), grad(v)) * dx(metadata=md)
    + inner(1.0/rho0*g, v) * ds(1, metadata=md)
    - inner(1.0/rho0/c0*v_n, v) * ds(2, metadata=md)
)
b = assemble_vector(L)
b.scatter_reverse(la.InsertMode.add)

# Set initial values for u_n and v_n
u_n.x.array[:] = 0.0
v_n.x.array[:] = 0.0

# ------------------ #
# RK slope functions #
# ------------------ #

def f(t: float, u: la.Vector, v: la.Vector, result: la.Vector):
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
    g.x.array[:] = window * source_amplitude * angular_frequency / \
        speed_of_sound * np.cos(angular_frequency * t)

    # Update fields
    u_n.x.array[:] = u.array[:]
    u_n.x.scatter_forward()
    v_n.x.array[:] = v.array[:]
    v_n.x.scatter_forward()

    # Assemble RHS
    b.array[:] = 0
    assemble_vector(b.array, L)
    b.scatter_reverse(la.InsertMode.add)

    # Solve
    result.array[:] = b.array[:] / m.array[:]

# --------------- #
# Solve using RK4 #
# --------------- #

# Runge-Kutta data
n_rk = 4
a_runge = np.array([0.0, 0.5, 0.5, 1.0])
b_runge = np.array([1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0])
c_runge = np.array([0.0, 0.5, 0.5, 1.0])

# Solution vectors at time step n
u_ = la.vector(V.dofmap.index_map)
v_ = la.vector(V.dofmap.index_map)

# Solution vectors at intermediate time step
un = la.vector(V.dofmap.index_map)
vn = la.vector(V.dofmap.index_map)

# Solution vectors at start of time step
u0 = la.vector(V.dofmap.index_map)
v0 = la.vector(V.dofmap.index_map)

# Slope vectors at intermediatte time step
ku = la.vector(V.dofmap.index_map)
kv = la.vector(V.dofmap.index_map)

# Temporal data
ts = start_time
tf = final_time
dt = time_step_size
step = 0
nstep = int((tf - ts) / dt) + 1

t = start_time
tic = time.time()
while t < tf:
    dt = min(dt, tf-t)

    # Store solution at start of time step
    u0.array[:] = u_.array[:]
    v0.array[:] = v_.array[:]

    # Runge-Kutta step
    for i in range(n_rk):
        un.array[:] = u0.array[:]
        vn.array[:] = v0.array[:]

        un.array[:] += a_runge[i] * dt * ku.array[:]
        vn.array[:] += a_runge[i] * dt * kv.array[:]

        tn = t + c_runge[i] * dt

        # Evaluate slopes
        ku.array[:] = vn.array[:]
        f(tn, un, vn, result=kv)

        # Update solution
        u_.array[:] += b_runge[i] * dt * ku.array[:]
        v_.array[:] += b_runge[i] * dt * kv.array[:]

    # Update time
    t += dt
    step += 1

    if step % 10 == 0 and MPI.COMM_WORLD.rank == 0:
        print(f"t: {t:5.5},\t Steps: {step}/{nstep}, \t u[0] = {u_.array[0]}", flush=True)

u_.scatter_forward()
v_.scatter_forward()
u_n.x.array[:] = u_.array[:]
v_n.x.array[:] = v_.array[:]

toc = time.time()
elapsed = toc - tic

if MPI.COMM_WORLD.rank == 0:
    print(f"Solve time: {elapsed}")
    print(f"Solve time per step: {elapsed/nstep}")

# --------------------- #
# Output final solution #
# --------------------- #

with VTXWriter(MPI.COMM_WORLD, "output_final.bp", u_n, "bp4") as f:
    f.write(0.0)
