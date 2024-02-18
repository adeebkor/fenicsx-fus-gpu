#
# Linear wave
# - Heterogenous media
# - Mass lumped scheme
# =================================
# Copyright (C) 2024 Adeeb Arif Kor

from mpi4py import MPI
import numpy as np

import basix
import basix.ufl

from dolfinx import cpp, la
from dolfinx.fem import assemble_vector, form, functionspace, Function
from dolfinx.io import XDMFFile, VTXWriter
from ufl import dx, grad, inner, Measure, TestFunction

# Source parameters
source_frequency = 0.5e6  # Hz
source_amplitude = 60000.0  # Pa
period = 1.0 / source_frequency  # s
angular_frequency = 2.0 * np.pi * source_frequency  # rad/s

# Material parameters
speed_of_sound = 1500.0  # m/s
density = 1000.0  # kg/m^3

# Domain parameter
domain_length = 0.12  # m

# FE parameters
degree_of_basis = 4

# Read mesh and mesh tags
with XDMFFile (MPI.COMM_WORLD, "mesh.xdmf", "r") as fmesh:
    mesh_name = "planar_3d_0"
    mesh = fmesh.read_mesh(name=f"{mesh_name}")
    tdim = mesh.topology.dim
    mt_cell = fmesh.read_meshtags(mesh, name=f"{mesh_name}_cells")
    mesh.topology.create_connectivity(tdim-1, tdim)
    mt_facet = fmesh.read_meshtags(mesh, name=f"{mesh_name}_facets")

# Boundary facets
ds = Measure('ds', subdomain_data=mt_facet, domain=mesh)

# Mesh parameters
num_cell = mesh.topology.index_map(tdim).size_local
hmin = np.array([cpp.mesh.h(
    mesh._cpp_object, tdim, np.arange(num_cell, dtype=np.int32)).min()])
mesh_size = np.zeros(1)
MPI.COMM_WORLD.Reduce(hmin, mesh_size, op=MPI.MIN, root=0)
MPI.COMM_WORLD.Bcast(mesh_size, root=0)

# Define a DG function space for the material parameters
V_DG = functionspace(mesh, ("DG", 0))
c0 = Function(V_DG)
c0.x.array[:] = speed_of_sound

rho0 = Function(V_DG)
rho0.x.array[:] = density

# Temporal parameters
CFL = 0.65
time_step_size = CFL * mesh_size / (speed_of_sound * degree_of_basis**2)
step_per_period = int(period / time_step_size) + 1
time_step_size = period / step_per_period
start_time = 0.0
final_time = domain_length / speed_of_sound + 8.0 / source_frequency
number_of_step = (final_time - start_time) / time_step_size + 1

# Define finite element and function space
family = basix.ElementFamily.P
variant = basix.LagrangeVariant.gll_warped
cell_type = mesh.basix_cell()

element = basix.ufl.element(family, cell_type, degree_of_basis, variant)
V = functionspace(mesh, element)

# Define functions
v = TestFunction(V)
u = Function(V)
g = Function(V)
u_n = Function(V)
v_n = Function(V)

# Quadrature parameters
qd = {"2": 3, "3": 4, "4": 6, "5": 8, "6": 10, "7": 12, "8": 14,
      "9": 16, "10": 18}
md = {"quadrature_rule": "GLL",
      "quadrature_degree": qd[str(degree_of_basis)]}

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


def f0(t: float, u: la.Vector, v: la.Vector, result: la.Vector):
    """
    Evaluate du/dt = f0(t, u, v)

    Parameters
    ----------
    t : Current time, i.e. tn
    u : Current u, i.e. un
    v : Current v, i.e. vn

    Return
    ------
    result : Result, i.e. k^{u}
    """

    result.array[:] = v.array[:]


def f1(t: float, u: la.Vector, v: la.Vector, result: la.Vector):
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
        f0(tn, un, vn, result=ku)
        f1(tn, un, vn, result=kv)

        # Update solution
        u_.array[:] += b_runge[i] * dt * ku.array[:]
        v_.array[:] += b_runge[i] * dt * kv.array[:]

    # Update time
    t += dt
    step += 1

    if step % 100 == 0 and MPI.COMM_WORLD.rank == 0:
        print(f"t: {t:5.5},\t Steps: {step}/{nstep}", flush=True)

    u_.scatter_forward()
    v_.scatter_forward()
    u_n.x.array[:] = u_.array[:]
    v_n.x.array[:] = v_.array[:]

# --------------------- #
# Output final solution #
# --------------------- #
with VTXWriter(MPI.COMM_WORLD, "output_final.bp", u_n, "bp4") as f:
    f.write(0.0)
