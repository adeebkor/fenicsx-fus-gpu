#
# Linear wave with attenuation
# - Heterogenous media
# - Mass lumped scheme
# =================================
# Copyright (C) 2024 Adeeb Arif Kor

from mpi4py import MPI
import numpy as np

import basix
import basix.ufl

from dolfinx import fem, la, io, mesh
import ufl

# ------------- #
# Problem setup #
# ------------- #

# Source parameters
freq0 = 10  # source frequency (Hz)
w0 = 2.0 * np.pi * freq0  # angular frequency (rad/s)
u0 = 1  # source speed (m/s)

# Material parameters
c0 = 1.5  # speed of sound (m/s)
rho0 = 1.0  # density (kg/m^3)
alphadB = 5  # attenuation of sound (dB/m)
delta0 = 2.0 * alphadB / 20 * np.log(10) * c0**3 / w0**2

# Domain parameter
L_d = 1.0  # domain length (m)

# Physical parameters
p0 = rho0*c0*u0  # pressure amplitude (Pa)
lmbda = c0/freq0  # wavelength (m)

# Mesh parameters
epw = 2  # elements per wavelength
nw = L_d / lmbda  # number of waves
nx = int(epw * nw + 1)  # number of elements
h = L_d / nx

# Basis degree
p = 4

# Generate mesh
msh = mesh.create_rectangle(
    MPI.COMM_WORLD,
    ((0.0, 0.0), (L_d, L_d)),
    (nx, nx),
    mesh.CellType.quadrilateral)

# Tag boundaries
tdim = msh.topology.dim

facets0 = mesh.locate_entities_boundary(
    msh, tdim-1, lambda x: x[0] < np.finfo(float).eps)
facets1 = mesh.locate_entities_boundary(
    msh, tdim-1, lambda x: x[0] > L_d - np.finfo(float).eps)

f_indices, f_pos = np.unique(np.hstack((facets0, facets1)), return_index=True)
f_values = np.hstack((np.full(facets0.shape, 1, np.intc),
                      np.full(facets1.shape, 2, np.intc)))
ft = mesh.meshtags(msh, tdim-1, f_indices, f_values[f_pos])

# Tag cells
cells0 = mesh.locate_entities(msh, tdim, lambda x: x[0] <= L_d/2.0)
cells1 = mesh.locate_entities(msh, tdim, lambda x: x[0] >= L_d/2.0)

# Define DG function for physical parameters
V_DG = fem.functionspace(msh, ("DG", 0))
c = fem.Function(V_DG)
c.x.array[:] = c0
c.x.array[cells1] = 2.8

rho = fem.Function(V_DG)
rho.x.array[:] = rho0
rho.x.array[cells1] = 1.85

delta = fem.Function(V_DG)
delta.x.array[:] = 0

# Temporal parameters
tstart = 0.0
tfinal = L_d / c0 + 8 / freq0

CFL = 0.5
dt = CFL * h / (c0 * p**2)

# Boundary facets
ds = ufl.Measure('ds', subdomain_data=ft, domain=msh)

# Define finite element and function space
cell_type = basix.cell.string_to_type(msh.ufl_cell().cellname())
element = basix.ufl.element(
    basix.ElementFamily.P, cell_type, p,
    basix.LagrangeVariant.gll_warped)
V = fem.functionspace(msh, element)

# Define functions
v = ufl.TestFunction(V)
u = fem.Function(V)
g = fem.Function(V)
dg = fem.Function(V)
u_n = fem.Function(V)
v_n = fem.Function(V)

# Quadrature parameters
qd = {"2": 3, "3": 4, "4": 6, "5": 8, "6": 10, "7": 12, "8": 14,
      "9": 16, "10": 18}
md = {"quadrature_rule": "GLL",
      "quadrature_degree": qd[str(p)]}

# Define forms
u.x.array[:] = 1.0
a = fem.form(
    ufl.inner(u/rho/c0/c0, v) * ufl.dx(metadata=md)
    + ufl.inner(delta/rho/c0/c0/c0*u, v) * ds(2, metadata=md))
m = fem.assemble_vector(a)
m.scatter_reverse(la.InsertMode.add)

L = fem.form(
    - ufl.inner(1.0/rho*ufl.grad(u_n), ufl.grad(v)) * ufl.dx(metadata=md)
    + ufl.inner(1.0/rho*g, v) * ds(1, metadata=md)
    - ufl.inner(1.0/rho/c0*v_n, v) * ds(2, metadata=md)
    - ufl.inner(delta/rho/c0/c0*ufl.grad(v_n), ufl.grad(v))
    * ufl.dx(metadata=md)
    + ufl.inner(delta/rho/c0/c0*dg, v) * ds(1, metadata=md))
b = fem.assemble_vector(L)
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
    T = 1 / freq0
    alpha = 4

    if t < T * alpha:
        window = 0.5 * (1 - np.cos(freq0 * np.pi * t / alpha))
        dwindow = 0.5 * np.pi * freq0 / alpha * np.sin(
            freq0 * np.pi * t / alpha)
    else:
        window = 1.0
        dwindow = 0.0

    # Update boundary condition
    g.x.array[:] = window * p0 * w0 / c0 * np.cos(w0 * t)
    dg.x.array[:] = dwindow * p0 * w0 / c0 * np.cos(w0 * t) \
        - window * p0 * w0**2 / c0 * np.sin(w0 * t)

    # Update fields
    u_n.x.array[:] = u.array[:]
    u_n.x.scatter_forward()
    v_n.x.array[:] = v.array[:]
    v_n.x.scatter_forward()

    # Assemble RHS
    b.array[:] = 0
    fem.assemble_vector(b.array, L)
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
t = tstart
step = 0
nstep = int((tfinal - tstart) / dt) + 1

while t < tfinal:
    dt = min(dt, tfinal-t)

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
with io.VTXWriter(msh.comm, "output_final.bp", u_n, "bp4") as f:
    f.write(0.0)
