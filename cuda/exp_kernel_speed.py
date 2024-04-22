#
# .. _exp_kernel_speed:
#
# Compare the performance of the stiffness kernel between different global DOF
# numbering
# ============================================================================
# Copyright (C) 2024 Adeeb Arif Kor

import sys
from time import perf_counter_ns

import numpy as np
from mpi4py import MPI

import numba.cuda as cuda

import basix
import basix.ufl
from dolfinx.fem import functionspace, Function
from dolfinx.mesh import create_box, CellType, GhostMode

from operators import stiffness_operator

float_type = np.dtype(sys.argv[1]).type

assert (float_type is np.float32) or (float_type is np.float64)

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
basis_degree = int(sys.argv[2])
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
element_per_wavelength = float(sys.argv[3])
num_element = int(element_per_wavelength * num_of_waves)

# Create mesh
mesh = create_box(
    MPI.COMM_WORLD,
    ((0.0, 0.0, 0.0), (domain_length, domain_length, domain_length)),
    (num_element, num_element, num_element),
    cell_type=CellType.hexahedron,
    ghost_mode=GhostMode.none,
    dtype=float_type,
)

# Mesh geometry data
x_dofs = mesh.geometry.dofmap
x_g = mesh.geometry.x
cell_type = mesh.basix_cell()

# Mesh data
tdim = mesh.topology.dim
gdim = mesh.geometry.dim
num_cells = mesh.topology.index_map(tdim).size_local

# Element type
family = basix.ElementFamily.P
variant = basix.LagrangeVariant.gll_warped

# Tensor product element
basix_element_tp = basix.create_tp_element(family, cell_type, basis_degree, variant)
perm = np.argsort(np.array(basix_element_tp.dof_ordering, dtype=np.int32))
element_tp = basix.ufl._BasixElement(basix_element_tp)

V_tp = functionspace(mesh, element_tp)
dofmap_tp = V_tp.dofmap.list
ndofs = V_tp.dofmap.index_map.size_local

if MPI.COMM_WORLD.rank == 0:
    print(f"Number of degrees-of-freedom: {ndofs}", flush=True)

# Basix element
basix_element = basix.create_element(family, cell_type, basis_degree, variant)
element = basix.ufl._BasixElement(basix_element)

V = functionspace(mesh, element)
dofmap = V.dofmap.list[:, perm]

# Create dummy input data
G = np.random.randn(num_cells, nd * nd * nd, 3 * (gdim - 1)).astype(float_type)
cell_constants = np.random.randn(num_cells).astype(float_type)

# Create 1D element for sum factorisation
element_1D = basix.create_element(
    basix.ElementFamily.P,
    basix.CellType.interval,
    basis_degree,
    basix.LagrangeVariant.gll_warped,
    dtype=float_type,
)
pts_1D, wts_1D = basix.quadrature.make_quadrature(
    basix.CellType.interval, quadrature_degree[basis_degree], basix.QuadratureType.gll
)
pts_1D, wts_1D = pts_1D.astype(float_type), wts_1D.astype(float_type)

table_1D = element_1D.tabulate(1, pts_1D)
dphi_1D = table_1D[1, :, :, 0]

# Create functions
u0 = Function(V_tp, dtype=float_type)
u0.interpolate(
    lambda x: 100
    * np.sin(2 * np.pi * x[0])
    * np.cos(3 * np.pi * x[1])
    * np.sin(4 * np.pi * x[2])
)

u1 = Function(V, dtype=float_type)
u1.interpolate(
    lambda x: 100
    * np.sin(2 * np.pi * x[0])
    * np.cos(3 * np.pi * x[1])
    * np.sin(4 * np.pi * x[2])
)

u0_h = u0.x.array
u1_h = u1.x.array

b0_h = np.zeros_like(u0_h, dtype=float_type)
b1_h = np.zeros_like(u1_h, dtype=float_type)

# Set the number of threads in a block
threadsperblock = (nd, nd, nd)
num_blocks = num_cells

# Allocate memory on the device
G_d = cuda.to_device(G)
cell_constants_d = cuda.to_device(cell_constants)
dphi_1D_d = cuda.to_device(dphi_1D)

dofmap_tp_d = cuda.to_device(dofmap_tp)
dofmap_d = cuda.to_device(dofmap)

u0_d = cuda.to_device(u0_h)
u1_d = cuda.to_device(u1_h)

b0_d = cuda.to_device(b0_h)
b1_d = cuda.to_device(b1_h)

# Call the stiffness operator function
stiff_operator_cell = stiffness_operator(basis_degree, float_type)
stiff_operator_cell[num_blocks, threadsperblock](
    u0_d, cell_constants_d, b0_d, G_d, dofmap_tp_d, dphi_1D_d
)
stiff_operator_cell[num_blocks, threadsperblock](
    u1_d, cell_constants_d, b1_d, G_d, dofmap_d, dphi_1D_d
)

nreps = 100

timing_stiffness = np.empty(nreps)
for rep in range(nreps):
    b1_h[:] = 0.0
    b1_d = cuda.to_device(b1_h)
    tic = perf_counter_ns()
    cuda.synchronize()
    stiff_operator_cell[num_blocks, threadsperblock](
        u1_d, cell_constants_d, b1_d, G_d, dofmap_d, dphi_1D_d
    )
    cuda.synchronize()
    toc = perf_counter_ns()
    timing_stiffness[rep] = toc - tic

timing_stiffness *= 1e-9

print(
    f"Elapsed time (basix ordering): "
    f"{timing_stiffness.mean():.7f} ± "
    f"{timing_stiffness.std():.7f} s"
)

timing_stiffness_tp = np.empty(nreps)
for rep in range(nreps):
    b0_h[:] = 0.0
    b0_d = cuda.to_device(b0_h)
    tic = perf_counter_ns()
    cuda.synchronize()
    stiff_operator_cell[num_blocks, threadsperblock](
        u0_d, cell_constants_d, b0_d, G_d, dofmap_tp_d, dphi_1D_d
    )
    cuda.synchronize()
    toc = perf_counter_ns()
    timing_stiffness_tp[rep] = toc - tic

timing_stiffness_tp *= 1e-9

print(
    f"Elapsed time (tensor product ordering): "
    f"{timing_stiffness_tp.mean():.7f} ± "
    f"{timing_stiffness_tp.std():.7f} s"
)
