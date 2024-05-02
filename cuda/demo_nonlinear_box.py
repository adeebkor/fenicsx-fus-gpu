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
from utils import facet_integration_domain, compute_diffusivity_of_sound

float_type = np.float64

# Source parameters
source_frequency = 0.5e6
source_amplitude = 60000.0
period = 1.0 / source_frequency
angular_frequency = 2.0 * np.pi * source_frequency

# Material parameters
speed_of_sound = 1500.0
density = 1000
nonlinear_coefficient = 100.0
attenuation_coefficient_dB = 50
attenuation_coefficient_Np = attenuation_coefficient_dB / 20 * np.log(10)
diffusivity_of_sound = compute_diffusivity_of_sound(
    angular_frequency, speed_of_sound, attenuation_coefficient_dB
)
