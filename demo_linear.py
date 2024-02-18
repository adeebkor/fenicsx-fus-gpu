#
# Linear wave
# - Heterogenous media
# =================================
# Copyright (C) 2024 Adeeb Arif Kor

import numpy as np

# Source parameters
source_frequency = 0.5e6  # Hz
source_amplitude = 60000.0  # Pa
period = 1.0 / source_frequency  # s
angular_frequency = 2.0 * np.pi * source_frequency  # rad/s

# Material parameters
speed_of_sound = 1500.0  # m/s
density = 1000.0  # kg/m^3

# Domain parameters
domain_length = 0.12  # m
