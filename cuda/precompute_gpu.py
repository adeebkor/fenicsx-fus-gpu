"""
==========
Precompute
==========

This file contains the functions to precompute the geometric data that is
used by the operators.

Copyright (C) 2025 Adeeb Arif Kor
"""

import numpy as np
import numba
from numba import float32
import numba.cuda as cuda


