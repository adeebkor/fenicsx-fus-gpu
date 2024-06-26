"""
=========
Operators
=========

This file contains the kernels for vector assembly. It includes the mass and
stiffness kernels.

Copyright (C) 2024 Adeeb Arif Kor
"""

import numpy as np
import numba
from numba import float32
import numba.cuda as cuda


@cuda.jit
def mass_operator(
    x: numba.types.Array,
    entity_constants: numba.types.Array,
    y: numba.types.Array,
    detJ_entity: numba.types.Array,
    entity_dofmap: numba.types.Array,
):
    """
    Compute the mass operator on some entities.

    This kernel computes the mass operator on a set of entities (cells or
    facets) defined by the entity dofmap. The mass operator is applied to the
    input array `x` and accumulates the result in the output array `y`.

    Parameters
    ----------
    x : numba.types.Array
        Input array.
    entity_constants : numba.types.Array
        Array containing the material coefficients associated with each entity.
    y : numba.types.Array
        Output array where the result of applying the mass operator is accumulated.
    detJ_entity : numba.types.Array
        Array containing the determinant of the Jacobian associated with each entity.
    entity_dofmap :  numba.types.Array
        2D array containing the local degrees of freedom (DOF) on given entities.

    Notes
    -----
    - The kernel performs atomic addition to accumulate results in the output array
      `y` to handle potential race conditions arising from concurrent writes.
    - This kernel should be launched with a sufficient number of threads and blocks
      to cover all the elements in `entity_dofmap`.
    - detJ_entity and entity_dofmap are assumed to be 2D arrays with the same shape.

    """
    thread_id = cuda.threadIdx.x  # Local thread ID (max: 1024)
    block_id = cuda.blockIdx.x  # Block ID (max: 2147483647)
    idx = thread_id + block_id * cuda.blockDim.x  # Global thread ID

    entity = idx // entity_dofmap.shape[1]
    local_dof = idx % entity_dofmap.shape[1]

    if idx < entity_dofmap.size:
        # Compute the global DOF index
        dof = entity_dofmap[entity, local_dof]

        # Compute the contribution of the current DOF to the mass operator
        value = x[dof] * detJ_entity[entity, local_dof] * entity_constants[entity]

        # Atomically add the computed value to the output array `y`
        cuda.atomic.add(y, dof, value)


def stiffness_operator(P, float_type):
    """
    Outer functions to define the compile-time constants for the stiffness
    operator.

    Parameters
    ----------
    P : basis function polynomial degree
    dphi : derivatives of the 1D basis functions
    float_type : floating-point type
    """

    n = P + 1

    @cuda.jit(lineinfo=True)
    def operator(
        x: numba.types.Array,
        entity_constants: numba.types.Array,
        y: numba.types.Array,
        G_entity: numba.types.Array,
        entity_dofmap: numba.types.Array,
        dphi: numba.types.Array,
    ):
        """
        Compute the stiffness operator on some entities.

        This kernel computes the stiffness operator on a set of entities (cells or
        facets) defined by the entity dofmap. The stiffness operator is applied to
        the input array `x` and accumulates the result in the output array `y`.

        Parameters
        ----------
        x : numba.types.Array
            Input array.
        entity_constants : numba.types.Array
            Array containing the material coefficients associated with each entity.
        y : numba.types.Array
            Output array where the result of applying the stiffness operator is accumulated.
        G_entity :
            Array containing the geometric factor associate with each entity
        entity_dofmap : numba.types.Array
            2D array containing the local degrees of freedom (DOF) on given entities.
        dphi :
            2D array containing the derivatives of the 1D basis functions
        """

        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y
        tz = cuda.threadIdx.z

        thread_id = tx * cuda.blockDim.y * cuda.blockDim.z + ty * cuda.blockDim.z + tz
        block_id = cuda.blockIdx.x

        scratch = cuda.shared.array(shape=(n, n, n), dtype=float_type)
        scratchx = cuda.shared.array(shape=(n, n, n), dtype=float_type)
        scratchy = cuda.shared.array(shape=(n, n, n), dtype=float_type)
        scratchz = cuda.shared.array(shape=(n, n, n), dtype=float_type)

        # Get dof index that this thread is computing
        dof = entity_dofmap[block_id, thread_id]

        # Gather x expression value required by this thread
        scratch[tx, ty, tz] = x[dof]
        cuda.syncthreads()

        # Apply contraction in the x-direction
        val_x = float32(0.0)
        for ix in range(n):
            val_x += dphi[tx, ix] * scratch[ix, ty, tz]

        # Apply contraction in the y-direction
        val_y = float32(0.0)
        for iy in range(n):
            val_y += dphi[ty, iy] * scratch[tx, iy, tz]

        # Apply contraction in the z-direction
        val_z = float32(0.0)
        for iz in range(n):
            val_z += dphi[tz, iz] * scratch[tx, ty, iz]

        # Apply transform
        G0 = G_entity[block_id, thread_id, 0]
        G1 = G_entity[block_id, thread_id, 1]
        G2 = G_entity[block_id, thread_id, 2]
        G3 = G_entity[block_id, thread_id, 3]
        G4 = G_entity[block_id, thread_id, 4]
        G5 = G_entity[block_id, thread_id, 5]
        coeff = entity_constants[block_id]

        fw0 = coeff * (G0 * val_x + G1 * val_y + G2 * val_z)
        fw1 = coeff * (G1 * val_x + G3 * val_y + G4 * val_z)
        fw2 = coeff * (G2 * val_x + G4 * val_y + G5 * val_z)

        scratchx[tx, ty, tz] = fw0
        scratchy[tx, ty, tz] = fw1
        scratchz[tx, ty, tz] = fw2

        cuda.syncthreads()
        # Apply contraction in the x-direction
        val_x = float32(0.0)
        for ix in range(n):
            val_x += dphi[ix, tx] * scratchx[ix, ty, tz]

        # Apply contraction in the y-direction
        val_y = float32(0.0)
        for iy in range(n):
            val_y += dphi[iy, ty] * scratchy[tx, iy, tz]

        # Apply contraction in the z-direction
        val_z = float32(0.0)
        for iz in range(n):
            val_z += dphi[iz, tz] * scratchz[tx, ty, iz]

        # Add contributions
        val = val_x + val_y + val_z

        # Atomically add the computed value to the output array `y`
        cuda.atomic.add(y, dof, val)

    return operator


@cuda.jit
def axpy(alpha: np.floating, x: numba.types.Array, y: numba.types.Array):
    """
    AXPY: y = a*x + y

    Parameters
    ----------
    alpha : scalar coefficient
    x : input vector
    y : input and output vector
    """

    i = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if i < x.size:
        y[i] = alpha * x[i] + y[i]


@cuda.jit
def copy(a: numba.types.Array, b: numba.types.Array):
    """
    Copy the entries of vector a to vector b

    Parameters
    ----------
    a : input vector
    b : output vector
    """

    i = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if i < a.size:
        b[i] = a[i]


@cuda.jit
def fill(alpha: np.floating, x: numba.types.Array):
    """
    Fill the entries of vector x with scalar alpha

    Parameters
    ----------
    alpha : scalar
    x : vector
    """

    i = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if i < x.size:
        x[i] = alpha


@cuda.jit
def pointwise_divide(a: numba.types.Array, b: numba.types.Array, c: numba.types.Array):
    """
    Pointwise divide: c = a / b

    Parameters
    ----------
    a : input vector
    b : input vector
    c : output vector
    """

    i = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if i < c.size:
        c[i] = a[i] / b[i]


@cuda.jit
def square(a: numba.types.Array, b: numba.types.Array):
    """
    Square the entries of vector a and store in vector b

    Parameters
    ----------
    a : input vector
    b : output vector
    """

    i = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if i < a.size:
        b[i] = a[i] * a[i]

