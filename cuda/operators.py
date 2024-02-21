import numba
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

    facet = idx // entity_dofmap.shape[1]
    local_dof = idx % entity_dofmap.shape[1]

    if idx < entity_dofmap.size:
        # Compute the global DOF index
        dof = entity_dofmap[facet, local_dof]

        # Compute the contribution of the current DOF to the mass operator
        value = x[dof] * detJ_entity[facet, local_dof] * entity_constants[facet]

        # Atomically add the computed value to the output array `y`
        cuda.atomic.add(y, dof, value)

nd = 5

@cuda.jit
def stiffness_operator(
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
    dphi : numba.types.Array
        Array containing the derivatives of the 1D basis functions
    nd : int
        Number of degrees-of-freedom in 1D
    """

    float_type = dphi.dtype

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    tz = cuda.threadIdx.z

    thread_id = tx * nd**2 + ty * nd + tz
    block_id = cuda.blockIdx.x

    x_ = cuda.shared.array(shape=(nd, nd, nd), dtype=float_type)
    y_ = cuda.shared.array(shape=(nd, nd, nd), dtype=float_type)
    z_ = cuda.shared.array(shape=(nd, nd, nd), dtype=float_type)

    w0 = cuda.shared.array(shape=(nd, nd, nd), dtype=float_type)
    w1 = cuda.shared.array(shape=(nd, nd, nd), dtype=float_type)
    w2 = cuda.shared.array(shape=(nd, nd, nd), dtype=float_type)
  
    # Get dof index that this thread is computing
    dof = entity_dofmap[block_id, thread_id]

    # Gather x expression value required by this thread
    x_[tz, ty, tx] = x[dof]
    y_[tx, tz, ty] = x[dof]
    z_[tx, ty, tz] = x[dof]
    cuda.syncthreads()

    # Apply contraction in the x-direction
    val_x = 0.0
    for ix in range(nd):
      val_x += dphi[tx, ix] * x_[tz, ty, ix]

    w0[tx, ty, tz] = val_x

    # Apply contraction in the y-direction
    val_y = 0.0
    for iy in range(nd):
      val_y += dphi[ty, iy] * y_[tx, tz, iy]

    w1[tx, ty, tz] = val_y

    # Apply contraction in the z-direction
    val_z = 0.0
    for iz in range(nd):
      val_z += dphi[tz, iz] * z_[tx, ty, iz]

    w2[tx, ty, tz] = val_z

    # Apply transform
    fw0 = entity_constants[block_id] * (
      G_entity[block_id] * w0[tx, ty, tz] + G_entity[block_id] * w1[tx, ty, tz]
      + G_entity[block_id] * w2[tx, ty, tz])
    fw1 = entity_constants[block_id] * (
      G_entity[block_id] * w0[tx, ty, tz] + G_entity[block_id] * w1)
    fw2 = entity_constants[block_id] * (G_entity[block_id] * w0[tx, ty, tz] + G_entity[block_id] * val_y)




