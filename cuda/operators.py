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
