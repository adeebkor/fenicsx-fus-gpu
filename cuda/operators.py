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

    Parameters
    ----------
    x : input array
    entity_constants : array containing the material coefficients
    y : output array
    detJ_f : array containing the determinant of the Jacobian on the boundary facets
    entity_dofmap : 2d array containing the local DOF on given entities
    """
    thread_id = cuda.threadIdx.x  # Local thread ID (max: 1024)
    block_id = cuda.blockIdx.x  # Block ID (max: 2147483647)
    idx = thread_id + block_id * cuda.blockDim.x  # Global thread ID

    facet = idx // entity_dofmap.shape[1]
    local_dof = idx % entity_dofmap.shape[1]

    if idx < entity_dofmap.size:
        dof = entity_dofmap[facet, local_dof]
        value = x[dof] * detJ_entity[facet, local_dof] * entity_constants[facet]
        cuda.atomic.add(y, dof, value)
