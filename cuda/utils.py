import numpy as np
import numpy.typing as npt
from mpi4py import MPI
from dolfinx.mesh import Mesh
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells


def compute_scatterer_data(index_map):
    """
    Extract scatterer data, i.e., obtain the owners and ghosts 
    degrees-of-freedom.

    Parameters
    ----------
    index_map : dolfinx index map

    Return
    ------
    owners_data : list containing owners data
    ghosts_data : list containing ghosts data
    """

    # Compute ghosts data in this process that are owned by other processes
    nlocal = index_map.size_local
    nghost = index_map.num_ghosts
    owners = index_map.owners
    unique_owners, owners_size = np.unique(owners, return_counts=True)
    owners_argsorted = np.argsort(owners)

    owners_offsets = np.cumsum(owners_size)
    owners_offsets = np.insert(owners_offsets, 0, 0)

    owners_idx = [np.zeros(size, dtype=np.int64) for size in owners_size]
    for i, owner in enumerate(unique_owners):
        begin = owners_offsets[i]
        end = owners_offsets[i + 1]
        owners_idx[i] = owners_argsorted[begin:end]

    # Compute owned data by this process that are ghosts data in other process
    shared_dofs = index_map.index_to_dest_ranks()
    shared_ranks = np.unique(shared_dofs.array)

    ghosts = []
    for shared_rank in shared_ranks:
        for dof in range(nlocal):
            if shared_rank in shared_dofs.links(dof):
                ghosts.append(shared_rank)

    ghosts = np.array(ghosts)
    unique_ghosts, ghosts_size = np.unique(ghosts, return_counts=True)
    ghosts_offsets = np.cumsum(ghosts_size)
    ghosts_offsets = np.insert(ghosts_offsets, 0, 0)

    all_requests = []

    # Send
    send_buff_idx = [np.zeros(size, dtype=np.int64) for size in owners_size]
    for i, owner in enumerate(unique_owners):
        begin = owners_offsets[i]
        end = owners_offsets[i + 1]
        send_buff_idx[i] = index_map.ghosts[owners_argsorted[begin:end]]
        reqs = MPI.COMM_WORLD.Isend(send_buff_idx[i], dest=owner)
        all_requests.append(reqs)

    # Receive
    recv_buff_idx = [np.zeros(size, dtype=np.int64) for size in ghosts_size]
    for i, ghost in enumerate(unique_ghosts):
        reqr = MPI.COMM_WORLD.Irecv(recv_buff_idx[i], source=ghost)
        all_requests.append(reqr)

    MPI.Request.Waitall(all_requests)

    ghosts_idx = [recv_buff - index_map.local_range[0] for recv_buff in recv_buff_idx]

    owners_data = [owners_idx, owners_size, unique_owners]
    ghosts_data = [ghosts_idx, ghosts_size, unique_ghosts]

    return owners_data, ghosts_data


def facet_integration_domain(facets: npt.NDArray[np.int32], mesh: Mesh):
    """
    Return the integration domain for the facet integration.

    Parameters
    ----------
    facets : array containing the facets indices.
    mesh : dolfinx mesh

    Returns
    -------
    boundary_data : array containing the cells and local facets indices on the
        boundary.
    """

    tdim = mesh.topology.dim

    # Find the cells that contains the integration facets
    cell_to_facet_map = mesh.topology.connectivity(tdim, tdim - 1)
    facet_to_cell_map = mesh.topology.connectivity(tdim - 1, tdim)

    boundary_facet_cell = np.zeros_like(facets, dtype=np.int32)
    for i, facet in enumerate(facets):
        boundary_facet_cell[i] = facet_to_cell_map.links(facet)[0]

    boundary_data = np.zeros((facets.size, 2), dtype=np.int32)

    for i, (facet, cell) in enumerate(zip(facets, boundary_facet_cell)):
        facets = cell_to_facet_map.links(cell)
        local_facet = np.where(facet == facets)
        boundary_data[i, 0] = cell
        boundary_data[i, 1] = local_facet[0][0]

    return boundary_data


def compute_eval_params(mesh, points, float_type):
    """
    Compute the parameters required for dolfinx.Function eval

    Parameters
    ----------

    mesh : dolfinx.mesh

    points : numpy.ndarray
            The evaluation points of shape (3 by n) where each row corresponds
            to x, y, and z coordinates.

    Returns
    -------

    points_on_proc : numpy.ndarray
            The evaluation points owned by the process.

    cells : list
            A list containing the cell index of the evaluation point.
    """

    tree = bb_tree(mesh, mesh.topology.dim, padding=1e-12)
    cells = []
    points_on_proc = []
    cell_candidates = compute_collisions_points(tree, points.T)
    cell_collisions = compute_colliding_cells(mesh, cell_candidates, points.T)

    for i, point in enumerate(points.T):
        # Only use evaluate points on current processor
        if len(cell_collisions.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(cell_collisions.links(i)[0])

    points_on_proc = np.array(points_on_proc, dtype=float_type)

    return points_on_proc, cells


def compute_diffusivity_of_sound(
    frequency: float, speed: float, attenuationdB: float
) -> float:
    attenuationNp = attenuationdB / 20 * np.log(10)  # (Np/m/MHz^2)
    diffusivity = 2 * attenuationNp * speed * speed * speed / frequency / frequency
    return diffusivity
