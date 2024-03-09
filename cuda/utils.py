import numpy as np
import numpy.typing as npt
from dolfinx.mesh import Mesh
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells


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


def compute_eval_params(mesh, points):
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

    points_on_proc = np.array(points_on_proc)

    return points_on_proc, cells