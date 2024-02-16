import numpy as np
import numpy.typing as npt
from dolfinx.mesh import Mesh


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
