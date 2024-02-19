"""
=========
Operators
=========

This file contains the kernels for vector assembly. It includes the boundary
facet, mass, and stiffness kernels.

Copyright (C) 2024 Adeeb Arif Kor
"""

import numpy as np
import numpy.typing as npt
import numba

from sum_factorisation import contract, transpose


@numba.njit(fastmath=True)
def mass_operator(
        x: npt.NDArray[np.floating],
        entity_constants: npt.NDArray[np.floating],
        y: npt.NDArray[np.floating],
        entity_detJ: npt.NDArray[np.floating],
        entity_dofmap: npt.NDArray[np.int32]):

    """
    Perform the vector assembly of the mass operator.

    Parameters
    ----------
    x : input vector
    entity_constant : constant values that are defined for each entity.
    y : output vector
    entity_detJ : scaled Jacobian determinant
    entity_dofmap : degrees-of-freedom map
    """

    num_entities = entity_constants.size

    for entity in range(num_entities):
        # Pack coefficients
        x_ = x[entity_dofmap[entity]]

        # Apply transform
        x_ *= entity_detJ[entity] * entity_constants[entity]

        # Add contributions
        y[entity_dofmap[entity]] += x_


@numba.njit(fastmath=True)
def stiffness_transform(
        Gc: npt.NDArray[np.floating],
        cell_constant: np.floating,
        fw0: npt.NDArray[np.floating],
        fw1: npt.NDArray[np.floating],
        fw2: npt.NDArray[np.floating],
        nq: int):

    """
    Geometric transformation

    Parameters
    ----------
    Gc : scaled geometric matrix
    cell_constant : constant value defined for the cell.
    fw0 : array
    fw1 : array
    fw2 : array
    nq : number of quadrature points in 1D
    """

    for q in range(nq*nq*nq):
        G_ = Gc[q]
        w0 = fw0[q]
        w1 = fw1[q]
        w2 = fw2[q]

        fw0[q] = cell_constant * (G_[0] * w0 + G_[1] * w1 + G_[2] * w2)
        fw1[q] = cell_constant * (G_[1] * w0 + G_[3] * w1 + G_[4] * w2)
        fw2[q] = cell_constant * (G_[2] * w0 + G_[4] * w1 + G_[5] * w2)


@numba.njit(fastmath=True)
def stiffness_operator(
        x: npt.NDArray[np.floating],
        cell_constants: npt.NDArray[np.floating],
        y: npt.NDArray[np.floating],
        G: npt.NDArray[np.floating],
        dofmap: npt.NDArray[np.int32],
        dphi: npt.NDArray[np.floating],
        nd: int,
        float_type: np.dtype[np.floating]):

    """"
    Perform the vector assembly of the stiffness operator.

    Parameters
    ----------
    x : input vector
    cell_constant : constant values that are defined for each cell.
    y : output vector
    G : geometric transformation data
    dofmap : degrees-of-freedom map
    dphi : derivatives of the 1D basis functions
    nd : number of degrees-of-freedom in 1D
    """

    num_cell = cell_constants.size

    # Initialise temporaries
    T1 = np.zeros((nd*nd*nd), float_type)
    T2 = np.zeros((nd*nd*nd), float_type)
    T3 = np.zeros((nd*nd*nd), float_type)
    T4 = np.zeros((nd*nd*nd), float_type)

    fw0 = np.zeros((nd*nd*nd), float_type)
    fw1 = np.zeros((nd*nd*nd), float_type)
    fw2 = np.zeros((nd*nd*nd), float_type)

    y0_ = np.zeros((nd*nd*nd), float_type)
    y1_ = np.zeros((nd*nd*nd), float_type)
    y2_ = np.zeros((nd*nd*nd), float_type)

    for cell in range(num_cell):

        T1[:] = 0.0
        T2[:] = 0.0
        T3[:] = 0.0
        T4[:] = 0.0

        fw0[:] = 0.0
        fw1[:] = 0.0
        fw2[:] = 0.0

        # Pack coefficients
        x_ = x[dofmap[cell]]

        # Apply contraction in the x-direction
        contract(dphi, x_, fw0, nd, nd, nd, nd, True)  # [q1, i1] x [i1, i2, i3] -> [q1, i2, i3] # noqa: E501

        # Apply contraction in the y-direction
        transpose(x_, T1, nd, nd, nd, nd, nd*nd, 1)  # [i1, i2, i3] -> [i2, i1, i3] # noqa: E501
        contract(dphi, T1, T2, nd, nd, nd, nd, True)  # [q2, i2] x [i2, i1, i3] -> [q2, i1, i3] # noqa: E501
        transpose(T2, fw1, nd, nd, nd, nd, nd*nd, 1)  # [q2, i1, i3] -> [i1, q2, i3] # noqa: E501

        # Apply contraction in the z-direction
        transpose(x_, T3, nd, nd, nd, 1, nd, nd*nd)  # [i1, i2, i3] -> [i3, i2, i1] # noqa: E501
        contract(dphi, T3, T4, nd, nd, nd, nd, True)  # [q3, i3] x [i3, i2, i1] -> [q3, i2, i1] # noqa: E501
        transpose(T4, fw2, nd, nd, nd, 1, nd, nd*nd)  # [q3, i2, i1] -> [i1, i2, q3] # noqa: E501

        # Apply transform
        stiffness_transform(G[cell], cell_constants[cell], fw0, fw1, fw2, nd)

        T1[:] = 0.0
        T2[:] = 0.0
        T3[:] = 0.0
        T4[:] = 0.0

        y0_[:] = 0.0
        y1_[:] = 0.0
        y2_[:] = 0.0

        # Apply contraction in the x-direction
        contract(dphi, fw0, y0_, nd, nd, nd, nd, False)  # [j1, q1] x [q1, j2, j3] -> [j1, j2, j3] # noqa: E501

        # Apply contraction in the y-direction
        transpose(fw1, T1, nd, nd, nd, nd, nd*nd, 1)  # [j1, q2, j3] -> [q2, j1, j3] # noqa: E501
        contract(dphi, T1, T2, nd, nd, nd, nd, False)  # [j2, q2] x [q2, j1, j3] -> [j2, j1, j3] # noqa: E501
        transpose(T2, y1_, nd, nd, nd, nd, nd*nd, 1)  # [j2, j1, j3] -> [j1, j2, j3] # noqa: E501

        # Apply contraction in the z-direction
        transpose(fw2, T3, nd, nd, nd, 1, nd, nd*nd)  # [j1, j2, q3] -> [q3, j2, j1] # noqa: E501
        contract(dphi, T3, T4, nd, nd, nd, nd, False)  # [j3, q3] x [q3, j2, j1] -> [j3, j2, j1] # noqa: E501
        transpose(T4, y2_, nd, nd, nd, 1, nd, nd*nd)  # [j3, j2, j1] -> [j1, j2, j3] # noqa: E501

        # Add contributions
        y[dofmap[cell]] += y0_ + y1_ + y2_


@numba.njit(fastmath=True)
def axpy(alpha: np.floating,
         x: npt.NDArray[np.floating],
         y: npt.NDArray[np.floating]):

    """
    AXPY: y = a*x + y

    Parameters
    ----------
    alpha : scalar coefficient
    x : input vector
    y : input and output vector
    """

    for i in range(y.size):
        y[i] = alpha*x[i] + y[i]


@numba.njit
def copy(a: npt.NDArray[np.floating], b: npt.NDArray[np.floating]):
    """
    Copy the entries of vector a to vector b

    Parameters
    ----------
    a : input vector
    b : output vector
    """

    for i in range(a.size):
        b[i] = a[i]


@numba.njit(fastmath=True)
def pointwise_divide(a: npt.NDArray[np.floating], b: npt.NDArray[np.floating],
                     c: npt.NDArray[np.floating]):
    """
    Pointwise divide: c = a / b

    Parameters
    ----------
    a : input vector
    b : input vector
    c : output vector
    """

    for i in range(c.size):
        c[i] = a[i] / b[i]
