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

def mass_operator(N: int, float_type: np.dtype[np.floating]):
    """
    Outer functions to defined the compile-time constants for the mass 
    operator.

    Parameters
    ----------
    N : number of degrees-of-freedom on the entity
    float_type: floating-point type
    """

    @numba.njit(fastmath=True)
    def operator(
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
    
        # Iniialise temporaries
        x_ = np.zeros(N, float_type)


        for entity in range(num_entities):
            # Pack coefficients
            for i in range(N):
                x_[i] = x[entity_dofmap[entity][i]]

            # Apply transform
            for i in range(N):
                x_[i] *= entity_detJ[entity][i] * entity_constants[entity]

            # Add contributions
            for i in range(N):
                y[entity_dofmap[entity][i]] += x_[i]

    return operator


def stiffness_operator(P, dphi, float_type):
    """
    Outer functions to define compile-time constants for the stiffness 
    operator.

    Parameters
    ----------
    P : basis function polynomial degree
    dphi : derivatives of the 1D basis functions
    float_type : floating-point type
    """

    n = P + 1
    N = n * n * n
    
    contract_pre = contract(n, n, n, n, True)
    contract_post = contract(n, n, n, n, False)
    transpose_y = transpose(n, n, n, n, n*n, 1)
    transpose_z = transpose(n, n, n, 1, n, n*n)

    @numba.njit(fastmath=True)
    def stiffness_transform(
            Gc: npt.NDArray[np.floating],
            cell_constant: np.floating,
            fw0: npt.NDArray[np.floating],
            fw1: npt.NDArray[np.floating],
            fw2: npt.NDArray[np.floating]):

        """
        Geometric transformation

        Parameters
        ----------
        Gc : scaled geometric matrix
        cell_constant : constant value defined for the cell.
        fw0 : array
        fw1 : array
        fw2 : array
        """

        for q in range(N):
            G_ = Gc[q]
            w0 = fw0[q]
            w1 = fw1[q]
            w2 = fw2[q]

            fw0[q] = cell_constant * (G_[0] * w0 + G_[1] * w1 + G_[2] * w2)
            fw1[q] = cell_constant * (G_[1] * w0 + G_[3] * w1 + G_[4] * w2)
            fw2[q] = cell_constant * (G_[2] * w0 + G_[4] * w1 + G_[5] * w2)


    @numba.njit(fastmath=True)
    def operator(
            x: npt.NDArray[np.floating],
            cell_constants: npt.NDArray[np.floating],
            y: npt.NDArray[np.floating],
            G: npt.NDArray[np.floating],
            dofmap: npt.NDArray[np.int32]):

        """"
        Perform the vector assembly of the stiffness operator.

        Parameters
        ----------
        x : input vector
        cell_constant : constant values that are defined for each cell.
        y : output vector
        G : geometric transformation data
        dofmap : degrees-of-freedom map
        """

        num_cell = cell_constants.size

        # Initialise temporaries
        x_ = np.zeros(N, float_type)

        T1 = np.zeros(N, float_type)
        T2 = np.zeros(N, float_type)
        T3 = np.zeros(N, float_type)
        T4 = np.zeros(N, float_type)

        fw0 = np.zeros(N, float_type)
        fw1 = np.zeros(N, float_type)
        fw2 = np.zeros(N, float_type)

        y0_ = np.zeros(N, float_type)
        y1_ = np.zeros(N, float_type)
        y2_ = np.zeros(N, float_type)

        for cell in range(num_cell):

            T1[:] = 0.0
            T2[:] = 0.0
            T3[:] = 0.0
            T4[:] = 0.0

            fw0[:] = 0.0
            fw1[:] = 0.0
            fw2[:] = 0.0

            # Pack coefficients
            for i in range(N):
                x_[i] = x[dofmap[cell][i]]

            # Apply contraction in the x-direction
            contract_pre(dphi, x_, fw0)  # [q1, i1] x [i1, i2, i3] -> [q1, i2, i3] # noqa: E501

            # Apply contraction in the y-direction
            transpose_y(x_, T1)  # [i1, i2, i3] -> [i2, i1, i3] # noqa: E501
            contract_pre(dphi, T1, T2)  # [q2, i2] x [i2, i1, i3] -> [q2, i1, i3] # noqa: E501
            transpose_y(T2, fw1)  # [q2, i1, i3] -> [i1, q2, i3] # noqa: E501


            # Apply contraction in the z-direction
            transpose_z(x_, T3)  # [i1, i2, i3] -> [i3, i2, i1] # noqa: E501
            contract_pre(dphi, T3, T4)  # [q3, i3] x [i3, i2, i1] -> [q3, i2, i1] # noqa: E501
            transpose_z(T4, fw2)  # [q3, i2, i1] -> [i1, i2, q3] # noqa: E501

            # Apply transform
            stiffness_transform(G[cell], cell_constants[cell], fw0, fw1, fw2)

            T1[:] = 0.0
            T2[:] = 0.0
            T3[:] = 0.0
            T4[:] = 0.0

            y0_[:] = 0.0
            y1_[:] = 0.0
            y2_[:] = 0.0

            # Apply contraction in the x-direction
            contract_post(dphi, fw0, y0_)  # [j1, q1] x [q1, j2, j3] -> [j1, j2, j3] # noqa: E501

            # Apply contraction in the y-direction
            transpose_y(fw1, T1)  # [j1, q2, j3] -> [q2, j1, j3] # noqa: E501
            contract_post(dphi, T1, T2)  # [j2, q2] x [q2, j1, j3] -> [j2, j1, j3] # noqa: E501
            transpose_y(T2, y1_)  # [j2, j1, j3] -> [j1, j2, j3] # noqa: E501

            # Apply contraction in the z-direction
            transpose_z(fw2, T3)  # [j1, j2, q3] -> [q3, j2, j1] # noqa: E501
            contract_post(dphi, T3, T4)  # [j3, q3] x [q3, j2, j1] -> [j3, j2, j1] # noqa: E501
            transpose_z(T4, y2_)  # [j3, j2, j1] -> [j1, j2, j3] # noqa: E501

            # Add contributions
            for i in range(N):
                y[dofmap[cell][i]] += y0_[i] + y1_[i] + y2_[i]

    return operator


def axpy(local_size: int):
    n = local_size

    @numba.njit(fastmath=True)
    def kernel(alpha: np.floating,
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

        for i in range(n):
            y[i] = alpha*x[i] + y[i]
    
    return kernel


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


@numba.njit
def fill(alpha: np.floating, x: npt.NDArray[np.floating]):
    """
    Fill the entries of vector x with scalar alpha

    Parameters
    ----------
    alpha : scalar
    x : vector
    """

    for i in range(x.size):
        x[i] = alpha


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
