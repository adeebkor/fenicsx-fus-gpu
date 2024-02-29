"""
=================
Sum factorisation
=================

This file contains the functions that is use in the sum-factorisation
algorithm, namely the transpose and contract functions.

Copyright (C) 2024 Adeeb Arif Kor
"""

import numpy as np
import numpy.typing as npt
import numba

def transpose(Na: int, Nb: int, Nc: int, offa: int, offb: int, offc: int):
    """
    Outer function to define compile-time constants of the transpose operator

    Parameters
    ----------
    Na : size of 1st dimension of A
    Nb : size of 2nd dimension of A
    Nc : size of 3rd dimension of A
    offa : offset for the 1st dimension of A
    offb : offset for the 2nd dimension of A
    offc : offset for the 3rd dimension of A
    """

    @numba.njit(fastmath=True)
    def operator(A: npt.NDArray[np.floating], B: npt.NDArray[np.floating]):
        """
        Perform the tranposition of the input tensor A (Na x Nb x Nc) and store it
        into the output tensor B.

        Parameters
        ----------
        A : input tensor
        B : output tensor
        """

        for a in range(Na):
            for b in range(Nb):
                for c in range(Nc):
                    B[offa * a + offb * b + offc * c] = A[a * Nb * Nc + b * Nc + c]
    
    return operator

def contract(Nk: int, Na: int, Nb: int, Nc: int, bool: bool):
    """
    Outer function to define compile-time constants of the contraction operator

    Parameters
    ----------
    Nk : size of 1st dimension of input tensor A
    Na : size of 2nd dimension of input tensor A
    Nb : size of 2nd dimension of input tensor B
    Nc : size of 3rd dimension of input tensor B
    bool : if True, the input tensor A is transposed
    
    """

    Nd = Nb * Nc

    @numba.njit(fastmath=True)
    def operator(A: npt.NDArray[np.floating], B: npt.NDArray[np.floating],
                 C: npt.NDArray[np.floating]):
        """
        Perform the tensor contraction between the input tensor A and B and store
        it in the output tensor C.

        Parameters
        ----------
        A : input tensor (Nk x Na)
        B : input tensor (Na x Nb x Nc)
        C : output tensor (Nk x Nb x Nc)
        """

        if bool:
            for k in range(Nk):
                for a in range(Na):
                    for d in range(Nd):
                        C[a * Nd + d] += A[a * Nk + k] * B[k * Nd + d]
        else:
            for k in range(Nk):
                for a in range(Na):
                    for d in range(Nd):
                        C[a * Nd + d] += A[k * Na + a] * B[k * Nd + d]

    return operator
