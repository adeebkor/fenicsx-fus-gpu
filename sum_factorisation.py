"""
=================
Sum factorisation
=================

This file contains the functions that is use in the sum-factorisation 
algorithm, namely the transpose and contract functions.

Copyright (C) 2024 Adeeb Arif Kor
"""

import numpy as np
import numba

float_type = np.float64


@numba.njit(fastmath=True)
def transpose(A, B, Na, Nb, Nc, offa, offb, offc):
    """
    Perform the tranposition of the input tensor A (Na x Nb x Nc) and store it
    into the output tensor B.

    Parameters
    ----------
    A : input tensor
    B : output tensor
    Na : size of 1st dimension of A
    Nb : size of 2nd dimension of A
    Nc : size of 3rd dimension of A
    offa : offset for the 1st dimension of A
    offb : offset for the 2nd dimension of A
    offc : offset for the 3rd dimension of A
    """

    for a in range(Na):
        for b in range(Nb):
            for c in range(Nc):
                B[offa * a + offb * b + offc * c] = A[a * Nb * Nc + b * Nc + c]


@numba.njit(fastmath=True)
def contract(A, B, C, Nk, Na, Nb, Nc, bool):
    """
    Perform the tensor contraction between the input tensor A and B and store 
    it in the output tensor C.
    
    Parameters
    ----------
    A : input tensor (Nk x Na)
    B : input tensor (Na x Nb x Nc)
    C : output tensor (Nk x Nb x Nc)
    Nk : size of 1st dimension of input tensor A
    Na : size of 2nd dimension of input tensor A
    Nb : size of 2nd dimension of input tensor B
    Nc : size of 3rd dimension of input tensor B
    bool : if True, the input tensor A is transposed
    """

    Nd = Nb * Nc

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
