import numpy as np
import numba

from sum_factorisation import contract, transpose

float_type = np.float64


@numba.njit(fastmath=True)
def mass_operator(x, coeffs, y, detJ, dofmap):
    nc = coeffs.size

    for c in range(nc):
        # Pack coefficients
        x_ = x[dofmap[c]]

        # Apply transform
        x_ *= detJ[c] * coeffs[c]

        # Add contributions
        y[dofmap[c]] += x_


@numba.njit(fastmath=True)
def stiffness_transform(Gc, coeff, fw0, fw1, fw2, nd):
    for iq in range(nd*nd*nd):
        G_ = Gc[iq]
        w0 = fw0[iq]
        w1 = fw1[iq]
        w2 = fw2[iq]

        fw0[iq] = coeff * (G_[0] * w0 + G_[1] * w1 + G_[2] * w2)
        fw1[iq] = coeff * (G_[1] * w0 + G_[3] * w1 + G_[4] * w2)
        fw2[iq] = coeff * (G_[2] * w0 + G_[4] * w1 + G_[5] * w2)


@numba.njit(fastmath=True)
def stiffness_operator(x, coeffs, y, G, dofmap, dphi, nd):
    nc = coeffs.size

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

    for c in range(nc):

        T1[:] = 0.0
        T2[:] = 0.0
        T3[:] = 0.0
        T4[:] = 0.0

        fw0[:] = 0.0
        fw1[:] = 0.0
        fw2[:] = 0.0

        # Pack coefficients
        x_ = x[dofmap[c]]

        # Apply contraction in the x-direction
        contract(dphi, x_, fw0, nd, nd, nd, nd, True)

        # Apply contraction in the y-direction
        transpose(x_, T1, nd, nd, nd, nd, nd*nd, 1)
        contract(dphi, T1, T2, nd, nd, nd, nd, True)
        transpose(T2, fw1, nd, nd, nd, nd, nd*nd, 1)

        # Apply contraction in the z-direction
        transpose(x_, T3, nd, nd, nd, 1, nd, nd*nd)
        contract(dphi, T3, T4, nd, nd, nd, nd, True)
        transpose(T4, fw2, nd, nd, nd, 1, nd, nd*nd)

        # Apply transform
        stiffness_transform(G[c], coeffs[c], fw0, fw1, fw2, nd)

        T1[:] = 0.0
        T2[:] = 0.0
        T3[:] = 0.0
        T4[:] = 0.0

        y0_[:] = 0.0
        y1_[:] = 0.0
        y2_[:] = 0.0

        # Apply contraction in the x-direction
        contract(dphi, fw0, y0_, nd, nd, nd, nd, False)

        # Apply contraction in the y-direction
        transpose(fw1, T1, nd, nd, nd, nd, nd*nd, 1)
        contract(dphi, T1, T2, nd, nd, nd, nd, False)
        transpose(T2, y1_, nd, nd, nd, nd, nd*nd, 1)

        # Apply contraction in the z-direction
        transpose(fw2, T3, nd, nd, nd, 1, nd, nd*nd)
        contract(dphi, T3, T4, nd, nd, nd, nd, False)
        transpose(T4, y2_, nd, nd, nd, 1, nd, nd*nd)

        # Add contributions
        y[dofmap[c]] += y0_ + y1_ + y2_


@numba.njit(fastmath=True)
def axpy(alpha, x, y):
    """
    axpy operation
    """
    for i in range(y.size):
        y[i] = alpha*x[i] + y[i]


@numba.njit
def copy(a, b):
    """
    Copying array
    """

    for i in range(a.size):
        b[i] = a[i]


@numba.njit(fastmath=True)
def pointwise_divide(a, b, c):
    """
    Pointwise divide operation
    c[i] = a[i] / b[i]
    """

    for i in range(c.size):
        c[i] = a[i] / b[i]
