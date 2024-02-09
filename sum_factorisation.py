import numpy as np
import numba


@numba.njit(fastmath=True)
def transpose(A, B, Na, Nb, Nc, offa, offb, offc):
    for a in range(Na):
        for b in range(Nb):
            for c in range(Nc):
                B[offa * a + offb * b + offc * c] = A[a * Nb * Nc + b * Nc + c]


@numba.njit(fastmath=True)
def contract(A, B, C, Nk, Na, Nb, Nc, bool):
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


if __name__ == "__main__":
    float_type = np.float32
    N = 4

    a = np.random.rand(N*N).astype(float_type)
    b = np.random.rand(N*N*N).astype(float_type)
    c = np.random.rand(N*N*N).astype(float_type)

    # Test transpose
    cr = c.reshape(N, N, N)
    c0_t = cr.transpose(1, 0, 2)

    ct_0 = np.empty_like(c, dtype=float_type)
    transpose(c, ct_0, N, N, N, N, N*N, 1)  # transpose(1, 0, 2)

    np.testing.assert_allclose(c0_t.flatten(), ct_0)

    c1_t = cr.transpose(2, 1, 0)

    ct_1 = np.empty_like(c, dtype=float_type)
    transpose(c, ct_1, N, N, N, 1, N, N*N)  # transpose(2, 1, 0)

    np.testing.assert_allclose(c1_t.flatten(), ct_1)

    # Test contract
    mat_a = np.random.rand(N*N).astype(float_type)
    mat_b = np.random.rand(N*N*N).astype(float_type)

    mat_ar = mat_a.reshape(N, N)

    # for i in range(N):
    #     for j in range(N):
    #         print(f"{mat_ar[i][j]} : {mat_a[j + i*N]}")

    mat_br = mat_b.reshape(N, N, N)

    # for i in range(N):
    #     for j in range(N):
    #         for k in range(N):
    #             print(f"{mat_br[i][j][k]} : {mat_b[k + j*N + i*N*N]}")

    mat_c = np.zeros_like(mat_b, dtype=float_type)

    contract(mat_a, mat_b, mat_c, N, N, N, N, False)

    mat_cr = mat_ar.T @ mat_br.reshape(N, N*N)
    mat_cr = mat_cr.reshape(N, N, N)

    # for i in range(N):
    #     for j in range(N):
    #         for k in range(N):
    #             print(f"{mat_cr[i][j][k]} : {mat_c[k + j*N + i*N*N]}")

    mat_ce = np.einsum("iq,ijk->qjk", mat_ar, mat_br)

    for i in range(N):
        for j in range(N):
            for k in range(N):
                print(f"{mat_ce[i][j][k]} : {mat_c[k + j*N + i*N*N]}")
