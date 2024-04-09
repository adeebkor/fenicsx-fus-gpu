"""
ADD DOCS!
"""

import numpy as np
import numba
import numba.cuda as cuda
from mpi4py import MPI


# @cuda.jit
# def pack(in_, out_, N):
#     thread_id = cuda.threadIdx.x
#     block_id = cuda.blockIdx.x
#     idx = thread_id + block_id * cuda.blockDim.x


def scatter_reverse(comm, owners_data, ghosts_data, N, float_type):
    """
    ADD DOC!
    """

    owners_idx, owners_size, owners_offsets, owners = owners_data
    ghosts_idx, ghosts_size, ghosts_offsets, ghosts = ghosts_data

    send_buff = cuda.to_device(np.zeros(np.sum(owners_size), dtype=float_type))
    recv_buff = cuda.to_device(np.zeros(np.sum(ghosts_size), dtype=float_type))

    def scatter(buffer):
        """
        ADD DOC!
        """

        all_requests = []

        # Pack code here!

        for i, dest in enumerate(owners):
            begin = owners_offsets[i]
            end = owners_offsets[i + 1]
            reqs = comm.Isend(send_buff[begin:end], dest=dest)
            all_requests.append(reqs)

        for i, src in enumerate(ghosts):
            begin = ghosts_offsets[i]
            end = ghosts_offsets[i + 1]
            reqr = comm.Irecv(recv_buff[begin:end], source=src)
            all_requests.append(reqr)

        MPI.Request.Waitall(all_requests)

        # Unpack code here!

    return scatter
