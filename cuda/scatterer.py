"""
ADD DOCS!
"""

import numpy as np
import numba
import numba.cuda as cuda
from mpi4py import MPI


@cuda.jit
def ghosts_dof(in_, out_, nlocal):
    thread_id = cuda.threadIdx.x
    block_id = cuda.blockIdx.x
    idx = thread_id + block_id * cuda.blockDim.x

    if idx < out_.size:
        out_[idx] = in_[idx + nlocal]


@cuda.jit
def pack(in_, out_, index):
    thread_id = cuda.threadIdx.x
    block_id = cuda.blockIdx.x
    idx = thread_id + block_id * cuda.blockDim.x
    
    if idx < index.size:
        out_[idx] = in_[index[idx]]


@cuda.jit
def unpack_rev(in_, out_, index):
    thread_id = cuda.threadIdx.x
    block_id = cuda.blockIdx.x
    idx = thread_id + block_id * cuda.blockDim.x
    
    if idx < index.size:
        cuda.atomic.add(out_, index[idx], in_[idx])


@cuda.jit
def unpack_fwd(in_, out_, index):
    thread_id = cuda.threadIdx.x
    block_id = cuda.blockIdx.x
    idx = thread_id + block_id * cuda.blockDim.x
    
    if idx < index.size:
        out_[index[idx]] = in_[idx]


def scatter_reverse(comm, owners_data, ghosts_data, N, float_type):
    """
    ADD DOC!
    """

    nlocal, nghost = N
    owners_idx, owners_size, owners_offsets, owners = owners_data
    ghosts_idx, ghosts_size, ghosts_offsets, ghosts = ghosts_data

    send_buff = [cuda.device_array((owner_size,), dtype=float_type) for owner_size in owners_size]
    recv_buff = [cuda.device_array((ghost_size,), dtype=float_type) for ghost_size in ghosts_size]
    ghosts_buff = cuda.device_array((nghost, ), dtype=float_type)

    def scatter(buffer):
        """
        ADD DOC!
        """

        all_requests = []

        # Set the number of threads
        threadsperblock = 128

        # Ghosts degrees-of-freedom
        numblocks_ghosts = (nghost + (threadsperblock - 1)) // threadsperblock
        ghosts_dof[numblocks_ghosts, threadsperblock](buffer, ghosts_buff, nlocal)

        # Pack
        numblocks_pack = [
            (owner_size + (threadsperblock - 1)) // threadsperblock
            for owner_size in owners_size
        ]
        for i, sb in enumerate(send_buff):
            pack[numblocks_pack[i], threadsperblock](ghosts_buff, sb, owners_idx[i])

        # Send
        for i, dest in enumerate(owners):
            reqs = comm.Isend(send_buff[i], dest=dest)
            all_requests.append(reqs)

        # Receive
        for i, src in enumerate(ghosts):
            reqr = comm.Irecv(recv_buff[i], source=src)
            all_requests.append(reqr)

        MPI.Request.Waitall(all_requests)

        # Unpack
        numblocks_unpack = [
            (ghost_size + (threadsperblock - 1)) // threadsperblock
            for ghost_size in ghosts_size
        ]
        for i, rb in enumerate(recv_buff):
            unpack_rev[numblocks_unpack[i], threadsperblock](rb, buffer, ghosts_idx[i])

    return scatter


def scatter_forward(comm, owners_data, ghosts_data, N, float_type):
    """
    ADD DOC!
    """

    nlocal, nghost = N
    owners_idx, owners_size, owners_offsets, owners = owners_data
    ghosts_idx, ghosts_size, ghosts_offsets, ghosts = ghosts_data

    send_buff = [cuda.device_array((ghost_size,), dtype=float_type) for ghost_size in ghosts_size]
    recv_buff = [cuda.device_array((owner_size,), dtype=float_type) for owner_size in owners_size]
    ghosts_buff = cuda.device_array((nghost, ), dtype=float_type)

    def scatter(buffer):
        """
        ADD DOC!
        """

        all_requests = []

        # Set the number of threads
        threadsperblock = 128

        # Pack
        numblocks_pack = [
            (ghost_size + (threadsperblock - 1)) // threadsperblock
            for ghost_size in ghosts_size
        ]
        for i, sb in enumerate(send_buff):
            pack[numblocks_pack[i], threadsperblock](buffer, sb, ghosts_idx[i])

        

