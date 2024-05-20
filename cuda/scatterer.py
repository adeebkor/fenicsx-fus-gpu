"""
==========
Scatterers
==========

This file contains the kernels for MPI communication, namely the scattering
operation. It includes the scatter reverse and scatter forward kernels.

Copyright (C) 2024 Adeeb Arif Kor
"""

import numpy as np
import numba
import numba.cuda as cuda
from mpi4py import MPI


@cuda.jit
def pack_fwd(in_: numba.types.Array, out_: numba.types.Array, index: numba.types.Array):
    """
    Pack coefficient.

    Parameters
    ----------
    in_ : input array
    out_ : output array
    index : indices of the input array to pack
    """

    thread_id = cuda.threadIdx.x
    block_id = cuda.blockIdx.x
    idx = thread_id + block_id * cuda.blockDim.x

    if idx < index.size:
        out_[idx] = in_[index[idx]]


@cuda.jit
def unpack_fwd(
    in_: numba.types.Array, out_: numba.types.Array, index: numba.types.Array, N: int
):
    """
    Unpack coefficient.

    Parameters
    ----------
    in_ : input array
    out_ : output array
    index : indices of the output array to unpack
    """

    thread_id = cuda.threadIdx.x
    block_id = cuda.blockIdx.x
    idx = thread_id + block_id * cuda.blockDim.x

    if idx < index.size:
        out_[index[idx] + N] = in_[idx]


@cuda.jit
def pack_rev(
    in_: numba.types.Array, out_: numba.types.Array, index: numba.types.Array, N: int
):
    """
    Pack coefficient.

    Parameters
    ----------
    in_ : input array
    out_ : output array
    index : indices of the input array to pack
    """

    thread_id = cuda.threadIdx.x
    block_id = cuda.blockIdx.x
    idx = thread_id + block_id * cuda.blockDim.x

    if idx < index.size:
        out_[idx] = in_[index[idx] + N]


@cuda.jit
def unpack_rev(
    in_: numba.types.Array, out_: numba.types.Array, index: numba.types.Array
):
    """
    Unpack coefficient.

    Parameters
    ----------
    in_ : input array
    out_ : output array
    index : indices of the output array to unpack
    """

    thread_id = cuda.threadIdx.x
    block_id = cuda.blockIdx.x
    idx = thread_id + block_id * cuda.blockDim.x

    if idx < index.size:
        cuda.atomic.add(out_, index[idx], in_[idx])


def scatter_reverse(
    comm: MPI.Comm,
    owners_data: list,
    ghosts_data: list,
    N: int,
    float_type: np.dtype[np.floating],
):
    """
    Outer function to capture the constant variables of the scatter reverse
    operation.

    Parameters
    ----------
    comm : MPI communicator
    owners_data : degrees-of-freedom data in this process that are owned by
        other processes
    ghosts_data : degrees-of-freedom data that are owned by this process and
        are ghosts in other processes
    N : size of local array
    float_type : buffer's floating-point type

    Return
    ------
    scatter : CUDA kernel
    """

    owners_idx, owners_size, owners = owners_data
    ghosts_idx, ghosts_size, ghosts = ghosts_data

    send_buff = [
        cuda.device_array((owner_size,), dtype=float_type) for owner_size in owners_size
    ]
    recv_buff = [
        cuda.device_array((ghost_size,), dtype=float_type) for ghost_size in ghosts_size
    ]

    def scatter(buffer: numba.types.Array):
        """
        Perform the scatter reverse operation of the buffer array.

        Parameters
        ----------
        buffer : array to perform scatter reverse
        """

        all_requests = []

        # Set the number of threads
        threadsperblock = 128

        # Pack
        numblocks_pack = [
            (owner_size + (threadsperblock - 1)) // threadsperblock
            for owner_size in owners_size
        ]
        for i, sb in enumerate(send_buff):
            pack_rev[numblocks_pack[i], threadsperblock](buffer, sb, owners_idx[i], N)
        
        # Synchronize
        cuda.synchronize()

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

        # Synchronize
        cuda.synchronize()

    return scatter


def scatter_forward(
    comm: MPI.Comm,
    owners_data: list,
    ghosts_data: list,
    N: int,
    float_type: np.dtype[np.floating],
):
    """
    Outer function to capture the constant variables of the scatter forward
    operation.

    Parameters
    ----------
    comm : MPI communicator
    owners_data : degrees-of-freedom data in this process that are owned by
        other processes
    ghosts_data : degrees-of-freedom data that are owned by this process and
        are ghosts in other processes
    N : size of local array
    float_type : buffer's floating-point type

    Return
    ------
    scatter : CUDA kernel
    """

    owners_idx, owners_size, owners = owners_data
    ghosts_idx, ghosts_size, ghosts = ghosts_data

    send_buff = [
        cuda.device_array((ghost_size,), dtype=float_type) for ghost_size in ghosts_size
    ]
    recv_buff = [
        cuda.device_array((owner_size,), dtype=float_type) for owner_size in owners_size
    ]

    def scatter(buffer: numba.types.Array):
        """
        Perform the scatter forward operation of the buffer array.

        Parameters
        ----------
        buffer : array to perform scatter forward
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
            pack_fwd[numblocks_pack[i], threadsperblock](buffer, sb, ghosts_idx[i])

        # Synchronize
        cuda.synchronize()

        # Send
        for i, dest in enumerate(ghosts):
            reqs = comm.Isend(send_buff[i], dest=dest)
            all_requests.append(reqs)

        # Receive
        for i, src in enumerate(owners):
            reqr = comm.Irecv(recv_buff[i], source=src)
            all_requests.append(reqr)

        MPI.Request.Waitall(all_requests)

        # Unpack
        numblocks_unpack = [
            (owner_size + (threadsperblock - 1)) // threadsperblock
            for owner_size in owners_size
        ]
        for i, rb in enumerate(recv_buff):
            unpack_fwd[numblocks_unpack[i], threadsperblock](
                rb, buffer, owners_idx[i], N
            )

        # Synchronize
        cuda.synchronize()

    return scatter
