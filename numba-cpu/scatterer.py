"""
==========
Scatterers
==========

This file contains the kernels for MPI communication, namely the scattering
operation. It includes the scatter reverse and scatter forward functions.

Copyright (C) 2024 Adeeb Arif Kor
"""

import numpy as np
import numpy.typing as npt
from mpi4py import MPI


def scatter_reverse(
        comm: MPI.Comm, owners_data: list, ghosts_data: list, N: int, 
        float_type: np.dtype[np.floating]):

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
    scatter : function
    """
    
    owners_idx, owners_size, owners_offsets, owners = owners_data
    ghosts_idx, ghosts_size, ghosts_offsets, ghosts = ghosts_data

    send_buff = np.zeros(np.sum(owners_size), dtype=float_type)
    recv_buff = np.zeros(np.sum(ghosts_size), dtype=float_type)

    def scatter(buffer: npt.NDArray[np.floating]):
        """
        Perform the scatter reverse operation of the buffer array.

        Parameters
        ----------
        buffer : arrays to perform scatter reverse
        """

        all_requests = []

        # Pack
        for i, idx in enumerate(owners_idx):
            send_buff[i] = buffer[N:][idx]
            
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

        # Unpack
        for i, idx in enumerate(ghosts_idx):
            buffer[idx] += recv_buff[i]

    return scatter


def scatter_forward(
        comm: MPI.Comm, owners_data: list, ghosts_data: list, N: int, 
        float_type: np.dtype[np.floating]):
    
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
    scatter : function
    """

    owners_idx, owners_size, owners_offsets, owners = owners_data
    ghosts_idx, ghosts_size, ghosts_offsets, ghosts = ghosts_data

    send_buff = np.zeros(np.sum(ghosts_size), dtype=float_type)
    recv_buff = np.zeros(np.sum(owners_size), dtype=float_type)

    def scatter(buffer):
        """
        Perform the scatter forward operation of the buffer array.

        Parameters
        ----------
        buffer : arrays to perform scatter forward
        """

        all_requests = []

        # Unpack
        for i, idx in enumerate(ghosts_idx):
            send_buff[i] = buffer[idx]
        
        for i, dest in enumerate(ghosts):
            begin = ghosts_offsets[i]
            end = ghosts_offsets[i + 1]
            reqs = comm.Isend(send_buff[begin:end], dest=dest)
            all_requests.append(reqs)

        for i, src in enumerate(owners):
            begin = owners_offsets[i]
            end = owners_offsets[i + 1]
            reqr = comm.Irecv(recv_buff[begin:end], source=src)
            all_requests.append(reqr)

        MPI.Request.Waitall(all_requests)

        # Pack
        for i, idx in enumerate(owners_idx):
            buffer[N:][idx] = recv_buff[i]
    
    return scatter
