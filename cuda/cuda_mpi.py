#
# This file is created to test whether the MPI communication is
# really sending from device-to-device without copying it to host.
# Typically run with nsys to get the profiling report and check the
# report using NSight Systems.

import time
from mpi4py import MPI
import numpy as np
import numba.cuda as cuda

comm = MPI.COMM_WORLD
size = comm.size
rank = comm.rank

cuda.select_device(rank) 

print(f"{rank} : {cuda.get_current_device()}")

N = 1_000_000
send_buff = np.random.randint(0, 10, N, dtype=np.int32)
recv_buff = np.zeros_like(send_buff, dtype=np.int32)

print(f"{rank} : before send (send) : {send_buff}")
print(f"{rank} : before send (recv) : {recv_buff}")

send_buff_d = cuda.to_device(send_buff)
recv_buff_d = cuda.to_device(recv_buff)

temp_buff_d = cuda.device_array_like(send_buff_d)
# cuda.default_stream().synchronize()

all_request = []

reqs = comm.Isend(send_buff_d, dest=(rank + 1) % 8)
all_request.append(reqs)

reqr = comm.Irecv(recv_buff_d, source=(rank + 7) % 8)
all_request.append(reqr)

temp_buff_d.copy_to_device(send_buff_d)

MPI.Request.Waitall(all_request)
cuda.default_stream().synchronize()
recv_buff = recv_buff_d.copy_to_host()

print(f"{rank} : after send (recv) : {recv_buff}")
