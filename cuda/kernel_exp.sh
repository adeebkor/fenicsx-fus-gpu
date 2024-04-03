#!/bin/bash

#source ~/spack/share/spack/setup-env.sh
#spack env activate fenicsx-env

for f in float32 float64
do
    for n in "3 2.6" "4 2.0" "5 1.6" "6 1.3" "7 1.1" "8 1.0" "9 0.9"  
    do
        set -- $n
        nsys profile -o report_P$1_$f python3 exp_kernel_speed.py $f $n $2
        # ncu -o stiffness_P$1_basix_$f --set full --kernel-name v2,cw51cXTLSUwv1sCUt9Ww0FEw09RRQPKzLTg4gaGKFsG2oMQGEYakJSQB1PQBk0Bynm21OiwU1a0UoLGhDpQE8oxrNQE_3d] --launch-skip 10 --launch-count 1 "python3" exp_kernel_speed.py $f $n $2
        # ncu -o stiffness_P$1_tp_$f --set full --kernel-name v1,cw51cXTLSUwv1sCUt9Ww0FEw09RRQPKzLTg4gaGKFsG2oMQGEYakJSQB1PQBk0Bynm21OiwU1a0UoLGhDpQE8oxrNQE_3d] --launch-skip 10 --launch-count 1 "python3" exp_kernel_speed.py $f $n $2
    done
done