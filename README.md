# FEniCSx - FUS GPU implementation

This repository contains codes to prototype the FUS solver that uses GPU. It includes
several prototypes of the relevant operators (mass and stiffness). There are the 
Numba CPU implementation, the pure C++ implementation, and the Numba CUDA implementation.

To run the Numba CPU code, run the following command:

```python3 time_operators.py```

this will print out the timings of the CPU Numba operators.

To run the C++ code, do:

1. ffcx forms.py
2. mkdir build
3. cd build
4. cmake ..
5. make
6. ./time_operators

this will print out the timings of the C++ operators.

## Dependencies

* FEniCSx
* Numpy
* Numba
