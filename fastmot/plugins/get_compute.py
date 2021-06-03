"""
Outputs some information on CUDA-enabled devices on your computer,
including current memory usage.

It's a port of https://gist.github.com/f0k/0d6431e3faa60bffc788f8b4daa029b1
from C to Python with ctypes, so it can run without compiling anything. Note
that this is a direct translation with no attempt to make the code Pythonic.
It's meant as a general demonstration on how to obtain CUDA device information
from Python without resorting to nvidia-smi or a compiled Python extension.
"""

import sys
import ctypes


CUDA_SUCCESS = 0


def main():
    libnames = ('libcuda.so', 'libcuda.dylib', 'cuda.dll')
    for libname in libnames:
        try:
            cuda = ctypes.CDLL(libname)
        except OSError:
            continue
        else:
            break
    else:
        return

    gpu_archs = set()

    n_gpus = ctypes.c_int()
    cc_major = ctypes.c_int()
    cc_minor = ctypes.c_int()

    result = ctypes.c_int()
    device = ctypes.c_int()
    error_str = ctypes.c_char_p()

    result = cuda.cuInit(0)
    if result != CUDA_SUCCESS:
        cuda.cuGetErrorString(result, ctypes.byref(error_str))
        print('cuInit failed with error code %d: %s' % (result, error_str.value.decode()))
        return 1

    result = cuda.cuDeviceGetCount(ctypes.byref(n_gpus))
    if result != CUDA_SUCCESS:
        cuda.cuGetErrorString(result, ctypes.byref(error_str))
        print('cuDeviceGetCount failed with error code %d: %s' % (result, error_str.value.decode()))
        return 1

    for i in range(n_gpus.value):
        if cuda.cuDeviceComputeCapability(ctypes.byref(cc_major), ctypes.byref(cc_minor), device) == CUDA_SUCCESS:
            gpu_archs.add(str(cc_major.value) + str(cc_minor.value))
    print(' '.join(gpu_archs))

    return 0


if __name__ == '__main__':
    sys.exit(main())
