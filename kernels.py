import numpy as np
import numba


@numba.njit()
def polynomial_kernel(p, q, d):
    """
    Polynomial kernel

    :param p:  vector
    :param q:  vector
    :param d:  kernel dimension
    """
    return np.power(np.dot(p, q), d)


@numba.njit()
def gaussian_kernel(p, q, d):
    """
    Gaussian kernel

    :param p: vector
    :param q: vector
    :param d: kernel width
    """
    return np.exp(-d * (np.linalg.norm(p - q) ** 2))


@numba.njit()
def kernel_matrix(A, B, d, kernel_func):
    """
    Kernel matrix calculated between matrices A and B
    """
    n, m = len(A), len(B)
    K = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            K[i, j] = kernel_func(A[i], B[j], d)
    return np.ascontiguousarray(K)


@numba.njit()
def gram_matrix(A, d, kernel_func):
    """
    Kernel matrix calculated between the train set A
    """
    n = len(A)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            z = kernel_func(A[i], A[j], d)
            K[i, j] = z
            K[j, i] = z
    return np.ascontiguousarray(K)
