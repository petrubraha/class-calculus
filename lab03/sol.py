import math
import numpy as np
import random
import scipy as sp
import sys

EPSILON = 1e-8

def identity_maxtrix(n: int):
    identity = np.zeros([n, n])
    for i in range(n):
        identity[i, i] = 1.0
    return identity

def compute_b(n, a, s):
    b = np.zeros(n)
    for i in range(n):
        for j in range(n):
            b[i] += s[j] * a[i, j]
    return b

def _compute_sigma(n, r, a) -> float:
    sigma = 0.0
    for i in range(r, n):
        sigma += a[i, r]
    return sigma

def _transform_a(a, u, r: int, beta: float):
    n = len(a)
    for j in range(r + 1, n):
        sum = 0
        for i in range(r, n):
            sum += u[i] * a[i, j]
        beta = sum / beta
        gamma = # todo

def qr_decomp_house(a):
    """Modifies the received buffer, i.e. a will be a upper triangular matrix."""
    n = len(a)
    q = identity_maxtrix(n)
    u = np.zeros(n)

    for r in range(n - 1):
        sigma = _compute_sigma(n, r, a)
        if sigma <= EPSILON:
            raise ValueError("a is singular")
        
        k = math.sqrt(sigma)
        if (a[r, r] > 0):
            k *= -1.0
        
        beta = sigma - k * a[r, r]
        
        u[r] = a[r, r] - k
        for i in range(r + 1, n):
            u[i] = a[i, r]

        _transform_a(a, u, r, beta)
        # todo
        
    return q, a

def qr_decomp_lib(a):
    return np.linalg.qr(a)

def compute_x(q, r, b):
    y = np.dot(q.T, b)
    x = sp.linalg.solve(r, y)
    return x

def main(n: int | None):
    if n == None:
        n = random.randrange(2, 3)
    a = np.random.rand(n, n)
    s = np.random.rand(n)
    b = compute_b(n, a, s)

    q_house, r_house = qr_decomp_house(a.copy())
    q_lib, r_lib = qr_decomp_lib(a)

    x_house = compute_x(q_house, r_house, b)
    x_lib = compute_x(q_lib, r_lib, b)
    euclidian_norm = np.linalg.norm(x_house - x_lib)
    print(euclidian_norm)

if __name__ == "__main__":
    n = sys.argv[1]
    if n is not None:
        n = int(n)
    main(n)
