import math
import numpy as np
import random
import scipy as sp
import sys

EPSILON = 1e-8

def display_matrix(matrix, name: str = "Matrix"):
    """functie care afiseaza matriciile
    also good for debugging """
    print(f"\n{name}:")
    print(matrix)
    print()

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
        sigma += a[i, r]**2 # la sigma a trebuie sa aduci patratele elementelor de la r -> n
    return sigma

def _transform_a(a, u, r: int, beta: float):
    n = len(a)
    for j in range(r + 1, n):
        gamma = 0.0
        for i in range(r, n):
            gamma += u[i] * a[i, j]
        gamma /= beta  # normalizam cu (BETA) = ||u||^2
 
        # a[i,j] = a[i,j] - gama * u[i]  pentru i = r->n
        for i in range(r, n):
            a[i, j] -= gamma * u[i]

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
        a[r, r] = k
        for i in range(r + 1, n):
            a[i, r] = 0.0

        for j in range(n):
            gamma = 0.0
            for i in range(r, n):
                gamma += u[i] * q[i, j]
            gamma /= beta
            for i in range(r, n):
                q[i, j] -= gamma * u[i]
        
    return q.T, a

def qr_decomp_lib(a):
    return np.linalg.qr(a)

def compute_x(q, r, b):
    y = np.dot(q.T, b)
    x = sp.linalg.solve(r, y)
    return x

def compute_inverse_house(q, r):
    """Calculam inversa matricei A folosind descompunerea QR (Householder)
    Input:  Q matricea ortogonala
            R matricea superior triunghiulara

    Output: Inv (A^-1) care este inversa lui A calculata folosind metoda Household
    
    """
    n = len(r)
    
    #matricea rezultat
    inv = np.zeros((n, n))
 
    for j in range(n):
        # b = Q^T * e[j] = coloana j din Q^T = linia j din Q
        b = q.T[:, j]   #ma rog aici doar facem niste sliceing pentru ca putem accesa diferit matriciile 
 
        # rezolva Rx = b — x devine coloana j din A^{-1}
        inv[:, j] = sp.linalg.solve_triangular(r, b)
 
    return inv


def main(n: int | None):
    if n == None:
        n = random.randrange(2, 3)
    a = np.random.rand(n, n)
    s = np.random.rand(n)
    b = compute_b(n, a, s)

    a_init = a.copy()
    b_init = b.copy()

    # Afisam matricele la inceput
    display_matrix(a_init, "Matricea A")
    display_matrix(s, "Vectorul s")
    display_matrix(b_init, "Vectorul b")

    q_house, r_house = qr_decomp_house(a.copy())
    q_lib, r_lib = qr_decomp_lib(a)
    display_matrix(q_lib, "r_lib")
    display_matrix(q_house, "r_house")

    x_house = compute_x(q_house, r_house, b)
    x_lib = compute_x(q_lib, r_lib, b)
    euclidian_norm = np.linalg.norm(x_house - x_lib)
    print(euclidian_norm)

    #punctul 4
    err_house_b = np.linalg.norm(a_init @ x_house - b_init)
    print(f"Cat de bine satisface x_house sistemul: {err_house_b:.2e}")

    err_lib_b = np.linalg.norm(a_init @ x_lib - b_init)
    print(f"Cat de bine satisface x_lib sistemul {err_lib_b:.2e}")

    err_house_s = np.linalg.norm(x_house - s) / np.linalg.norm(s)
    print(f"Eroare fata de s (x_house) {err_house_s:.2e}")

    err_lib_s = np.linalg.norm(x_lib - s) / np.linalg.norm(s)
    print(f"Eroarem fata de s (x_lib)= {err_lib_s:.2e}")

    #punctul 5

    # calculeaza A^-1 prin QR Householder (coloana cu coloana)
    inv_house = compute_inverse_house(q_house, r_house)
 
    # calculeaza A^-1 prin librarie (referinta)
    inv_lib = np.linalg.inv(a_init)
 
    # ||A^{-1}_house - A^{-1}_lib||_2 — diferenaa dintre cele doua inverse
    print(f"||inv_house - inv_lib||= {np.linalg.norm(inv_house - inv_lib):.2e}")


if __name__ == "__main__":
    
    #punctul 6t ig? nu era deja fct ? :))
    n = sys.argv[1]
    if n is not None:
        n = int(n)
    main(n)
