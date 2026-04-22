import numpy as np
from numpy.linalg import norm, svd, inv, matrix_rank, cond, cholesky, eigvalsh, pinv

np.set_printoptions(precision=8, suppress=True, linewidth=120)

#metoda jacobi pentru valori si vectori proprii la matrice simetrice (p = n)
def jacobi(A, eps=1e-10, k_max=1000):
    n, m = A.shape
    #verificam sa fie patrata si simetrica
    assert n == m, f"trebuie matrice patrata, avem {n}x{m}"
    assert np.allclose(A, A.T, atol=1e-10), "matricea trebuie sa fie simetrica"
    
    A_init = A.copy() #tinem o copie pentru verificarea de la final
    A = A.copy().astype(float)
    U = np.eye(n)
    k = 0
    
    #cautam (p, q) cel mai mare element in modul din partea strict inferior triunghiulara
    def gaseste_pq(A):
        p, q = 1, 0
        max_val = abs(A[1, 0])
        for i in range(2, n):
            for j in range(i):
                if abs(A[i, j]) > max_val:
                    max_val = abs(A[i, j])
                    p, q = i, j
        return p, q, max_val
    
    p, q, max_val = gaseste_pq(A)
    
    while max_val > eps and k <= k_max:
        #calculam c, s, t pentru unghiul de rotatie
        alpha = (A[p, p] - A[q, q]) / (2 * A[p, q])
        if alpha >= 0:
            t = -alpha + np.sqrt(alpha**2 + 1)
        else:
            t = -alpha - np.sqrt(alpha**2 + 1)
        c = 1.0 / np.sqrt(1 + t**2)
        s = t / np.sqrt(1 + t**2)
        
        #actualizam A direct, fara matrice auxiliara (formulele (5) din curs)
        for j in range(n):
            if j != p and j != q:
                a_pj = A[p, j]
                a_qj = A[q, j]
                A[p, j] = c * a_pj + s * a_qj
                A[q, j] = -s * a_pj + c * a_qj
                A[j, p] = A[p, j]
                A[j, q] = A[q, j]
        
        a_pq = A[p, q]
        A[p, p] = A[p, p] + t * a_pq
        A[q, q] = A[q, q] - t * a_pq
        A[p, q] = 0.0
        A[q, p] = 0.0
        
        #actualizam U direct, doar coloanele p si q (formulele (7) din curs)
        for i in range(n):
            u_ip = U[i, p]
            U[i, p] = c * u_ip + s * U[i, q]
            U[i, q] = -s * u_ip + c * U[i, q]
        
        k += 1
        p, q, max_val = gaseste_pq(A)
    
    #valorile proprii sunt pe diagonala matricei finale
    val_proprii = np.diag(A).copy()
    
    print(f"\niteratii jacobi: {k}")
    print(f"valori proprii (jacobi): {np.sort(val_proprii)}")
    
    #comparam cu numpy (eigvalsh e specializat pe matrice simetrice)
    val_numpy = np.sort(eigvalsh(A_init))
    print(f"valori proprii (numpy):  {val_numpy}")
    
    #verificarea ceruta: ||A_init * U - U * Lambda||
    Lambda = np.diag(val_proprii)
    diff = norm(A_init @ U - U @ Lambda)
    print(f"||A_init * U - U * Lambda|| = {diff:.2e}")
    
    return val_proprii, U

#sirul de matrice prin factorizari cholesky iterate
def cholesky_iterat(A, eps=1e-10, k_max=1000):
    n, m = A.shape
    #verificam sa fie patrata si simetrica
    assert n == m, f"trebuie matrice patrata, avem {n}x{m}"
    assert np.allclose(A, A.T, atol=1e-10), "matricea trebuie sa fie simetrica"
    
    A_curr = A.copy().astype(float)
    k = 0
    
    #sirul A^(k) = L_k * L_k^T, A^(k+1) = L_k^T * L_k
    while k <= k_max:
        try:
            #numpy.linalg.cholesky returneaza L triunghiulara inferioara (A = L * L^T)
            L = cholesky(A_curr)
        except np.linalg.LinAlgError:
            print(f"[eroare] matricea nu e pozitiv definita la iteratia {k}")
            return None
        
        A_next = L.T @ L
        diff = norm(A_next - A_curr)
        A_curr = A_next
        k += 1
        
        #conditia de oprire: diferenta intre doua matrice consecutive e suficient de mica
        if diff < eps:
            break
    
    print(f"\niteratii cholesky: {k}")
    print("ultima matrice calculata A^(k):")
    print(A_curr)
    
    #pe diagonala apar valorile proprii (in ordine descrescatoare)
    val_proprii = np.diag(A_curr)
    print(f"valori proprii (cholesky iterat): {val_proprii}")
    
    val_numpy = np.sort(eigvalsh(A))[::-1] #descrescator ca sa se potriveasca
    print(f"valori proprii (numpy, desc):     {val_numpy}")
    
    return A_curr

#analiza svd pentru p > n
def svd_analysis(A):
    p, n = A.shape
    #verificam sa fie matrice inalta
    assert p > n, f"trebuie p > n, avem {p}x{n}"
    
    #calculam svd propriu-zis
    U_svd, sigma, Vt = svd(A, full_matrices=True)
    V = Vt.T #il transpunem ca sa fie ca in formulele de la curs
    
    print(f"\ndim dimensiuni: A e {p}x{n}")
    
    #valorile singulare sunt deja sortate
    print(f"valori singulare (sigma): {sigma}")
    
    #rangul calculat in doua feluri
    #manual folosind o toleranta mica
    tol = max(p, n) * np.finfo(float).eps * sigma[0]
    rang_manual = np.sum(sigma > tol)
    rang_numpy = matrix_rank(A)
    
    print(f"rang manual: {rang_manual}")
    print(f"rang numpy: {rang_numpy}")
    
    #numarul de conditionare
    sigma_pos = sigma[sigma > tol]
    cond_manual = sigma_pos[0] / sigma_pos[-1]
    cond_numpy = cond(A)
    
    print(f"cond manual: {cond_manual:.8f}")
    print(f"cond numpy: {cond_numpy:.8f}")
    
    #pseudoinversa moore-penrose A^I
    #facem manual matricea S_I de n x p
    S_I = np.zeros((n, p))
    for i in range(len(sigma)):
        if sigma[i] > tol:
            S_I[i, i] = 1.0 / sigma[i]
    
    A_I = V @ S_I @ U_svd.T
    print("\npseudoinversa moore-penrose (A_I):")
    print(A_I)
    
    #pseudoinversa least-squares A^J
    #asta merge doar daca rangul e maxim
    AtA = A.T @ A
    try:
        AtA_inv = pinv(AtA)
        A_J = AtA_inv @ A.T
        print("\npseudoinversa least-squares (A_J):")
        print(A_J)
        
        #verificam cat de aproape sunt
        diff_norm = norm(A_I - A_J, 1)
        print(f"\ndiferenta norma 1: {diff_norm:.2e}")
        
    except np.linalg.LinAlgError:
        print("\n[eroare] AtA nu e inversabila, nu putem calcula A_J")
    
    return sigma, rang_manual, cond_manual, A_I

#teste jacobi
print("="*20)
print("jacobi test 1: (valori proprii -1, 0, 2)")
J1 = np.array([
    [0, 0, 1],
    [0, 0, 1],
    [1, 1, 1]
], dtype=float)
jacobi(J1)

print("\n" + "="*20)
print("jacobi test 2: (valori 0, 2(1-sqrt2), 2(1+sqrt2))")
J2 = np.array([
    [1, 1, 2],
    [1, 1, 2],
    [2, 2, 2]
], dtype=float)
jacobi(J2)

print("\n" + "="*20)
print("jacobi test 3: matrice 4x4 (valori 0, 0, 2(4+-sqrt21))")
J3 = np.array([
    [1, 2, 3, 4],
    [2, 3, 4, 5],
    [3, 4, 5, 6],
    [4, 5, 6, 7]
], dtype=float)
jacobi(J3)

#teste cholesky iterat - trebuie matrice pozitiv definite
print("\n" + "="*20)
print("cholesky test 1: matrice 3x3 diagonal dominanta")
C1 = np.array([
    [4, 1, 1],
    [1, 3, 0],
    [1, 0, 2]
], dtype=float)
cholesky_iterat(C1)

print("\n" + "="*20)
print("cholesky test 2: matrice 2x2 (valori proprii 1 si 3)")
C2 = np.array([
    [2, 1],
    [1, 2]
], dtype=float)
cholesky_iterat(C2)

print("\n" + "="*20)
print("cholesky test 3: matrice 4x4 pozitiv definita")
M = np.array([
    [1, 2, 0, 1],
    [0, 1, 1, 0],
    [2, 0, 1, 1],
    [1, 1, 0, 2]
], dtype=float)
C3 = M.T @ M #garantat pozitiv definita
cholesky_iterat(C3)

#teste svd
print("\n" + "="*20)
print("svd test 1: matrice 5x3 cu rang mic")
A1 = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [2, 1, 0],
    [3, 3, 3]
], dtype=float)
svd_analysis(A1)

print("\n" + "="*20)
print("svd test 2: matrice 4x2 cu rang maxim")
A2 = np.array([
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8]
], dtype=float)
svd_analysis(A2)

print("\n" + "="*20)
print("svd test 3: matrice 6x3 cu dependente")
A3 = np.array([
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 2],
    [2, 1, 3],
    [0, 0, 0],
    [3, 2, 5]
], dtype=float)
svd_analysis(A3)