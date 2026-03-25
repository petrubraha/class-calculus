import numpy as np
import sys
import os

# ── Constante ────────────────────────────────────────────────────────────────

EPSILON = 1e-8   # precizie de calcul (default)
K_MAX   = 10000  # numar maxim de iteratii Gauss-Seidel
DIV_THR = 1e10   # prag de divergenta

# ── Citire date ───────────────────────────────────────────────────────────────

def load_vector(path: str) -> np.ndarray:
    # citeste un vector dintr-un fisier, cate un element pe linie
    return np.loadtxt(path)

def load_system(folder: str, idx: int):
    """
    Incarca un sistem liniar Ax = b din fisierele:
        d0_i.txt  — diagonala principala
        d1_i.txt  — diagonala secundara de ordin p
        d2_i.txt  — diagonala secundara de ordin q
        b_i.txt   — vectorul termenilor liberi

    Returneaza: d0, d1, d2, b  (np.ndarray)
    """
    d0 = load_vector(os.path.join(folder, f"d0_{idx}.txt"))
    d1 = load_vector(os.path.join(folder, f"d1_{idx}.txt"))
    d2 = load_vector(os.path.join(folder, f"d2_{idx}.txt"))
    b  = load_vector(os.path.join(folder, f"b_{idx}.txt"))
    return d0, d1, d2, b

# ── Punctul 1: dimensiunea sistemului ────────────────────────────────────────

def get_n(d0: np.ndarray, b: np.ndarray) -> int:
    # n = numarul de elemente din d0 (= numarul de elemente din b)
    # cele doua trebuie sa coincida
    assert len(d0) == len(b), "d0 si b au dimensiuni diferite!"
    return len(d0)

# ── Punctul 2: ordinele diagonalelor p si q ───────────────────────────────────

def get_p_q(n: int, d1: np.ndarray, d2: np.ndarray):
    """
    Diagonala secundara de ordin p are n-p elemente:
        a[0,p], a[1,p+1], ..., a[n-1-p, n-1]  => |d1| = n - p  => p = n - |d1|

    Diagonala secundara de ordin q are n-q elemente:
        a[q,0], a[q+1,1], ..., a[n-1, n-1-q]  => |d2| = n - q  => q = n - |d2|
    """
    p = n - len(d1)
    q = n - len(d2)
    return p, q

# ── Punctul 3: verificare diagonala principala ────────────────────────────────

def check_diagonal(d0: np.ndarray, eps: float) -> bool:
    # toate elementele de pe diagonala principala trebuie sa fie nenule
    # |a[i,i]| > eps  pentru orice i
    return np.all(np.abs(d0) > eps)

# ── Punctul 4: Gauss-Seidel ───────────────────────────────────────────────────

def gauss_seidel(d0, d1, d2, b, p, q, eps):
    """
    Metoda Gauss-Seidel pentru sistemul Ax = b cu matrice rara.

    Matricea A are elemente nenule doar pe:
        - diagonala principala:  a[i,i]   = d0[i]
        - diagonala de ordin p:  a[i,i+p] = d1[i]  (si simetric a[i+p,i] = d1[i])
        - diagonala de ordin q:  a[i,i+q] = d2[i]  (si simetric a[i+q,i] = d2[i])

    Formula Gauss-Seidel pentru componenta i:
        x_new[i] = ( b[i]
                   - a[i, i-p] * x_new[i-p]   <- deja actualizat (j < i)
                   - a[i, i+p] * x_old[i+p]   <- inca neactualizat (j > i)
                   - a[i, i-q] * x_new[i-q]   <- deja actualizat (j < i)
                   - a[i, i+q] * x_old[i+q]   <- inca neactualizat (j > i)
                   ) / a[i,i]

    Conditie de oprire: ||x_new - x_old||_2 < eps
    """
    n  = len(d0)
    xp = np.zeros(n)   # x^(k)   — iteratia precedenta
    xc = np.zeros(n)   # x^(k+1) — iteratia curenta

    for k in range(K_MAX):
        xp[:] = xc     # salveaza iteratia anterioara

        for i in range(n):
            s = b[i]   # incepem cu termenul liber b[i]

            # contributia diagonalei d1 (ordin p):
            #   stanga:  a[i, i-p] = d1[i-p]  daca i-p >= 0  (x deja actualizat)
            if i >= p:
                s -= d1[i - p] * xc[i - p]
            #   dreapta: a[i, i+p] = d1[i]    daca i+p < n   (x din iteratia anterioara)
            if i + p < n:
                s -= d1[i] * xp[i + p]

            # contributia diagonalei d2 (ordin q):
            #   stanga:  a[i, i-q] = d2[i-q]  daca i-q >= 0  (x deja actualizat)
            if i >= q:
                s -= d2[i - q] * xc[i - q]
            #   dreapta: a[i, i+q] = d2[i]    daca i+q < n   (x din iteratia anterioara)
            if i + q < n:
                s -= d2[i] * xp[i + q]

            # imparte la elementul diagonal (a[i,i] = d0[i])
            xc[i] = s / d0[i]

        # criteriu de oprire: ||x_new - x_old||_2 < eps
        delta = np.linalg.norm(xc - xp)

        if delta < eps:
            print(f"  Convergenta la iteratia k={k+1}, delta={delta:.2e}")
            return xc   # solutia aproximativa

        # criteriu de divergenta
        if delta > DIV_THR:
            print(f"  Divergenta la iteratia k={k+1}, delta={delta:.2e}")
            return None

    print(f"  Nu a convergat in {K_MAX} iteratii, delta={delta:.2e}")
    return None

# ── Punctul 5: calcul y = A * xGS ────────────────────────────────────────────

def matvec(d0, d1, d2, x, p, q):
    """
    Calculeaza y = A * x fara a construi matricea A explicit.

    Fiecare element y[i] = suma produselor elementelor nenule de pe linia i cu x:
        y[i] = d0[i] * x[i]
             + d1[i-p] * x[i-p]   daca i >= p
             + d1[i]   * x[i+p]   daca i+p < n
             + d2[i-q] * x[i-q]   daca i >= q
             + d2[i]   * x[i+q]   daca i+q < n

    Fiecare element din d1, d2 este accesat exact de doua ori
    (o data ca element 'stanga', o data ca element 'dreapta').
    """
    n = len(d0)
    y = np.zeros(n)

    for i in range(n):
        y[i] = d0[i] * x[i]          # contributia diagonalei principale

        if i >= p:                    # a[i, i-p] = d1[i-p]
            y[i] += d1[i - p] * x[i - p]
        if i + p < n:                 # a[i, i+p] = d1[i]
            y[i] += d1[i] * x[i + p]

        if i >= q:                    # a[i, i-q] = d2[i-q]
            y[i] += d2[i - q] * x[i - q]
        if i + q < n:                 # a[i, i+q] = d2[i]
            y[i] += d2[i] * x[i + q]

    return y

# ── Punctul 6: norma infinit ||A*xGS - b||∞ ──────────────────────────────────

def inf_norm(v: np.ndarray) -> float:
    # ||v||∞ = max |v[i]|  —  cea mai mare componenta in valoare absoluta
    return np.max(np.abs(v))

# ── Main ──────────────────────────────────────────────────────────────────────

def solve_system(folder: str, idx: int, eps: float):
    print(f"\n{'='*50}")
    print(f"Sistem {idx}  (eps={eps:.0e})")
    print(f"{'='*50}")

    # incarca fisierele
    d0, d1, d2, b = load_system(folder, idx)

    # ── Punctul 1 ──
    n = get_n(d0, b)
    print(f"[1] Dimensiunea sistemului: n = {n}")

    # ── Punctul 2 ──
    p, q = get_p_q(n, d1, d2)
    print(f"[2] Ordine diagonale: p = {p}, q = {q}")

    # ── Punctul 3 ──
    if not check_diagonal(d0, eps):
        print("[3] EROARE: diagonala principala contine elemente nule!")
        return
    print(f"[3] Diagonala principala: toate elementele sunt nenule (OK)")

    # ── Punctul 4 ──
    print(f"[4] Rulam Gauss-Seidel...")
    x_gs = gauss_seidel(d0, d1, d2, b, p, q, eps)

    if x_gs is None:
        print("[4] Solutia nu a putut fi aproximata (divergenta).")
        return

    print(f"    Primele 5 componente ale solutiei: {x_gs[:5]}")

    # ── Punctul 5 ──
    y = matvec(d0, d1, d2, x_gs, p, q)
    print(f"[5] y = A * xGS calculat")

    # ── Punctul 6 ──
    residual = inf_norm(y - b)
    print(f"[6] ||A*xGS - b||∞ = {residual:.6e}")


def main():
    # folderul cu datele poate fi dat ca argument, default = "data"
    folder = sys.argv[1] if len(sys.argv) > 1 else "data"

    # precizia poate fi data ca al doilea argument, default = 1e-8
    eps = float(sys.argv[2]) if len(sys.argv) > 2 else EPSILON

    # ruleaza pentru toate cele 5 sisteme care au fisierul b disponibil
    for idx in range(1, 6):
        b_path = os.path.join(folder, f"b_{idx}.txt")
        if not os.path.exists(b_path):
            print(f"\nSistem {idx}: fisierul b_{idx}.txt lipseste, skip.")
            continue
        solve_system(folder, idx, eps)


if __name__ == "__main__":
    main()