import numpy as np
import constants as c
import util as u

# ----------------------------------------
# P(x)  = x^3 - 6x^2 + 11x - 6
# P'(x) = 3x^2 - 12x + 11
# P''(x)= 6x - 12
# roots = [1, 2, 3]
# ----------------------------------------
POLY_COEFFS       = [1.0, -6.0, 11.0, -6.0]
FIRST_DERIV_COEFFS = [3.0, -12.0, 11.0]
SECOND_DERIV_COEFFS = [6.0, -12.0]

def evaluate_horner(coeffs, v):
    d = coeffs[0]
    for i in range(1, len(coeffs)):
        d = d * v + coeffs[i]
    return d

def P(x):
    return evaluate_horner(POLY_COEFFS, x)

def P_prime(x):
    return evaluate_horner(FIRST_DERIV_COEFFS, x)

def P_double_prime(x):
    return evaluate_horner(SECOND_DERIV_COEFFS, x)

def newton_delta(x):
    return P(x) / P_prime(x)

def olver_delta(x):
    px = P(x)
    ppx = P_prime(x)
    pppx = P_double_prime(x)
    c_k = (px ** 2) * pppx / (ppx ** 3)
    return px / ppx + 0.5 * c_k

def compute_R(coeffs):
    a0 = abs(coeffs[0])
    A = max(abs(c) for c in coeffs[1:])
    return (a0 + A) / a0

# ----------------------------------------
# Core functionality and loop
# ----------------------------------------
def compute_sequence(x0, delta_func):
    x = x0
    k = 0

    while True:
        if abs(P_prime(x)) <= c.EPSILON:
            return None, k

        dx = delta_func(x)
        x  = x - dx
        k += 1

        if abs(dx) < c.MIN_DIFF:
            break
        if abs(dx) > c.MAX_DIFF:
            return None, k
        if k >= c.MAX_ITERATION_COUNT:
            return None, k

    return x, k

# ----------------------------------------
def main():
    R = compute_R(POLY_COEFFS)
    print(f"All real roots lie in  [-{R:.6f}, {R:.6f}]\n")

    initial_points = np.linspace(-R, R, 200)

    newton_roots = []
    olver_roots  = []

    for x0 in initial_points:
        # --- Newton ---
        root_n, iters_n = compute_sequence(x0, newton_delta)
        if root_n is not None and abs(P(root_n)) < c.EPSILON:
            u.add_if_distinct(newton_roots, root_n)

        # --- Olver ---
        root_o, iters_o = compute_sequence(x0, olver_delta)
        if root_o is not None and abs(P(root_o)) < c.EPSILON:
            u.add_if_distinct(olver_roots, root_o)

    # ---- Display on screen ----
    print("=" * 60)
    print("Newton's Method - distinct roots found:")
    print("=" * 60)
    for i, r in enumerate(newton_roots, 1):
        _, iters = compute_sequence(r, newton_delta)
        print(f"  root {i}: {r:>20.12f}   (verification P(r) = {P(r):.2e})")
    print(f"  Total distinct roots: {len(newton_roots)}\n")

    print("=" * 60)
    print("Olver's Method - distinct roots found:")
    print("=" * 60)
    for i, r in enumerate(olver_roots, 1):
        print(f"  root {i}: {r:>20.12f}   (verification P(r) = {P(r):.2e})")
    print(f"  Total distinct roots: {len(olver_roots)}\n")

    # ---- Convergence comparison ----
    print("=" * 60)
    print("Convergence comparison  (starting from x0 = 0.5)")
    print("=" * 60)
    x0_test = 0.5
    rn, itn = compute_sequence(x0_test, newton_delta)
    ro, ito = compute_sequence(x0_test, olver_delta)
    print(f"  Newton:  root ~= {rn:.12f},  iterations = {itn}")
    print(f"  Olver :  root ~= {ro:.12f},  iterations = {ito}")
    print()

    # ---- Write results to file ----
    output_path = "results.txt"
    with open(output_path, "w") as f:
        f.write(f"Polynomial: P(x) = x^3 - 6x^2 + 11x - 6\n")
        f.write(f"Coefficients: {POLY_COEFFS}\n")
        f.write(f"Root interval: [-{R:.6f}, {R:.6f}]\n")
        f.write(f"Precision epsilon = {c.EPSILON}\n\n")

        f.write("Newton's Method - distinct roots:\n")
        for i, r in enumerate(newton_roots, 1):
            f.write(f"  root {i}: {r:.12f}   P(r) = {P(r):.2e}\n")
        f.write(f"  Total: {len(newton_roots)}\n\n")

        f.write("Olver's Method - distinct roots:\n")
        for i, r in enumerate(olver_roots, 1):
            f.write(f"  root {i}: {r:.12f}   P(r) = {P(r):.2e}\n")
        f.write(f"  Total: {len(olver_roots)}\n\n")

        f.write("Convergence comparison (x0 = 0.5):\n")
        f.write(f"  Newton:  root = {rn:.12f},  iterations = {itn}\n")
        f.write(f"  Olver :  root = {ro:.12f},  iterations = {ito}\n")

    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
