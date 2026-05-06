import math
import logging
import numpy as np


# logger 
# scrie pasii intr-un fisier txt, stdout-ul ramane neatins
log = logging.getLogger("tema8")
log.setLevel(logging.DEBUG)
log.propagate = False  # nu vrem sa se duca si la stdout

handler = logging.FileHandler("tema8.log", mode="w", encoding="utf-8")
handler.setFormatter(logging.Formatter("%(message)s"))
log.addHandler(handler)


def sigmoid(z):
    # forma stabila numeric - evita overflow la exp(-z) cand z e negativ mare
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)


def gradient_aproximativ(F, x, h=1e-5):
    # formula cu 4 puncte(eroare h^4)
    n = len(x)
    grad = np.zeros(n)

    for i in range(n):
        x_p2 = x.copy(); x_p2[i] += 2*h
        x_p1 = x.copy(); x_p1[i] += h
        x_m1 = x.copy(); x_m1[i] -= h
        x_m2 = x.copy(); x_m2[i] -= 2*h

        grad[i] = (-F(x_p2) + 8*F(x_p1) - 8*F(x_m1) + F(x_m2)) / (12*h)

    return grad


def backtracking(F, x, grad, beta=0.8):
    # pornim cu eta=1,il scadem daca F creste ff putin
    eta = 1.0
    p = 1
    Fx = F(x) # ca sa nu mai apelam de fiecare data functia in loop
    norm_sq = float(np.dot(grad, grad))

    while True:
        F_dupa = F(x - eta * grad)
        
        if F_dupa > Fx - (eta / 2.0) * norm_sq:
            descrestere=False
        else:
            descrestere=True

        if descrestere==True and p < 8:
            eta *= beta     # efectiv η = η*β  :)  
            p += 1
        else:
            break

    # intoarcem si p ca sa logam cate incercari am facut
    return eta, p


def gradient_descent(F, x_init, metoda_grad, grad_an=None,metoda_eta="backtracking", eta_const=1e-3,epsilon=1e-5, k_max=30000, h=1e-5):
    # schema do-while din pdf
    x = x_init.copy().astype(float)
    k = 0
    norm_pas = float('inf')
    motiv_oprire = None

    while True:
        # calculam gradientul: analitic sau aproximativ
        if metoda_grad == "analitic":
            grad = grad_an(x)
        else:
            grad = gradient_aproximativ(F, x, h=h)

        # alegem eta: constant sau backtracking
        if metoda_eta == "constant":
            eta = eta_const
            p_bt = None
        else:
            eta, p_bt = backtracking(F, x, grad)

        # pasul propriu-zis
        x_nou = x - eta * grad
        norm_grad = float(np.linalg.norm(grad))
        norm_pas = eta * norm_grad

        # logam iteratia - doar primele 10 si apoi din 100 in 100
        # ca sa nu umflam fisierul cu 30000 de linii inutile
        log_acum = (k < 10) or (k % 100 == 0)
        if log_acum:
            if p_bt is not None:
                detaliu_eta = f"eta={eta:.6f} (bt p={p_bt})"
            else:
                detaliu_eta = f"eta={eta:.6f}"

            log.debug(
                f"  k={k:>6}  x=[{x[0]:>12.6f}, {x[1]:>12.6f}]  "
                f"||grad||={norm_grad:.6e}  {detaliu_eta}  "
                f"eta*||grad||={norm_pas:.6e}  F(x)={F(x):.6f}"
            )

        x = x_nou
        k += 1

        # cele 3 conditii de stop din pdf
        if norm_pas < epsilon:
            motiv_oprire = f"convergenta (eta*||grad|| = {norm_pas:.2e} < eps)"
            break
        if k > k_max:
            motiv_oprire = f"k_max atins ({k_max})"
            break
        if norm_pas > 1e10:
            motiv_oprire = f"divergenta (eta*||grad|| = {norm_pas:.2e} > 1e10)"
            break

    if norm_pas <= epsilon:
        a_convergit = True
    else:
        a_convergit = False

    log.debug(f"STOP la k={k}: {motiv_oprire}")
    log.debug(f"x_final = [{x[0]:.6f}, {x[1]:.6f}]   F(x_final) = {F(x):.6f}\n")

    return x, k, a_convergit


# functiile de test din pdf 

# f1: pierdere logistica, nu are minim finit (minimul e la infinit)
def f1(x):
    s_dif = sigmoid(x[0] - x[1])
    s_sum = sigmoid(x[0] + x[1])
    # protectie contra log(0) cand sigmoid se duce la 0 sau 1
    a = max(1.0 - s_dif, 1e-15)
    b = max(s_sum, 1e-15)
    return -math.log(a) - math.log(b)

def grad_f1(x):
    s_dif = sigmoid(x[0] - x[1])
    s_sum = sigmoid(x[0] + x[1])
    return np.array([s_dif + s_sum - 1.0, s_sum - s_dif - 1.0])


# f2: x1^2 + x2^2 - 2x1 - 4x2 - 1, minim la (1, 2)
def f2(x):
    return x[0]**2 + x[1]**2 - 2*x[0] - 4*x[1] - 1

def grad_f2(x):
    return np.array([2*x[0] - 2, 2*x[1] - 4])


# f3: 3x1^2 - 12x1 + 2x2^2 + 16x2 - 10, minim la (2, -4)
def f3(x):
    return 3*x[0]**2 - 12*x[0] + 2*x[1]**2 + 16*x[1] - 10

def grad_f3(x):
    return np.array([6*x[0] - 12, 4*x[1] + 16])


# f4: x1^2 - 4x1x2 + 4.5x2^2 - 4x2 + 3, minim la (8, 4)
# Hessianul e prost conditionat (lambda_max ~10.8, lambda_min ~0.18)
def f4(x):
    return x[0]**2 - 4*x[0]*x[1] + 4.5*x[1]**2 - 4*x[1] + 3

def grad_f4(x):
    return np.array([2*x[0] - 4*x[1], -4*x[0] + 9*x[1] - 4])


# f5: x1^2*x2 - 2x1*x2^2 + 3x1*x2 + 4, minim local la (-1, 0.5)
def f5(x):
    return x[0]**2 * x[1] - 2*x[0]*x[1]**2 + 3*x[0]*x[1] + 4

def grad_f5(x):
    return np.array([
        2*x[0]*x[1] - 2*x[1]**2 + 3*x[1],
        x[0]**2 - 4*x[0]*x[1] + 3*x[0]
    ])



probleme = [
    ("f1: logistic loss",            f1, grad_f1, np.array([0.5, 0.5]),  None),
    ("f2: x1^2+x2^2-2x1-4x2-1",      f2, grad_f2, np.array([0.0, 0.0]),  np.array([1.0, 2.0])),
    ("f3: 3x1^2-12x1+2x2^2+16x2-10", f3, grad_f3, np.array([0.0, 0.0]),  np.array([2.0, -4.0])),
    ("f4: x1^2-4x1x2+4.5x2^2-4x2+3", f4, grad_f4, np.array([7.0, 4.0]),  np.array([8.0, 4.0])),
    ("f5: x1^2*x2-2x1x2^2+3x1x2+4",  f5, grad_f5, np.array([-1.2, 0.4]), np.array([-1.0, 0.5])),
]

print("Tema 8 - gradient descent\n")
log.debug("Tema 8 - log iteratii gradient descent")
log.debug("Logam primele 10 iteratii si apoi din 100 in 100\n")

for nume, F, grad_an, x_init, x_optim in probleme:
    print("=" * 74)
    print(f"  {nume}")
    print(f"  x_init = {x_init}     x* = {x_optim}")
    print("=" * 74)
    print(f"  {'gradient':<14}{'eta':<22}{'k':<14}{'x_final':<28}{'F(x_final)':<14}")
    print("  " + "-" * 74)

    log.debug("#" * 74)
    log.debug(f"# {nume}")
    log.debug(f"# x_init = {x_init}     x* = {x_optim}")
    log.debug("#" * 74)

    for metoda_grad in ("analitic", "aproximativ"):
        for metoda_eta, eticheta_eta in (("constant", "eta=10^-3 const"),
                                          ("backtracking", "backtracking")):

            log.debug(f"\n--- gradient={metoda_grad}, eta={eticheta_eta} ---")

            x_fin, k, ok = gradient_descent(
                F, x_init, metoda_grad,
                grad_an=grad_an, metoda_eta=metoda_eta
            )

            if ok:
                stat = str(k)
            else:
                stat = f"{k} (DIV)"

            print(f"  {metoda_grad:<14}{eticheta_eta:<22}{stat:<14}"
                  f"{str(np.round(x_fin, 5)):<28}{F(x_fin):<14.6f}")
    print()
