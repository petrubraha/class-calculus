import time
import random
import math
import lib

UNIT = 1.0
"""Global float constant used internally."""

def find_machine_precision():
    """Returns the machine precision and the computed power (m)."""
    
    # The number of iterations = m = power.
    power = 0
    old_precision = 10
    machine_precision = 0.1
    
    while UNIT + machine_precision > UNIT:
        old_precision = machine_precision
        machine_precision /= 10
        power += 1

    # UNIT + old_precision / 10 == UNIT 
    return old_precision, power - 1

if __name__ == "__main__":
    machine_precision, power = find_machine_precision()
    print(f"Machine precision: {machine_precision}, m: {power}")
    print(f"First verification (should NOT be 1.0): {UNIT + machine_precision}")
    print(f"Second verification (should be 1.0): {UNIT + machine_precision / 10}")


# hmm interesant if __name__== "__main__" :D 

print("="*74)
#EXERCITIUL 2
u, _ = find_machine_precision()
x = 1.0
y = u / 10
z = u / 10

st = (x + y) + z    # (1.0 + u/10) + u/10
dr = x + (y + z)    # 1.0 + (u/10 + u/10)

if st!=dr:
    print(f"(x + y) + z = {st}")
    print(f"x + (y + z) = {dr}")
    print(f"Neasociativ  {st!=dr}")
# acum trebuie sa gasim x y z pentru care inmultirea ESTE ASOCIATIVA

s=time.time()
for incercare in range(1,100_001):
    x = 10 ** random.uniform(40, 200)
    y = 10 ** random.uniform(-180, -100)
    z = 10 ** random.uniform(-155, -90)
    st = (x * y) * z
    dr = x * (y * z)

    if st != dr and st > 0 and dr == 0.0:
        print(f"Gasit dupa {incercare} incercari \n x = {x:.2e}, y = {y:.2e}, z = {z:.2e}")
        print(f"(x*y)*z = {st:.2e}")
        print(f"x*(y*z) = {dr}")
        break

print(f"Timp total {time.time()-s:.4f} secunde")

#EXERCITIUL 3
print("=" * 74)
avg_dif_lib_frac = 0.0
avg_dif_lib_poly = 0.0

c_timp_lib = 0.0 
c_timp_frac = 0.0 
c_timp_poly = 0.0

for i in range(0, 10_000):
    input = random.uniform(-math.pi / 2, math.pi / 2)
    while input == -math.pi / 2 or input == math.pi / 2:
        input = random.uniform(-math.pi / 2, math.pi / 2)
    
    try:
        input = lib.normalize(input)
    except IOError as e:
        print(f"Iteratia {i}: {e}")
        continue

    start_time = time.time()
    res_lib = math.tan(input)
    t_lib = time.time() - start_time
    c_timp_lib += t_lib

    start_time = time.time()
    res_frac = lib.tan_cont_frac(input)
    t_frac = time.time() - start_time
    c_timp_frac += t_frac
    dif_lib_frac = abs(res_lib - res_frac)
    avg_dif_lib_frac += dif_lib_frac

    start_time = time.time()
    res_poly = lib.tan_poly_approx(input)
    t_poly = time.time() - start_time
    c_timp_poly += t_poly
    dif_lib_poly = abs(res_lib - res_poly)
    avg_dif_lib_poly += dif_lib_poly

    lib.print_to_file("logger.txt", i, input, res_lib, res_frac, res_poly, t_lib, t_frac, t_poly)

avg_dif_lib_frac /= 10_000
avg_dif_lib_poly /= 10_000

print(f"Diferenta in medie dintre lib si fractii continue: {avg_dif_lib_frac:.2e}")
print(f"Diferenta in medie dintre lib si folosind polinoame: {avg_dif_lib_poly:.2e}")
print("=" * 74)
print(f"Timp librarie: {c_timp_lib:.4f} seconds")
print(f"Timp Lentz modificat: {c_timp_frac:.4f} seconds")
print(f"Timp polinoame: {c_timp_poly:.4f} seconds")