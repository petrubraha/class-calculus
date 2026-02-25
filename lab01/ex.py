u = 1.0
while 1.0 + u != 1.0:
    u /= 10
u *= 10          # acum u satisface 1.0 + u != 1.0 și 1.0 + u/10 == 1.0

print(f"Precizia mașină u = {u}")          # ar trebui să iasă 1e-15
print(f"Verificare: 1 + u != 1     → {1.0 + u != 1.0}")
print(f"Verificare: 1 + u/10 == 1  → {1.0 + u/10 == 1.0}\n")