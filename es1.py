
"""ESERCIZIO 1"""

import numpy as np
import matplotlib.pyplot as plt

""" Funzione e Nodi equispaziati """
def f(x):
    return np.exp(-x) * np.cos(2*x)
a = -2
b = 2

x_equi = np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
f_equi = f(x_equi)

print("\nNodi equispaziati e rispettive ordinate: ")
print("     xi          f(x)")

for xi, yi in zip(x_equi, f_equi):
    print(f"{xi:8.2f}, {yi:12.6f}")

""" Matrice di Vandermonde """

def vandermonde(xnodi):
    n = xnodi.shape[0]
    A = np.zeros((n,n))
    for j in range(n):
        A[:, j] = xnodi**j
    return A

""" Fattorizzazione LU con piovot totale """

def LU(A): 
    A = np.array(A, dtype = float)
    n, m = A.shape
    if n != m:
        raise ValueError("\nLa matrice deve essere quadrata")

    U = A.copy()
    L = np.eye(n)
    P = np.eye(n)
    Q = np.eye(n)

    for i in range(n-1):
        
        # calcolo il pivot totale
        sub = np.abs(U[i:, i:])
        r, c = np.unravel_index(np.argmax(sub), sub.shape)
        r = r + i
        c = c + i

        if abs(U[r, c]) < np.finfo(float).eps:
            raise ValueError ("Pivot null: matrice singolare o quasi singolare")
        
        # scambio le righe

        if r != i:
            U[[i, r], :] = U[[r, i], :]
            P[[i, r], :] = P[[r, i], :]
            
            if i > 0: 
                L[[i, r], :i] = L[[r, i], :i]

        # scambio le colonne
        if c != i:
            U[: , [i, c]] = U[: , [c, i]]
            Q[: , [i, c]] = Q[: , [c, i]] 
        
        # Eliminazione di GAUSS

        for j in range (i+1 , n):
            L[j, i] = U[j, i] / U[i, i]
            U[j, i:] = U[j, i:] - L[j, i] * U[i,i:]
            U[j, i] = 0.0
    
    return L, U, P, Q

""" Risoluzione della Fattorizzazione """

def sol_LU(A, b):
    L, U, P, Q = LU(A)
    y = np.linalg.solve(L, P @ b)
    z = np.linalg.solve(U, y)

    x = Q @ z

    return x, L, U, P, Q

""" Polinimio interpolare con nodi equispaziati """

A_equi = vandermonde(x_equi)

# coef_equi --> COEFFICIENTE DI EQUIVALENZA DEL POLINOMIO

coef_equi, L_equi, U_equi, P_equi, Q_equi = sol_LU(A_equi, f_equi)

print("\nCoefficienti del polinomio con nodi equispaziati: ")
print(coef_equi)

def valuta_polinomio(coef, x):
    return np.polyval(coef[::-1], x)

""" Nodi di Chebyshev """

def cheby(a, b, n):
    k = np.arange(0, n+1)
    x = (a + b) / 2 + (b - a) / 2 * np.cos((2 * k + 1) * np.pi / (2 * (n + 1)))
    return x

n = 8

x_cheby = cheby(a, b, n)
f_cheby = f(x_cheby)

A_cheby = vandermonde(x_cheby)

coef_cheby, L_cheby, U_cheby, P_cheby, Q_cheby = sol_LU(A_cheby, f_cheby)

# controllo di sicurezza: verifica che il polinomio costruito con i nodi di Chebyshev passi davvero per i punti di interpolazione
# l'errore dev'essere circa nullo, salvo arrotondamenti numerici

errore_nodi_cheby = np.linalg.norm(valuta_polinomio(coef_cheby, x_cheby) - f_cheby, np.inf)
print("\nErrore sui nodi di Chebyshev:", errore_nodi_cheby)

print("\nNodi di Chebyshev: ")
print(x_cheby)

print("\nCoeffincienti del polinomio con nodi di Chebyshev: ")
print(coef_cheby)


"""Tabella di conzionamento delle matrice di Vandermonde e delle matrici di Fattorrizzazione """
    # calcolo dei punti

xx = np.linspace(a, b, 550)
ff = f(xx)
p_equi = valuta_polinomio(coef_equi, xx)
p_cheby = valuta_polinomio(coef_cheby, xx)

# calcolo degli errori

errore_equi  = np.linalg.norm(p_equi - ff, np.inf)
errore_cheby = np.linalg.norm(p_cheby - ff, np.inf)

residuo_equi = np.linalg.norm(A_equi @ coef_equi - f_equi, np.inf) / np.linalg.norm(f_equi, np.inf)
residuo_cheby = np.linalg.norm(A_cheby @ coef_cheby - f_cheby, np.inf) / np.linalg.norm(f_cheby, np.inf)

# creazione tabella 
""" Tabella del condizionamento e degli errori """

print("\n")
print("_" * 96)
print("{:<15} {:>15} {:>15} {:>15} {:>15} {:>15}".format(
    "Nodi", "cond(A)", "cond(L)", "cond(U)", "Residuo", "Errore max"))
print("_" * 96)

print("{:<15} {:>15.4e} {:>15.4e} {:>15.4e} {:>15.4e} {:>15.4e}".format(
    "Equispaziati",
    np.linalg.cond(A_equi),
    np.linalg.cond(L_equi),
    np.linalg.cond(U_equi),
    residuo_equi,
    errore_equi))

print("{:<15} {:>15.4e} {:>15.4e} {:>15.4e} {:>15.4e} {:>15.4e}".format(
    "Chebyshev",
    np.linalg.cond(A_cheby),
    np.linalg.cond(L_cheby),
    np.linalg.cond(U_cheby),
    residuo_cheby,
    errore_cheby))

print("_" * 96)


""" Grafico """

plt.figure(figsize=(9, 6))

plt.plot(xx, ff, label = "funzione f(x)", linewidth = 2)
plt.plot(xx, p_equi, "--", label = "interpolazione nodi equispaziati")
plt.plot(xx, p_cheby, "-.", label = "interpolazione nodi di Chebyshev")
plt.plot(x_equi, f_equi, "o", label = "nodi equispaziati")
plt.plot(x_cheby, f_cheby, "*", label = "nodi di Chebyshev")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Interpolazione di f(x) = e^{-x}cos(2x)")
plt.legend()
plt.grid(True)
plt.show() 