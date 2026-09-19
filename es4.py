
""" Esercizio 4 """

# Il punto 1 dell'esercizio 4: " Individuare graficamente la radice positiva".
# Il grafico verrà mostrato alla fine del codice in modo da non bloccare l'esecuzione.

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import pandas as pd

def f(x):
    return np.cos(x) - x

def df(x):
    return -np.sin(x) - 1

""" Metodo di Newton """

def newton(f, df, x0, atol = 1e-8, rtol = 1e-8, maxit = 100):
    x_old = x0
    
    iterazioni = 0
    valutazioni_f = 0
    valutazioni_df = 0

    for k in range(maxit):
        fx = f(x_old)
        dfx = df(x_old)

        valutazioni_f += 1
        valutazioni_df += 1

        x_new = x_old - fx / dfx

        errore_stimato = abs(x_new - x_old)
        iterazioni += 1

        if errore_stimato <= atol + rtol * abs(x_new):
            return x_new, iterazioni, valutazioni_f, valutazioni_df, f(x_new), errore_stimato
        
        x_old = x_new
    
    return x_new, iterazioni, valutazioni_f, valutazioni_df, f(x_new), errore_stimato


""" Metodo delle secanti """

def secanti(f, x0, x1, atol = 1e-8, rtol = 1e-8, maxit = 100):
    x_old = x0
    x_new = x1

    f_old = f(x_old)
    f_new = f(x_new)

    valutazioni_f = 2
    iterazioni = 0

    for k in range(maxit):
        x_next = x_new - f_new * (x_new - x_old) / (f_new - f_old)

        errore_stimato = abs(x_next - x_new)
        iterazioni += 1

        if errore_stimato <= atol + rtol * abs(x_next):
            return x_next, iterazioni, valutazioni_f, f(x_next), errore_stimato
        
        x_old = x_new
        f_old = f_new

        x_new = x_next
        f_new = f(x_new)
        valutazioni_f += 1
    
    return x_next, iterazioni, valutazioni_f, f(x_next), errore_stimato 


""" Confrontro il metodo di Newton e il medoto delle secanti """

atol = 1e-8
rtol = 1e-8

x0 = 0.5
x1 = 1.0

# Metodo di Newton

radice_newton, iter_newton, val_f_newton, val_df_newton, f_newton, err_newton = newton(f, df, x0, atol, rtol)

print("Metodo di Newton")
print("Radice: ", radice_newton)
print("Iterazioni: ", iter_newton)
print("Valutazion di f: ", val_f_newton)
print("Valutazioni di f': ", val_df_newton)
print("f(radice): ", f_newton)
print("Errore stimato: ", err_newton)

# Metodo della secanti

radice_secanti, iter_secanti, val_f_secanti, f_secanti, err_secanti = secanti(f, x0,x1, atol, rtol)

print("\nMetodo delle Secanti")
print("Radice: ", radice_secanti)
print("Iterazioni: ", iter_secanti)
print("Valutazion di f: ", val_f_secanti)
print("f(radice): ", f_secanti)
print("Errore stimato: ", err_secanti)

""" Confronto con scipy.optimize.newton() """

# Metodo di Newton con scipy

radice_scipy_newton, info_newton = optimize.newton( 
    f,
    x0,
    fprime = df,
    tol = atol,
    rtol = rtol,
    maxiter = 100,
    full_output = True
)

print("\nScipy - Newton")
print("Radice: ", radice_scipy_newton)
print(info_newton)

# Metodo delle secanti con scipy

radice_scipy_secanti, info_secanti = optimize.newton(
    f,
    x0,
    x1 = x1,
    tol = atol,
    rtol = rtol,
    maxiter = 100,
    full_output = True
)

print("\nScipy - Secanti")
print("Radice: ", radice_scipy_secanti)
print(info_secanti)


""" Tabella riassuntiva formattata """

righe = [
    [
        "Newton",
        f"{radice_newton:.12f}",
        iter_newton,
        val_f_newton,
        val_df_newton,
        f"{f_newton:.3e}",
        f"{err_newton:.3e}"
    ],
    [
        "Secanti",
        f"{radice_secanti:.12f}",
        iter_secanti,
        val_f_secanti,
        "-",
        f"{f_secanti:.3e}",
        f"{err_secanti:.3e}"
    ],
    [
        "Scipy Newton",
        f"{radice_scipy_newton:.12f}",
        info_newton.iterations,
        info_newton.function_calls,
        "-",
        f"{f(radice_scipy_newton):.3e}",
        "-"
    ],
    [
        "Scipy Secanti",
        f"{radice_scipy_secanti:.12f}",
        info_secanti.iterations,
        info_secanti.function_calls,
        "-",
        f"{f(radice_scipy_secanti):.3e}",
        "-"
    ]
]

colonne = [
    "Metodo",
    "Radice approssimata",
    "Iterazioni",
    "Valutazioni f",
    "Valutazioni f'",
    "f(xn)",
    "Errore stimato finale"
]

larghezze = [16, 22, 12, 16, 16, 14, 24]


def stampa_riga(valori):
    riga = "|"
    for valore, larghezza in zip(valori, larghezze):
        riga += f" {str(valore):^{larghezza}} |"
    print(riga)


separatore = "+"
for larghezza in larghezze:
    separatore += "-" * (larghezza + 2) + "+"

print("\nTabella riassuntiva")
print(separatore)
stampa_riga(colonne)
print(separatore)

for riga in righe:
    stampa_riga(riga)

print(separatore)


""" Grafico """


x_radice = 0.739085133215
y_radice = f(x_radice)

x = np.linspace(0, 1.2, 400)
y = f(x)

plt.figure(figsize=(7, 5))

plt.plot(x, y, label=r"$f(x)=\cos(x)-x$")
plt.axhline(0, color="black", linewidth=1)

plt.scatter(
    x_radice,
    y_radice,
    color="red",
    s=80,
    zorder=5,
    label=r"$x^* \approx 0.73908513$"
)

plt.axvline(
    x_radice,
    color="red",
    linestyle="--",
    linewidth=1
)

plt.grid(True)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Individuazione grafica della radice positiva")
plt.legend()
plt.show()
