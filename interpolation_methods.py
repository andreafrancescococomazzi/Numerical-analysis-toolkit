
"""Interpolation methods: linear spline, cubic spline, and global polynomial."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

""" Funzione """

def f(x):
    x = np.asarray(x, dtype=float)  
    return np.log(2.0 + x) / (1.0 + x**2)


""" Matrice di Vandermonde """

def vandermonde(x):
    x = np.asarray(x, dtype=float)
    n = x.size
    A = np.zeros((n, n), dtype=float)

    # prima colonna
    
    A[:, 0] = 1.0

    # ogni colonna successiva si ottiene moltiplicando
    # la colonna precedente per il vettore dei nodi
    
    for j in range(1, n):
        A[:, j] = A[:, j - 1] * x

    return A


""" Fattorizzazione LU con pivot totale """

def lu_tot(A):
    A = np.asarray(A, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("La matrice A deve essere quadrata")
    
    n = A.shape[0]

    U = A.copy()
    L = np.eye(n)
    P = np.eye(n)
    Q = np.eye(n)

    for k in range(n - 1):

        sotto_matrice = np.abs(U[k:, k:])

        indice_locale = np.unravel_index(
            np.argmax(sotto_matrice),
            sotto_matrice.shape
        )

        riga_pivot = k + indice_locale[0]
        colonna_pivot = k + indice_locale[1]

        pivot_max = abs(U[riga_pivot, colonna_pivot])

        if pivot_max == 0.0:
            raise np.linalg.LinAlgError("La matrice è singolare")

        # Scambio le righe

        if riga_pivot != k:

            U[[k, riga_pivot], :] = U[[riga_pivot, k], :]
            P[[k, riga_pivot], :] = P[[riga_pivot, k], :]

            if k > 0:
                L[[k, riga_pivot], :k] = L[[riga_pivot, k], :k]
        
        # Scambio le colonne

        if colonna_pivot != k:

            U[:, [k, colonna_pivot]] = U[:, [colonna_pivot, k]]

            Q[:, [k, colonna_pivot]] = Q[:, [colonna_pivot, k]]

        # Eliminazione di Gauss

        for i in range(k + 1, n):

            L[i, k] = U[i, k] / U[k, k]

            U[i, k:] = U[i, k:] - L[i, k] * U[k, k:]

            U[i, k] = 0.0
        
    if U[-1, -1] == 0.0:
        raise np.linalg.LinAlgError("La matrice è singolare")
        
    return L, U, P, Q


""" Sostituzione in avanti """

def sostituzione_avanti(L, b):
    L = np.asarray(L, dtype=float)
    b = np.asarray(b, dtype=float)

    n = b.size
    x = np.zeros(n, dtype=float)
    
    for i in range(n):

        somma = np.dot(L[i, :i], x[:i])
        
        x[i] = (b[i] - somma) / L[i, i]
        
    return x


""" Sostituzione all'indietro """

def sostituzione_indietro(U, b):

    U = np.asarray(U, dtype=float)
    b = np.asarray(b, dtype=float)

    n = b.size
    x = np.zeros(n, dtype=float)

    for i in range(n - 1, -1, -1):

        if U[i, i] == 0.0:
            raise np.linalg.LinAlgError(
                "Elemento diagonale nullo nella matrice U"
            )

        somma = np.dot(U[i, i + 1:], x[i + 1:])

        x[i] = (b[i] - somma) / U[i, i]

    return x


""" Risoluzione della fattorizzazione LU con pivot totale """

def risultato_LU(A, b):

    L, U, P, Q = lu_tot(A)
   
    termine_noto_permutato = P @ b
   
    z = sostituzione_avanti(L, termine_noto_permutato)
   
    y = sostituzione_indietro(U, z)
   
    coefficienti = Q @ y
    
    return coefficienti, L, U, P, Q


def main():

    np.set_printoptions(precision=10, suppress=True)

    # intervallo di interpolazione

    a = -0.5
    b = 3.0

    # nodi assegnati

    indici = np.arange(8)

    x_nodi = -0.5 + indici * (3.5 / 7.0)
    y_nodi = f(x_nodi)

    # griglia di 200 punti

    x_valutazione = np.linspace(a, b, 200)
    y_esatta = f(x_valutazione)

    # stampo i nodi

    print("\nNodi di Interpolazione")
    print("_" * 51)

    print(
        f"{'i':^5}"
        f"{'x_i':^20}"
        f"{'f(x_i)':^28}"
    )

    for i, xi, yi in zip(indici, x_nodi, y_nodi):

        print(
            f"{i:^5d}"
            f"{xi:^20.10f}"
            f"{yi:^28.14e}"
        )
    print("_" * 51)

    """ Spline Lineare e Spline Cubica """

    spline_lineare = interp1d(
        x_nodi,
        y_nodi,
        kind="linear"
    )

    spline_cubica = interp1d(
        x_nodi,
        y_nodi,
        kind="cubic"
    )

    # valutazione delle spline sui 200 punti

    y_lineare = spline_lineare(x_valutazione)
    y_cubica = spline_cubica(x_valutazione)

    # controllo dell'interpolazione nei nodi

    error_nodale_lineare = np.linalg.norm(
        spline_lineare(x_nodi) - y_nodi,
        np.inf
    )

    error_nodale_cubica = np.linalg.norm(
        spline_cubica(x_nodi) - y_nodi,
        np.inf
    )

    """ Polinomio interpolante globale """

    A = vandermonde(x_nodi)

    coefficienti, L, U, P, Q = risultato_LU(
        A,
        y_nodi
    )

    # np.polyval richiede i coefficienti ordinati
    # dalla potenza più alta alla potenza più bassa

    y_polinomio = np.polyval(
        coefficienti[::-1],
        x_valutazione
    )

    y_polinomio_nodi = np.polyval(
        coefficienti[::-1],
        x_nodi
    )

    """ Controllo degli errori """
    
    error_fattorizzazione = np.linalg.norm(
        P @ A @ Q - L @ U,
        np.inf
    )
    
    residuo_relativo = (
        np.linalg.norm(
            A @ coefficienti - y_nodi,
            np.inf
        )
        /
        np.linalg.norm(
            y_nodi,
            np.inf
        )
    )
    
    error_nodale_polinomio = np.linalg.norm(
        y_polinomio_nodi - y_nodi,
        np.inf
    )
    
    condizionamento_A = np.linalg.cond(A, 2)

    """ Controllo degli errori sui 200 punti """

    error_lineare = np.abs(
        y_esatta - y_lineare
    )

    error_cubica = np.abs(
        y_esatta - y_cubica
    )

    error_polinomio = np.abs(
        y_esatta - y_polinomio
    )

    error_max_lineare = np.max(error_lineare)

    error_max_cubica = np.max(error_cubica)

    error_max_polinomio = np.max(error_polinomio)

    indice_lineare = np.argmax(error_lineare)

    indice_cubica = np.argmax(error_cubica)
    
    indice_polinomio = np.argmax(error_polinomio)

    """ Stampo i risultati ottenuti """

    print("\nMatrice di Vandermonde")
    print("_" * 98)

    # Intestazione delle colonne
    print(
        "  "
        + "  ".join(
            f"{('x^' + str(j)):>10}"
            for j in range(A.shape[1])
        )
    )

    print("_" * 98)

    # Stampa della matrice, una riga alla volta
    for riga in A:
        print(
            "[ "
            + "  ".join(
                f"{valore:10.5f}"
                for valore in riga
            )
            + " ]"
        )

    print("_" * 98)

    print("\nCoefficienti del polinomio")
    print("_" * 37)

    print(
        "p_7(x) = c_0 + c_1*x + ... + c_7*x^7"
    )

    for j, coefficiente in enumerate(coefficienti):

        print(
            f"c_{j} = "
            f"{coefficiente: .14e}"
        )
    print("_" * 37)

    print("\nControllo degli errori")
    print("_" * 50)
    print(
        f"||P A Q - L U||_inf       = "
        f"{error_fattorizzazione:.14e}"
    )

    print(
        f"Residuo relativo          = "
        f"{residuo_relativo:.14e}"
    )

    print(
        f"Errore nodale lineare     = "
        f"{error_nodale_lineare:.14e}"
    )

    print(
        f"Errore nodale cubica      = "
        f"{error_nodale_cubica:.14e}"
    )

    print(
        f"Errore nodale polinomio   = "
        f"{error_nodale_polinomio:.14e}"
    )

    print(
        f"cond_2(A)                 = "
        f"{condizionamento_A:.14e}"
    )
    print("_" * 50)

    print("\nControllo degli errori sui 200 punti")
    print("_" * 70)

    print(
        f"{'Metodo':<24s} "
        f"{'Errore massimo':>22s} "
        f"{'x del massimo':>22s}"
    )

    print(
        f"{'Spline lineare':<24s} "
        f"{error_max_lineare:22.14e} "
        f"{x_valutazione[indice_lineare]:22.14e}"
    )

    print(
        f"{'Spline cubica':<24s} "
        f"{error_max_cubica:22.14e} "
        f"{x_valutazione[indice_cubica]:22.14e}"
    )

    print(
        f"{'Polinomio globale':<24s} "
        f"{error_max_polinomio:22.14e} "
        f"{x_valutazione[indice_polinomio]:22.14e}"
    )
    print("_" * 70)
    

    """ Individuo il metodo più accurato """

    nome_metodo = np.array([
        "spline lineare",
        "spline cubica",
        "polinomio globale"
    ])

    error_max = np.array([
        error_max_lineare,
        error_max_cubica,
        error_max_polinomio
    ])

    indice_metedo_migliore = np.argmin(error_max)

    metodo_accurato = nome_metodo[indice_metedo_migliore]

    print("\nConclusione")
    print("_" * 105)

    print(
        "Il metodo più accurato sui 200 punti è:",
        metodo_accurato
    )

    print(
        "La spline cubica è il metodo più affidabile "
        "dal punto di vista della stabilità perché "
        "è un metodo locale"
    )

    print(
        "Il polinomio globale richiede la risoluzione "
        "della matrice di Vandermonde nella base monomiale"
    )

    print(
        "Il pivot totale stabilizza l'eliminazione di Gauss, "
        "ma non elimina il mal condizionamento della matrice"
    )
    print("_" * 105)

    """ Grafico """

    plt.figure(figsize=(10, 6))

    plt.plot(
        x_valutazione,
        y_esatta,
        linewidth=2.5,
        label="funzione esatta"
    )

    plt.plot(
        x_valutazione,
        y_lineare,
        "--",
        linewidth=1.8,
        label="spline lineare"
    )

    plt.plot(
        x_valutazione,
        y_cubica,
        "-.",
        linewidth=1.8,
        label="spline cubica"
    )

    plt.plot(
        x_valutazione,
        y_polinomio,
        ":",
        linewidth=2.2,
        label="polinomio globale"
    )

    plt.plot(
        x_nodi,
        y_nodi,
        "o",
        markersize=7,
        label="nodi"
    )

    plt.xlabel("x")
    plt.ylabel("y")

    plt.title(
        "Confronto tra funzione e interpolanti"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__": 
    main()

