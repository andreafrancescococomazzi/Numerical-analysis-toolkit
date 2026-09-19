"""Least-squares approximation: normal equations and QR decomposition."""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sla


""" Funzione """

def f(x):
    x = np.asarray(x, dtype=float)
    return np.exp(x)


""" Matrice di Vandermonde """

def vandermonde(x, grado):
    x = np.asarray(x, dtype=float)

    n = x.size
    A = np.zeros((n, grado + 1), dtype=float)

    # prima colonna

    A[:, 0] = 1.0

    # ogni colonna successiva si ottiene moltiplicando
    # la colonna precedente per il vettore dei nodi

    for j in range(1, grado + 1):
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
            raise np.linalg.LinAlgError(
                "La matrice è singolare"
            )

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
        raise np.linalg.LinAlgError(
            "La matrice è singolare"
        )

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



""" Stampa della matrice di Vandermonde """



def stampa_matrice(A):

    n_righe, n_colonne = A.shape

    # intestazioni delle colonne
    nomi_colonne = [
        f"x^{j}"
        for j in range(n_colonne)
    ]

    # intestazioni delle righe
    nomi_righe = [
        f"{i}"
        for i in range(n_righe)
    ]

    # valori della matrice in notazione scientifica
    valori_tabella = [
        [
            f"{A[i, j]:.3e}"
            for j in range(n_colonne)
        ]
        for i in range(n_righe)
    ]

    fig, ax = plt.subplots(figsize=(22, 8))
    ax.axis("off")

    tabella = ax.table(
        cellText=valori_tabella,
        rowLabels=nomi_righe,
        colLabels=nomi_colonne,
        cellLoc="center",
        rowLoc="center",
        loc="center"
    )

    tabella.auto_set_font_size(False)
    tabella.set_fontsize(8)
    tabella.scale(1.0, 1.4)

    plt.title(
        "Matrice di Vandermonde",
        pad=20
    )

    plt.tight_layout()

    # salvo la matrice come immagine
    plt.savefig(
        "matrice_vandermonde.png",
        dpi=250,
        bbox_inches="tight"
    )

    print("\nLa matrice di Vandermonde è stata salvata nei file:")
    print(" - matrice_vandermonde.png")
   


def main():

    np.set_printoptions(
        precision=10,
        suppress=True
    )

    # grado del polinomio

    grado = 14

    # punti assegnati

    x_nodi = np.linspace(
        0.0,
        1.0,
        18
    )

    y_nodi = f(x_nodi)

    # griglia di 200 punti

    x_valutazione = np.linspace(
        0.0,
        1.0,
        200
    )

    y_esatta = f(x_valutazione)

    # stampo i punti assegnati

    print("\nPunti assegnati")
    print("-" * 58)

    print(
        f"{'i':^5} "
        f"{'x_i':^20} "
        f"{'y_i = e^(x_i)':^28}"
    )

    print("-" * 58)

    for i, xi, yi in zip(
        range(x_nodi.size),
        x_nodi,
        y_nodi
    ):

        print(
            f"{i:^5d} "
            f"{xi:^20.10f} "
            f"{yi:^28.14e}"
        )


    """ Costruzione della matrice di Vandermonde """

    A = vandermonde(
        x_nodi,
        grado
    )

    print(
        "\nDimensione della matrice di Vandermonde:",
        A.shape
    )

    print("\nMatrice di Vandermonde")
    
    stampa_matrice(A)


    """ Numero di condizionamento della matrice A """

    condizionamento_A = np.linalg.cond(
        A,
        2
    )


    """ Costruzione del sistema delle equazioni normali """

    AT_A = A.T @ A

    AT_b = A.T @ y_nodi


    """ Numero di condizionamento della matrice A^T A """

    condizionamento_AT_A = np.linalg.cond(
        AT_A,
        2
    )


    """ Risoluzione mediante equazioni normali e LU con pivot totale """

    coefficienti_LU, L, U, P, Q = risultato_LU(
        AT_A,
        AT_b
    )


    """ Risoluzione mediante fattorizzazione QR """

    Q_QR, R_QR = sla.qr(
        A,
        mode="economic"
    )

    termine_noto_QR = Q_QR.T @ y_nodi

    coefficienti_QR = sostituzione_indietro(
        R_QR,
        termine_noto_QR
    )


    """ Valutazione dei due polinomi sui 200 punti """

    # I coefficienti sono memorizzati nell'ordine:
    #
    # c_0, c_1, ..., c_14.
    #
    # np.polyval richiede invece:
    #
    # c_14, c_13, ..., c_0.

    y_LU = np.polyval(
        coefficienti_LU[::-1],
        x_valutazione
    )

    y_QR = np.polyval(
        coefficienti_QR[::-1],
        x_valutazione
    )


    """ Calcolo dei residui relativi """

    residuo_relativo_LU = (
        np.linalg.norm(
            A @ coefficienti_LU - y_nodi,
            2
        )
        /
        np.linalg.norm(
            y_nodi,
            2
        )
    )

    residuo_relativo_QR = (
        np.linalg.norm(
            A @ coefficienti_QR - y_nodi,
            2
        )
        /
        np.linalg.norm(
            y_nodi,
            2
        )
    )


    """ Calcolo degli errori relativi sui 200 punti """

    # L'errore relativo viene calcolato confrontando
    # i valori dei polinomi con la funzione esatta e^x
    # sui 200 punti della griglia di valutazione.

    errore_relativo_LU = (
        np.linalg.norm(
            y_LU - y_esatta,
            2
        )
        /
        np.linalg.norm(
            y_esatta,
            2
        )
    )

    errore_relativo_QR = (
        np.linalg.norm(
            y_QR - y_esatta,
            2
        )
        /
        np.linalg.norm(
            y_esatta,
            2
        )
    )


    """ Differenza tra i coefficienti ottenuti """

    differenza_coefficienti = np.linalg.norm(
        coefficienti_LU - coefficienti_QR,
        2
    )

    differenza_relativa_coefficienti = (
        differenza_coefficienti
        /
        np.linalg.norm(
            coefficienti_QR,
            2
        )
    )


    """ Controlli delle fattorizzazioni """

    errore_fattorizzazione_LU = np.linalg.norm(
        P @ AT_A @ Q - L @ U,
        np.inf
    )

    errore_ortogonalita_QR = np.linalg.norm(
        Q_QR.T @ Q_QR
        - np.eye(grado + 1),
        np.inf
    )

    errore_fattorizzazione_QR = np.linalg.norm(
        A - Q_QR @ R_QR,
        np.inf
    )


    """ Errore massimo sui 200 punti """

    errore_max_LU = np.max(
        np.abs(
            y_LU - y_esatta
        )
    )

    errore_max_QR = np.max(
        np.abs(
            y_QR - y_esatta
        )
    )


    """ Stampo i coefficienti ottenuti """

    print(
        "\nCoefficienti del polinomio di grado 14"
    )

    print("-" * 67)

    print(
        f"{'j':^5} "
        f"{'Coefficienti LU':^28} "
        f"{'Coefficienti QR':^28}"
    )

    print("-" * 67)

    for j in range(grado + 1):

        print(
            f"{j:^5d} "
            f"{coefficienti_LU[j]:^28.14e} "
            f"{coefficienti_QR[j]:^28.14e}"
        )


    """ Stampo i risultati ottenuti """

    print("\nNumeri di condizionamento")
    print("-" * 62)

    print(
        f"cond_2(A)       = "
        f"{condizionamento_A:.14e}"
    )

    print(
        f"cond_2(A^T A)   = "
        f"{condizionamento_AT_A:.14e}"
    )

    print(
        f"cond_2(A)^2     = "
        f"{condizionamento_A**2:.14e}"
    )


    print("\nControllo delle fattorizzazioni")
    print("-" * 62)

    print(
        f"||P A^T A Q - L U||_inf = "
        f"{errore_fattorizzazione_LU:.14e}"
    )

    print(
        f"||A - Q R||_inf         = "
        f"{errore_fattorizzazione_QR:.14e}"
    )

    print(
        f"||Q^T Q - I||_inf       = "
        f"{errore_ortogonalita_QR:.14e}"
    )


    print("\nConfronto tra i due metodi")
    print("-" * 91)

    print(
        f"{'Metodo':<25s} "
        f"{'Residuo relativo':>20s} "
        f"{'Errore relativo':>20s} "
        f"{'Errore massimo':>20s}"
    )

    print("-" * 91)

    print(
        f"{'Equazioni normali - LU':<25s} "
        f"{residuo_relativo_LU:20.14e} "
        f"{errore_relativo_LU:20.14e} "
        f"{errore_max_LU:20.14e}"
    )

    print(
        f"{'Fattorizzazione QR':<25s} "
        f"{residuo_relativo_QR:20.14e} "
        f"{errore_relativo_QR:20.14e} "
        f"{errore_max_QR:20.14e}"
    )


    print("\nDifferenza tra i coefficienti")
    print("-" * 62)

    print(
        f"||c_LU - c_QR||_2 = "
        f"{differenza_coefficienti:.14e}"
    )

    print(
        f"||c_LU - c_QR||_2 / ||c_QR||_2 = "
        f"{differenza_relativa_coefficienti:.14e}"
    )


    """ Conclusione """

    print("\nConclusione")
    print("-" * 62)

    print(
        "La matrice di Vandermonde A è fortemente "
        "mal condizionata."
    )

    print(
        "La costruzione delle equazioni normali peggiora "
        "il condizionamento."
    )

    print(
        "In aritmetica esatta vale "
        "cond_2(A^T A) = cond_2(A)^2."
    )

    print(
        "Il valore numerico calcolato può non coincidere "
        "esattamente con il quadrato a causa degli errori "
        "di arrotondamento e ")
    print(
        "dell'elevato condizionamento."
    )

    print(
        "Il metodo QR evita la formazione esplicita "
        "della matrice A^T A."
    )

    print(
        "Per questo motivo la fattorizzazione QR "
        "risulta numericamente più stabile."
    )


    """ Grafico dei polinomi ottenuti """

    plt.figure(figsize=(10, 6))

    plt.plot(
        x_valutazione,
        y_esatta,
        linewidth=2.5,
        label="funzione esatta"
    )

    plt.plot(
        x_valutazione,
        y_LU,
        "--",
        linewidth=1.8,
        label="equazioni normali - LU"
    )

    plt.plot(
        x_valutazione,
        y_QR,
        "-.",
        linewidth=1.8,
        label="fattorizzazione QR"
    )

    plt.plot(
        x_nodi,
        y_nodi,
        "o",
        markersize=6,
        label="punti assegnati"
    )

    plt.xlabel("x")
    plt.ylabel("y")

    plt.title(
        "Polinomio di miglior approssimazione di grado 14"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
   


    """ Grafico degli errori assoluti """

    plt.figure(figsize=(10, 6))

    plt.semilogy(
        x_valutazione,
        np.abs(y_LU - y_esatta),
        linewidth=1.8,
        label="errore equazioni normali - LU"
    )

    plt.semilogy(
        x_valutazione,
        np.abs(y_QR - y_esatta),
        linewidth=1.8,
        label="errore fattorizzazione QR"
    )

    plt.xlabel("x")
    plt.ylabel("errore assoluto")

    plt.title(
        "Confronto degli errori assoluti"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
