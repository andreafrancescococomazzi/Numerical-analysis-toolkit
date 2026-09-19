
"""Spectral decomposition of a symmetric matrix."""

import numpy as np


def matrice(A, formato="decimale"):
    A = np.asarray(A, dtype=float)

    for riga in A:
        if formato == "scientifico":
            valori = "  ".join(f"{valore:14.6e}" for valore in riga)

        else:
            valori = "  ".join(f"{valore:14.8f}" for valore in riga)
        
        print("[ " + valori + " ]")

def main():
    np.set_printoptions( precision=10, suppress=True)

    B = np.array([
        [6.0, -2.0, 0.0, 0.0],
        [-2.0, 5.0, -1.0, 0.0],
        [0.0, -1.0, 4.0, -1.0],
        [0.0, 0.0, -1.0, 3.0]
    ])

    print("\nMatrice B")
    print("_"*70)

    matrice(B)

    print("_" * 70)


    """ Calcolo degli autovalori e degli autovettori """

    # La matrice B è reale e simmetrice.
    # Per questo motivo utilizzo numpy.linalg.eigh.

    autovalori, autovettori = np.linalg.eigh(B)

    """ Oridinamento degli autovalori in ordine crescente """

    indici = np.argsort(autovalori)
    autovalori_ordinati = autovalori[indici]
    autovettori_ordinati = autovettori[:, indici]

    """ Costruisco le matrici Q e D"""

    # Le colonne della matrice Q sono gli autovettori associati agli autovalori ordinati.

    Q = autovettori_ordinati

    # La matrice D contiene gli autovalori ordinati sulla diagonale principale

    D = np.diag(autovalori_ordinati)

    
    """ Ortogonalità degli autovettori """

    # Se gli autovettori sono ortonormali allora otteniamo:
    # Q^T Q = I

    matrice_ortogonale = Q.T @ Q
    I = np.eye(B.shape[0])

    errore_ortogonalità = np.linalg.norm(matrice_ortogonale - I, np.inf)

    
    """ Verifico le coppie autovalore-autovettore """ 

    # Utilizzo la seguente formula per calocolare le coppie:  B * q_j = λ_j * q_j
    # Utilizzo la seguente formula per calcolare la norma del residuo: ||B * q_j - λ_j * q_j||2

    residuo_autocoppie = np.zeros(autovalori_ordinati.size)
    for j in range(autovalori_ordinati.size):
        residuo_autocoppie[j] = np.linalg.norm(B @ Q[:, j] - autovalori_ordinati[j] * Q[:, j], 2)

    """Verifico B = Q D Q^T """

    B_ricostruita = Q @ D @ Q.T
    errore_decomposizione = B - B_ricostruita
    norma_infinito_errore = np.linalg.norm(errore_decomposizione, np.inf)

    """ Stampo gli autovalori ordinati """

    print("\nAutovalori ordinati in ordine crescente")
    print("_" * 30)

    print(
        f"{'j':^5}"
        f"{'λ_j':^25}"
    )

    for j, autovalore in enumerate(autovalori_ordinati):
        print(
            f"{j + 1:^5d}"
            f"{autovalore:^25.14e}"
        )

    print("_" * 30)

    """ Stampo gli autovettori associati """

    print("\nMatrice Q degli autovettori")
    print("_" * 70)
    matrice(Q)
    print("_" * 70)

    """ Stampo la matrice diagonale D """

    print("\nMatrice diagonale")
    print("_" * 70)
    matrice(D)
    print("_" * 70)

    """ Stampo il controllo di ortogonalità """

    print("\nVerifica dell'ortogonalità: Q^T Q")
    print("_" * 70)
    matrice(matrice_ortogonale)
    print("_" * 70)
    print("\n||Q^T Q - T||_inf = " 
          f"{errore_ortogonalità:.14e}"
          )
    
    """ Stampo i risultati delle coppie autovalore-autovettore """
    print("\nResidui delle coppie autovalore-autovettore")
    print("_" * 45)

    print(f"{'j':^5}" f"{'||B q_j - λ_j * q_j||_2':^40}")

    for j, residuo in enumerate(residuo_autocoppie):
        print(
            f"{j + 1:^5d}"
            f"{residuo:^40.14e}" 
            )
    print("_" * 45)

    """ Stampo la matrice ricostruita """

    print("\nMatrice ricostruita Q D Q^T ")
    print("_" * 68)
    matrice(B_ricostruita)
    print("_" * 68)

    """ Stampo l'errore della decomposizione """
    
    print("\nMatrice errore B - Q D D^T")
    print("_" * 78)
    matrice(errore_decomposizione, formato="scientifico")
    print("_" * 78)

    print("\n||B q_j - λ_j * q_j||_inf = "
          f"{norma_infinito_errore:.14e}"
        )
    

    """ Conclusione """

    print("\nConclusione")
    print("_" * 105)

    print("Gli autovalori sono reali e gli autovettori associati risultano ortonormali.")
    print("La matrice Q è quindi ortogonale e soddisfa numericamente le relazione Q^T Q = I.")
    print("||B - Q D Q^T||_inf è dell'ordine degli errori di arrotondamento.")
    print("La decomposizione spettrale B = Q D Q^T risulta quindi verificata numericamente.")

    print("_" * 105)


if __name__ == "__main__":
    main()
