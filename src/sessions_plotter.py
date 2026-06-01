#!/usr/bin/env python

"""
    sessions_plotter.py
    by talshapira

    ...
"""

#   ####################################################################    #
#   LIBRERIE

import matplotlib.pyplot as plt
import numpy as np

#   ####################################################################    #
#   COSTANTI

MTU = 1500
""" Maximum Transmission Unit di Ethernet. """

BIN_SIZE = 20
""" Dimensione in byte di ogni bin sull'asse Y dell'istogramma 2D. """

#   ####################################################################    #
#   FUNZIONI

def session_2d_histogram(ts, sizes, plot=False, tps=None):
    """ È la funzione chiave che costruisce un FlowPic, cioè un istogramma
    2D 1500x1500 che rappresenta la distribuzione spazio-temporale dei pacchetti
    di una finestra di traffico di rete.

    L'asse X rappresenta il tempo normalizzato nell'intervallo [0, 1500].
    L'asse Y rappresenta la dimensione dei pacchetti in byte, da 0 a 1500 (MTU).

    Args:
        ts (numpy.ndarray): Array 1D dei timestamp di arrivo dei pacchetti,
            espressi in secondi relativi all'inizio della finestra. Shape: (N,).
        sizes (numpy.ndarray): Array 1D delle dimensioni in byte dei pacchetti,
            allineato per indice con ts: cioè, sizes[i] è la dimensione del
            pacchetto ts[i]. Shape: (N,).
        plot (bool, optional): Se True, visualizza l'istogramma con matplotlib
            usando una colormap binaria (cioè, bianco se 0 pacchetti, nero se 1+
            pacchetti). Defaults to False.
        tps (float, optional): Durata fissa in secondi da usare come riferimento
            per la normalizzazione dell'asse X. Defaults to None.

    Returns:
        numpy.ndarray: Matrice 2D di shape (1500, 1500) e dtype uint16,
            dove H[y, x] è il numero di pacchetti con dimensione y byte arrivati
            al tempo normalizzato x.
    """

    #   ####################################################################    #
    #   INIZIALIZZAZIONE

    # Costruisce una griglia di bins 1500x1500 per l'istogramma 2D,
    # con 1 byte per bin sull'asse Y e 1 pixel per bin sull'asse X.
    b = (range(0, MTU + 1, BIN_SIZE), range(0, MTU + 1, BIN_SIZE))

    #   ####################################################################    #
    #   CONTROLLO DI VALIDITÀ DEGLI ARGOMENTI

    # Determina la durata temporale di riferimento su cui normalizzare l'asse X.
    if tps is None:
        max_delta_time = ts[-1] - ts[0]
    else:
        max_delta_time = tps

    #   ####################################################################    #
    #   NORMALIZZAZIONE DEI TIMESTAMP in 3 passi
    #   L'intera durata della finestra viene sempre mappata su 1500 pixel,
    #   indipendentemente da quanto dura in secondi.

    # ts_norm = map(int, ((np.array(ts) - ts[0]) / max_delta_time) * MTU)

    # 1. Rende i timestamp relativi all'inizio della finestra.
    ts_norm = np.array(ts) - ts[0]

    # 2. Scala nell'intervallo [0.0, 1.0].
    ts_norm = ts_norm / max_delta_time

    # 3. Scala nell'intervallo [0, 1500].
    ts_norm = ts_norm * MTU

    #   ####################################################################    #
    #   COSTRUZIONE DEL FLOWPIC

    H, xedges, yedges = np.histogram2d(
        sizes,
        ts_norm,
        bins=b
    )

    #   ####################################################################    #
    #   VISUALIZZAZIONE OPZIONALE DELL'ISTOGRAMMA

    if plot:
        plt.pcolormesh(
            xedges, yedges,
            (H > 0).astype(np.uint8),
            cmap='binary_r',
            vmin=0, vmax=1
        )

        # plt.colorbar()
        plt.xlim(0, MTU)
        plt.ylim(0, MTU)
        plt.set_cmap('binary')
        plt.show()

    #   ####################################################################    #
    #   RITORNO DELLA MATRICE 2D

    # Converte i conteggi in interi a 16 bit senza segno
    # per occupare meno spazio in memoria.
    return H.astype(np.uint16)

    # end
