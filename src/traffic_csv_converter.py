#!/usr/bin/env python

"""
    traffic_csv_converter.py
    by talshapira

    Questo modulo contiene funzioni per convertire i file CSV di traffico
    in dataset di istogrammi 2D, pronti per essere utilizzati in modelli di
    machine learning.
"""

#   ####################################################################    #
#   LIBRERIE

import os
import csv
import sessions_plotter as sp
import re
import numpy as np

#   ####################################################################    #
#   COSTANTI

INPUT       = "../dataset/classes_csvs/browsing/reg/CICNTTor_browsing.raw.csv"
INPUT_DIR   = "../dataset/classes_csvs/browsing/reg/"

TPS = 60
""" TimePerSession in secs. """

# DELTA_T = 60
DELTA_T = 15
""" Delta T between splitted sessions in secs. """

MIN_TPS = 50
""" Minimum TimePerSession in secs. """

MIN_DIM = 100000
""" Minimum number of packets in a session. """

#   ####################################################################    #
#   FUNZIONI

def export_class_dataset(dataset, class_dir):
    """ Salva su disco un array NumPy di istogrammi 2D per una singola classe
    di traffico, in un file .npy con nome che identifica la classe.

    Args:
        dataset (numpy.ndarray): Array di shape (N, 1, H, W) contenente
            gli istogrammi 2D delle sessioni di traffico della classe,
            dove:
            - N : il numero di sessioni.
            - H, W : le dimensioni dell'istogramma (tipicamente 1500x1500 in FlowPic).
        class_dir (str): Percorso della cartella della classe.
    """

    print("Start export dataset")

    # Si estrae il nome della classe dal percorso della cartella utilizzando
    # una regex che prende le ultime due parole, cioè il nome della classe
    # e il tipo di traffico.
    # Esempio: "browsing_reg" da "../dataset/classes_csvs/browsing/reg/"
    type_name = re.findall(r"[\w']+", class_dir)[-2:]

    # Salva l'array NumPy nella stessa cartella della classe, con il nome costruito sopra.
    np.save(class_dir + "/" + "_".join(type_name) + "_TPS" + str(TPS) + "_DELTA_T" + str(DELTA_T), dataset)

    print(dataset.shape)

    # end

def traffic_csv_converter(file_path):
    """ Converte un file CSV di traffico di rete in un dataset
    adatto all'architettura FlowPic, da utilizzare come input alla CNN.

    ## Formato atteso del file CSV \n
    Colonna n.0 : label della sessione (es. "AIM_Chat"). \n
    Colonne n.1..n.5 : 5-tupla che identifica la sessione. \n
    Colonna n.6 : timestamp di inizio della sessione. \n
    Colonna n.7 : numero totale di pacchetti nella sessione. \n
    Colonne n.8 ... n.(8+length-1) : timestamp di arrivo di ciascun pacchetto. \n
    Colonne n.(9+length) ... (fine) : dimensione in byte di ciascun pacchetto. \n

    Args:
        file_path (str): Percorso del file CSV da convertire.

    Returns:
        numpy.ndarray: Array di shape (N, 1, 1500, 1500) contenente gli
        istogrammi 2D delle sessioni di traffico, dove N è il numero di
        finestre temporali valide.
        Restituisce un array vuoto di shape (0,) se nessuna finestra valida
        viene trovata nel file.
    """

    #   ####################################################################    #
    #   INIZIALIZZAZIONE

    print("Running on " + file_path)

    dataset = []
    counter = 0

    #   ####################################################################    #
    #   LETTURA DEL FILE CSV

    with open(file_path, 'r') as csv_file:
        
        reader = csv.reader(csv_file)

        # Ogni sessione è un flusso unidirezionale identificato da una 5-tupla:
        # IP sorgente, porta sorgente, IP destinazione, porta destinazione, protocollo.
        # I restanti campi contengono metadati aggiuntivi sulla sessione,
        # come il nome dell'applicazione, il timestamp di inizio e il numero di pacchetti totali.
        for (i, row) in enumerate(reader):

            #   ############################################################    #
            #   ESTRAZIONE DELLA 5-TUPLA E DEI METADATI

            session_tuple_key = {
                # Chiave di riferimento per il debug, che non
                # influisce direttamente sulla costruzione dell'istogramma.
                "label": row[0],

                # La 5-tupla che identifica univocamente
                # la sessione di traffico.
                "source_ip": row[1],
                "source_port": int(row[2]),
                "destination_ip": row[3],
                "destination_port": int(row[4]),
                "protocol": row[5],

                # Timestamp di inizio della sessione, espresso in
                # secondi assoluti dall'inizio della cattura.
                "start_time": float(row[6]),

                # Numero totale di pacchetti nella sessione,
                # che determina la lunghezza dei campi successivi contenenti
                # i timestamp e le dimensioni dei pacchetti.
                "length": int(row[7])
            }

            if i == 0:
                print(session_tuple_key)

            #   ############################################################    #
            #   ESTRAZIONE DEI TIMESTAMP E DELLE DIMENSIONI DEI PACCHETTI

            # Array dei timestamp di arrivo di ciascun pacchetto, espressi in secondi assoluti dall'inizio della cattura.
            ts = np.array(row[8:8+session_tuple_key["length"]], dtype=float)

            # Array delle dimensioni in byte di ciascun pacchetto, nell'ordine corrispondente ai timestamp.
            sizes = np.array(row[9+session_tuple_key["length"]:], dtype=int)

            """
            NOTA BENE
            'ts' e 'sizes' hanno sempre la stessa lunghezza 'length',
            e i loro elementi sono allineati per indice:
            ciò significa che ts[i] e sizes[i] descrivono lo stesso pacchetto.

            In particolare, tra l'ultimo timestamp e il primo size c'è una
            cella vuota (rappresentata da una doppia virgola ',,').
            Lo slice 'row[9+session_tuple_key["length"]:' salta correttamente questa cella vuota.
            """

            """
            ESEMPIO
            La sessione AIM_Chat contiene 260 pacchetti catturati e
            determina quanti elementi occupano le colonne successive:

            -   ts : occupa le colonne 8..267  (dato da 8 + 260 - 1)

                I valori sono già relativi al primo pacchetto (infatti, il primo è 0.0).
                La sessione copre circa 723 secondi (~12 minuti) di traffico AIM_Chat.

                ts[0]   =   0.0         > primo pacchetto (riferimento)
                ts[1]   =   0.264737    > secondo pacchetto, arrivato 0.26s dopo
                ...
                ts[9]   =  31.936928    > decimo pacchetto, arrivato circa 32s dopo
                ts[259] = 723.553638    > ultimo pacchetto, arrivato circa 723s dopo

            -   sizes : occupa le colonne 269..528 (9 + 260 = 269, fino a fine riga)

                sizes[0]    = 40    > pacchetto ACK/keepalive (header TCP puro)
                sizes[1]    = 285   > secondo pacchetto, contenente dati (payload) dell'applicazione AIM_Chat
                sizes[2]    = 40    > terzo pacchetto, ACK di conferma del secondo
                sizes[3]    = 538
                ...

            """

            #   ############################################################    #
            #   FILTRO DI QUALITÀ DELLA SESSIONE

            # Scarta le sessioni con 10 o meno pacchetti (oltre ai primi 8 campi di metadati),
            # troppo corte per costruire un istogramma 2D significativo.
            if not(session_tuple_key["length"] > 10):
                continue

            # print("Filtro SESSION : OK")

            #   ############################################################    #
            #   ANALISI DELLA SESSIONE

            # Calcola quante finestre temporali scorrevoli è possibile
            # estrarre dalla sessione, in base ai valori costanti e la sua durata totale.
            num_windows = int(ts[-1]/DELTA_T - TPS/DELTA_T) + 1

            # Itera su ciascuna finestra temporale scorrevole.
            # Ogni finestra 't' inizia a 't * DELTA_T' secondi e dura 'TPS' secondi.
            for t in range(num_windows):

                #   ########################################################    #
                #   Applicazione di una maschera booleana.
                #   Seleziona solo i pacchetti il cui timestamp cade all'interno della finestra corrente.

                mask = ((ts >= t * DELTA_T) & (ts <= (t * DELTA_T + TPS)))
                ts_mask = ts[mask]
                sizes_mask = sizes[mask]

                #   ########################################################    #
                #   Filtro di qualità della finestra.

                # La finestra deve contenere più di 10 pacchetti,
                # altrimenti l'istogramma è troppo sparso per essere utile.
                if not(len(ts_mask) > 10):
                    continue

                # print("Filtro WIN_1 : OK")

                # La finestra deve coprire almeno 'MIN_TPS' secondi di traffico effettivo,
                # evitando finestre in cui i pacchetti sono tutti concentrati in pochi secondi.
                if not(ts_mask[-1] - ts_mask[0] > MIN_TPS):
                    continue

                # print("Filtro WIN_2 : OK")

                # La finestra deve contenere almeno 'MIN_DIM' pacchetti,
                # altrimenti l'istogramma è troppo vuoto per essere utile.
                if not(np.sum(sizes_mask) > MIN_DIM):
                    continue

                # print("Filtro WIN_3 : OK")

                #   ########################################################    #
                #   Costruzione dell'istogramma 2D.
                
                h = sp.session_2d_histogram(ts_mask, sizes_mask, plot=True)

                #   ########################################################    #
                #   Aggiunta dell'istogramma al dataset.

                dataset.append([h])

                #   ########################################################    #
                #   Stampa di debug.

                counter += 1

                #   Stampa un messaggio ogni 100 sessioni processate per monitorare l'avanzamento.
                if counter % 100 == 0:
                    print(counter)
                
            # end for t

        # end for (i, row)
    
    #   ####################################################################    #
    #   RITORNO DEL DATASET

    return np.asarray(dataset)

    # end

def traffic_class_converter(dir_path):
    """ Converte tutti i file CSV di una singola cartella classe/tipo in un
    unico dataset NumPy di FlowPic.

    Args:
        dir_path (str): percorso della cartella contenente i file CSV.

    Returns:
        numpy.ndarray: array di shape (N, 1, 1500, 1500) contenente tutti i FlowPic
        estratti da tutti i file CSV della cartella, dove N è la somma delle finestre
        temporali valide trovate in ciascun file.
    
    Raises:
        ValueError: Se la cartella non contiene alcun file "csv".
        StopIteration: Se dir_path non esiste o non è accessibile.
    """

    #   ####################################################################    #
    #   INIZIALIZZAZIONE

    dataset_tuple = ()

    all_paths = [
        # Costruisce il percorso completo.
        os.path.join(dir_path, fn)

        # Restituisce una tripla:
        # (cartella_corrente, [sottocartelle], [file]).
        # Prende solo il terzo elemento, cioè la lista dei nomi di file.
        for fn in next(os.walk(dir_path))[2]

        # Estrae l'estensione del file e prende solo quelli con estensione "csv".
        if (".csv" in os.path.splitext(fn)[-1])
    ]

    #   ####################################################################    #
    #   LETTURA DEI FILE CSV DI CLASSE

    for file_path in all_paths:

        # La virgola serve creare una tupla con un solo elemento,
        # invece di interpretare l'operazione += come somma di array.
        dataset_tuple += (traffic_csv_converter(file_path),)

        # end for file_path
    
    #   ####################################################################    #
    #   RITORNO DEL DATASET

    # Concatena tutti gli array della tupla lungo l'asse 0 (il numero di campioni),
    # producendo un unico array con tutti i FlowPic della classe.
    return np.concatenate(dataset_tuple, axis=0)

    # end

#   ####################################################################    #
#   MAIN

if __name__ == '__main__':

    # Per l'intera classe
    # print("working on " + INPUT_DIR)
    # dataset = traffic_class_converter(INPUT_DIR)
    # print(dataset.shape)
    # export_class_dataset(dataset, INPUT_DIR)

    # Per CSV specifico
    print("working on " + INPUT)
    dataset = traffic_csv_converter(INPUT)
    print(dataset.shape)
    export_class_dataset(dataset, INPUT_DIR)

    # end