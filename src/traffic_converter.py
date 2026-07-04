"""

    traffic_converter.py
    by Mario Gabriele Carofano

    Questo modulo raccoglie le funzioni necessarie per convertire i dati di
    traffico di rete (provenienti dai dataset Mirage in formato "pickle")
    in un dataset di istogrammi 2D (FlowPics) in formato numpy, pronto per
    essere utilizzato per l'addestramento di modelli ML (e QML) per la
    classificazione di attività o applicazioni nel traffico di rete mobile.

"""

#   ########################################################################    #
#   LIBRERIE

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from constants import (
    FEATURES_LIST,
    MTU, BIN_SIZE
)

#   ########################################################################    #
#   FUNZIONI di CONVERSIONE del traffico in FLOWPIC

def mirage_pickle_converter(
    file_path: str,
    filters: dict = None,
    debug : bool = False,
    flows_to_inspect: list[int] = None
):
    
    """ Converte un file .pickle di traffico di rete in un dataset
    adatto all'architettura FlowPic, da utilizzare come input alla CNN.

    ## Formato atteso del file .pickle
    -   PRIMA SCANSIONE
        Deve contenere una lista di array NumPy con la seguente struttura:
        - p         : una lista di np.ndarray di lunghezza N.
        - p[i]      : un np.ndarray di shape (T, 4), di tipo float64, dove T è il numero di pacchetti per flusso.
        - p[i][j]   : il j-esimo pacchetto dell'i-esimo flusso.
    -   SECONDA SCANSIONE
        Deve contenere un array NumPy di shape (N, ) di etichette dei flussi.
        Ogni etichetta rappresenta il nome dell'applicazione o dell'attività associata al flusso corrispondente.

    ## Features per pacchetto
    - DIR                       : Direzione del pacchetto (0.0 = upstream, 1.0 = downstream).
    - PL (Packet Length)        : Dimensione del payload IP in byte
    - WIN (TCP Window Size)     : Dimensione finestra TCP (0.0 per UDP).
    - IAT (Inter-Arrival Time)  : Tempo tra pacchetti consecutivi in secondi.

    Args:
        file_path (str): Percorso del file PICKLE da convertire.
        filters (dict, optional): Dizionario contenente i filtri da applicare ai flussi. Defaults to None. Può contenere le seguenti chiavi: 'min_tps', 'min_packets', 'min_dim'.
        debug (bool, optional): Se True, stampa informazioni di debug e il plot dell'istogramma 2D durante la conversione. Defaults to False.
        flows_to_inspect (list, optional): Lista di indici dei flussi da selezionare per il debug. Defaults to None.

    Returns:
        numpy.ndarray: Array di shape (N, 1, 1500, 1500) contenente gli
        istogrammi 2D delle sessioni di traffico, dove N è il numero di
        finestre temporali valide.
        Restituisce un array vuoto di shape (0,) se nessuna finestra valida
        viene trovata nel file.
    """

    #   ####################################################################    #
    #   VALIDITÀ DEI PARAMETRI

    if flows_to_inspect is not None and not isinstance(flows_to_inspect, list):
        raise ValueError("Il parametro 'flows_to_inspect' deve essere una lista di indici.")
    
    if filters and not isinstance(filters, dict):
        raise ValueError("Il parametro 'filters' deve essere un dizionario.")
    
    debug = debug or (flows_to_inspect is not None)

    #   ####################################################################    #
    #   INIZIALIZZAZIONE

    if debug:
        print(f"[DEBUG] Inizio conversione del file: {file_path}")

    dataset = []
    counter = 0

    min_packets = None
    min_dim = None
    min_tps = None
    
    if filters:
        # Numero minimo di pacchetti che una finestra deve contenere per essere considerata valida.
        min_packets = filters.get('min_packets', None)

        # Numero minimo di byte che una finestra deve contenere per essere considerata valida.
        min_dim = filters.get('min_dim', None)

        # Durata minima in secondi che una finestra deve coprire per essere considerata valida.
        min_tps = filters.get('min_tps', None)

    if debug:
        print(f"[DEBUG] Parametri di filtro\n")
        print(f"MIN_PACKETS: {min_packets}" if min_packets is not None else "")
        print(f"MIN_DIM: {min_dim}" if min_dim is not None else "")
        print(f"MIN_TPS: {min_tps}" if min_tps is not None else "")
        print("\n")

    #   ####################################################################    #
    #   LETTURA DEL FILE PICKLE

    with open(file_path, 'rb') as file:

        # Il file pickle di un dataset Mirage contiene due oggetti serializzati:
        # 1. Una lista di array NumPy, dove ogni array rappresenta un flusso di traffico.
        # 2. Un array NumPy di etichette (nomi delle applicazioni o attività) corrispondenti ai flussi.

        x_raw = pickle.load(file)
        y_raw = np.array(pickle.load(file))

        #   ################################################################    #
        #   Recupero dei biflussi selezionati.

        if flows_to_inspect is not None:
            iterable = (
                (id, x_raw[id], y_raw[id])
                for id in flows_to_inspect
            )
            total = len(flows_to_inspect)
        else:
            iterable = (
                (id, flow, label)
                for id, (flow, label)
                in enumerate(zip(x_raw, y_raw))
            )
            total = len(x_raw)

        #   ################################################################    #
        #   Definizione della barra di avanzamento con tqdm.

        progress = tqdm(
            iterable,
            total = total,
            desc = "Elaborazione flussi",

            # Nasconde la barra di avanzamento se il debug è attivo, per evitare output ridondanti.
            disable = debug
        )

        #   ################################################################    #
        #   Inizio del ciclo di elaborazione dei flussi.

        for (i, flow, label) in progress:

            #   ############################################################    #
            #   Stampa un messaggio di debug all'inizio
            #   per monitorare l'avanzamento.

            if debug:
                print(f"[DEBUG] Elaborazione del flusso n.{i}")
                print(f"[DEBUG] Etichetta: {label}")
                print(f"[DEBUG] Numero di pacchetti nel flusso: {flow.shape[0]}\n")

            #   ############################################################    #
            #   Cancellazione dei pacchetti di padding

            # Crea una maschera che salva i metadati dei pacchetti che
            # compaiono prima della prima riga di padding nel flusso.
            # Usando ".all(axis=1)" si identifica solo le righe interamente a -1,
            # che sono inequivocabilmente padding.
            padding_mask = (flow == -1).all(axis=1)

            # Si tronca l'array al primo indice di padding trovato.
            padding_indices = np.where(padding_mask)[0]

            # Se esistono righe di padding, il taglio avviene al primo indice di padding;
            # altrimenti il flusso è completo (nessun padding) e si conservano tutte le T righe.
            cutoff = padding_indices[0] if len(padding_indices) > 0 else len(flow)
            real_rows = flow[:cutoff]

            # Si salta il flusso se non contiene pacchetti reali.
            if len(real_rows) == 0:
                if debug:
                    print(f"[DEBUG] Flusso n.{i} scartato: nessun pacchetto reale trovato.")
                continue

            #   ############################################################    #
            #   Estrazione dei timestamp e delle dimensioni dei pacchetti

            # Array dei timestamp di arrivo di ciascun pacchetto,
            # espressi in secondi assoluti dall'inizio della cattura.
            # Dato che i timestamp sono in millisecondi, si divide per 1000.0 per ottenere i secondi.
            ts = np.cumsum(real_rows[:, 3]) / 1000.0

            # Array delle dimensioni in byte di ciascun pacchetto,
            # nell'ordine corrispondente ai timestamp.
            sizes = real_rows[:, 1].astype(int)

            if debug:
                print(f"[DEBUG] Timestamp e dimensioni pacchetti:")
                print(pd.DataFrame({'Timestamp': ts, 'Size': sizes}), "\n")

            #   ############################################################    #
            #   Applicazione di filtri di qualità

            if ts.max() - ts.min() <= 0:
                if debug:
                    print(f"[DEBUG] Flusso n.{i} scartato per durata non valida.")
                continue

            if min_packets:
                filter_1 = sizes.shape[0]
                if not(filter_1 > min_packets):
                    if debug:
                        print(f"[DEBUG] Flusso n.{i} scartato per numero pacchetti insufficiente ({filter_1}).")
                    continue

            if min_dim:
                filter_2 = np.sum(sizes)
                if not(filter_2 > min_dim):
                    if debug:
                        print(f"[DEBUG] Flusso n.{i} scartato per dimensione insufficiente ({filter_2} byte).")
                    continue

            if min_tps:
                filter_3 = ts[-1] - ts[0]
                if not(filter_3 > min_tps):
                    if debug:
                        print(f"[DEBUG] Flusso n.{i} scartato per durata insufficiente ({filter_3:.2f} secondi).")
                    continue

            #   ############################################################    #
            #   Costruzione dell'istogramma 2D.

            # Aggiungi 'padding_indices' zeri all'inizio di ts e sizes.
            if len(real_rows) != len(flow):
                ts = np.pad(ts, (len(padding_indices), 0), mode='constant', constant_values=0.0)
                sizes = np.pad(sizes, (len(padding_indices), 0), mode='constant', constant_values=0)

                if debug:
                    print(f"[DEBUG] Flusso n.{i} contiene {len(real_rows)} pacchetti reali e {len(padding_indices)} pacchetti di padding.")
                    print(f"[DEBUG] ts.shape = {ts.shape}, sizes.shape = {sizes.shape}")
                
                # end if
            
            if debug:
                print(f"[DEBUG] Pacchetti reali del flusso n.{i}:")
                print(pd.DataFrame(real_rows, columns=FEATURES_LIST), "\n")
            
            hist = session_2d_histogram(
                ts, sizes,
                plot = (debug or flows_to_inspect is not None),
                title = label
            )

            #   ########################################################    #
            #   Aggiunta dell'istogramma al dataset.

            dataset.append([hist])
            counter += 1

            # end for (i, (flow, label))
        # end open
    
    #   ####################################################################    #
    #   RITORNO DEL DATASET

    ret = np.asarray(dataset)

    if debug:
        print(f"[DEBUG] Conversione completata.")
        print(f"[DEBUG] Percentuale di finestre temporali valide: {counter/len(x_raw) * 100:.2f}%")
        print(f"[DEBUG] Dimensione del dataset risultante: {ret.shape}")

    return ret

    # end

def session_2d_histogram(ts, sizes, plot=False, title=None):
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
        title (str, optional): Titolo della finestra di traffico, usato per il plot. Defaults to None.

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

    if len(ts) != len(sizes):
        raise ValueError("Gli array 'ts' e 'sizes' devono avere la stessa lunghezza.")

    #   ####################################################################    #
    #   NORMALIZZAZIONE DEI TIMESTAMP
    #   L'intera durata della finestra viene sempre mappata su 1500 pixel,
    #   indipendentemente da quanto dura in secondi.

    # 1. Rende i timestamp relativi all'inizio della finestra.
    ts_norm = np.array(ts) - ts[0]

    # 2. Scala nell'intervallo [0.0, 1.0].
    ts_norm = ts_norm / (ts.max() - ts.min())

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

        if title is not None:
            plt.title(f"FlowPic - {title}")
        else:
            plt.title("FlowPic")

        plt.xlim(0, MTU)
        plt.xlabel(f"Tempo normalizzato [0 - {MTU}]")
        plt.ylim(0, MTU)
        plt.ylabel(f"PL (Packet Length) [Byte]")
        plt.set_cmap('binary')

        plt.show()

    #   ####################################################################    #
    #   RITORNO DELLA MATRICE 2D

    # Converte i conteggi in interi a 16 bit senza segno
    # per occupare meno spazio in memoria.
    return H.astype(np.uint16)

    # end

#   ########################################################################    #
#   Altre FUNZIONI

def export_dataset(dataset, dir_path, dataset_name, debug=False):
    """ Salva su disco un array NumPy di istogrammi 2D,
    in un file .npy il cui nome identifica le caratteristiche
    del dataset elaborato.

    Args:
        dataset (numpy.ndarray): Array di shape (N, 1, H, W) contenente
            gli istogrammi 2D delle sessioni di traffico, dove:
            - N : il numero di sessioni.
            - H, W : le dimensioni dell'istogramma (tipicamente 1500x1500 in FlowPic).
        dir_path (str): Percorso della cartella in cui salvare il file .npy.
        dataset_name (str): Nome del file PICKLE elaborato per costruire il dataset. Viene utilizzato per estrarre metadati e costruire il nome del file .npy.
        debug (bool, optional): Se True, stampa informazioni di debug durante il salvataggio. Defaults to False.
    """

    #   ####################################################################    #
    #   INIZIALIZZAZIONE

    if debug:
        print(f"[DEBUG] Cartella di salvataggio: {dir_path}")
        print(f"[DEBUG] Nome del file originale: {dataset_name}")

    counter = 0
    campi = []
    indices = [
        0, # dataset name
        4, # number of packets
        5, # number of features
        6, # traffic type
        8, # padding
        # 9  # hash code
    ]

    #   ####################################################################    #
    #   ESTRAZIONE DEI METADATI DAL NOME DEL FILE

    ret_name = dataset_name.split('.')[0]
    metadata = ret_name.split('_')

    for idx in indices:
        if idx < len(metadata) and metadata[idx]:
            campi.append(metadata[idx])
            counter += 1
        elif debug:
            print(f"[DEBUG] Campo metadata[{idx}] non trovato.")
    
    if counter == len(indices):
        ret_name = "_".join(campi)

    #   ####################################################################    #
    #   SALVATAGGIO DEL DATASET SU DISCO

    if debug:
        print(f"[DEBUG] Nome del percorso completo di salvataggio: {dir_path}{ret_name}.npy")

    np.save(f"{dir_path}{ret_name}", dataset)

    # end
