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
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from constants import (
    FEATURES_LIST,
    MTU, BIN_SIZE
)

#   ########################################################################    #
#   FUNZIONI di ANALISI del traffico

def mirage_pickle_converter(
    file_path: str,
    filters: dict = None,
    debug : bool = False,
    debug_cycle : bool = False,
    flows_to_inspect: list[int] = None
) -> Tuple[np.ndarray, pd.DataFrame]:
    
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
        Ogni etichetta rappresenta il nome dell'applicazione o dell'attività
        associata al flusso corrispondente.

    ## Features per pacchetto
    - DIR                       : Direzione del pacchetto (0.0 = upstream, 1.0 = downstream).
    - PL (Packet Length)        : Dimensione del payload IP in byte
    - WIN (TCP Window Size)     : Dimensione finestra TCP (0.0 per UDP).
    - IAT (Inter-Arrival Time)  : Tempo tra pacchetti consecutivi in secondi.

    Args:
        file_path (str): Percorso del file PICKLE da convertire.
        filters (dict, optional): 
        Dizionario contenente i filtri da applicare ai flussi. Defaults to None.
        Può contenere le seguenti chiavi: 'min_tps', 'min_packets', 'min_dim'.
        debug (bool, optional):
        Se True, stampa informazioni di debug e il plot dell'istogramma 2D
        durante la conversione. Defaults to False.
        debug_cycle (bool, optional):
        Se True, stampa informazioni di debug per ogni ciclo di elaborazione.
        Defaults to False.
        flows_to_inspect (list, optional):
        Lista di indici dei flussi da selezionare per il debug. Defaults to None.

    Returns:
        numpy.ndarray: Array di shape (N, 1, D, D) contenente gli
        istogrammi 2D delle sessioni di traffico, dove N è il numero di
        finestre temporali valide e D è la dimensione dell'istogramma, calcolata in base ai parametri MTU e BIN_SIZE.
        Restituisce un array vuoto di shape (0,) se nessuna finestra valida
        viene trovata nel file. \n

        **pd.DataFrame** \n
        DataFrame contenente i metadati dei flussi validi, con le colonne "FlowID", "DatasetID" e "Label".
    """

    #   ####################################################################    #
    #   VALIDITÀ DEI PARAMETRI

    if flows_to_inspect is not None and not isinstance(flows_to_inspect, list):
        raise ValueError("Il parametro 'flows_to_inspect' deve essere una lista di indici.")
    
    if filters and not isinstance(filters, dict):
        raise ValueError("Il parametro 'filters' deve essere un dizionario.")
    
    debug = debug or (flows_to_inspect is not None)
    debug_cycle = debug_cycle or (flows_to_inspect is not None)

    #   ####################################################################    #
    #   INIZIALIZZAZIONE

    if debug:
        print(f"[DEBUG] Inizio conversione del file: {file_path}")

    dataset = []

    flow_ids = []
    dataset_ids = []
    labels = []

    min_packets = None
    min_dim = None
    min_tps = None

    counter = {
        # Contatore delle finestre temporali valide.
        "valid": 0,

        # Contatore delle finestre temporali scartate
        # perché non contengono pacchetti reali (solo padding).
        "no_real_packets": 0,

        # Contatore delle finestre temporali scartate
        # per durata non valida.
        "invalid_duration": 0,

        # Contatore delle finestre temporali scartate
        # per numero di pacchetti insufficiente.
        "insufficient_packets": 0,

        # Contatore delle finestre temporali scartate
        # per dimensione insufficiente.
        "insufficient_dim": 0,

        # Contatore delle finestre temporali scartate
        # per durata insufficiente.
        "insufficient_tps": 0
    }
    
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
            disable = debug_cycle
        )

        #   ################################################################    #
        #   Inizio del ciclo di elaborazione dei flussi.

        for (i, flow, label) in progress:

            #   ############################################################    #
            #   Stampa un messaggio di debug all'inizio
            #   per monitorare l'avanzamento.

            if debug_cycle:
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
                counter["no_real_packets"] += 1
                if debug_cycle:
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

            if debug_cycle:
                print(f"[DEBUG] Timestamp e dimensioni pacchetti:")
                print(pd.DataFrame({'Size': sizes, 'Timestamp': ts}), "\n")

            #   ############################################################    #
            #   Applicazione di filtri di qualità

            if ts.max() - ts.min() <= 0:
                counter["invalid_duration"] += 1
                if debug_cycle:
                    print(f"[DEBUG] Flusso n.{i} scartato per durata non valida.")
                continue

            if min_packets:
                filter_1 = sizes.shape[0]
                if not(filter_1 > min_packets):
                    counter["insufficient_packets"] += 1
                    if debug_cycle:
                        print(f"[DEBUG] Flusso n.{i} scartato per numero pacchetti insufficiente ({filter_1}).")
                    continue

            if min_dim:
                filter_2 = np.sum(sizes)
                if not(filter_2 > min_dim):
                    counter["insufficient_dim"] += 1
                    if debug_cycle:
                        print(f"[DEBUG] Flusso n.{i} scartato per dimensione insufficiente ({filter_2} byte).")
                    continue

            if min_tps:
                filter_3 = ts[-1] - ts[0]
                if not(filter_3 > min_tps):
                    counter["insufficient_tps"] += 1
                    if debug_cycle:
                        print(f"[DEBUG] Flusso n.{i} scartato per durata insufficiente ({filter_3:.2f} secondi).")
                    continue

            #   ############################################################    #
            #   Costruzione dell'istogramma 2D.

            # Aggiungi 'padding_indices' zeri all'inizio di ts e sizes.
            if len(real_rows) != len(flow):
                ts = np.pad(ts, (len(padding_indices), 0), mode='constant', constant_values=0.0)
                sizes = np.pad(sizes, (len(padding_indices), 0), mode='constant', constant_values=0)

                if debug_cycle:
                    print(f"[DEBUG] Flusso n.{i} contiene {len(real_rows)} pacchetti reali e {len(padding_indices)} pacchetti di padding.")
                    print(f"[DEBUG] sizes.shape = {sizes.shape}, ts.shape = {ts.shape}")
                
                # end if
            
            if debug_cycle:
                print(f"[DEBUG] Pacchetti reali del flusso n.{i}:")
                print(pd.DataFrame(real_rows, columns=FEATURES_LIST), "\n")
            
            hist = session_2d_histogram(
                sizes, ts,
                plot = debug_cycle,
                title = label
            )

            #   ########################################################    #
            #   Aggiunta dell'istogramma al dataset.

            flow_ids.append(i)
            dataset_ids.append(counter["valid"])
            labels.append(label)

            dataset.append([hist])
            counter["valid"] += 1

            # end for (i, (flow, label))
        # end open
    
    #   ####################################################################    #
    #   RITORNO DEL DATASET e dei METADATI

    debug_counter = pd.DataFrame(counter, index=["[DEBUG] Conteggio dei pacchetti"]).T
    if debug_counter is not None and not debug_counter.empty:
        assert debug_counter.sum().iloc[0] == len(x_raw), \
            "[ERROR] Il conteggio dei pacchetti" \
            "non corrisponde al numero di flussi in input."

    ret = np.asarray(dataset, dtype=np.float32)

    metadata = pd.DataFrame({
        "FlowID": flow_ids,
        "DatasetID": dataset_ids,
        "Label": labels,
    })

    if debug:
        print(f"\n[DEBUG] Conversione completata.")
        print(debug_counter)
        print(f"\n[DEBUG] Numero totale di flussi nel file: {len(x_raw)}")
        print(f"[DEBUG] Percentuale di finestre temporali valide: {counter['valid']}/{len(x_raw)} = {counter['valid']/len(x_raw) * 100:.2f}%")
        print(f"[DEBUG] Dimensione del dataset risultante: {ret.shape}")

    return ret, metadata

    # end

def session_2d_histogram(sizes, ts, plot=False, title=None):
    """ È la funzione chiave che costruisce un FlowPic, cioè un istogramma
    2D che rappresenta la distribuzione spazio-temporale dei pacchetti
    di una finestra di traffico di rete.

    L'asse X rappresenta il tempo normalizzato nell'intervallo [0, 1500].
    L'asse Y rappresenta la dimensione dei pacchetti in byte, da 0 a 1500 (MTU).

    Args:
        sizes (numpy.ndarray): Array 1D delle dimensioni in byte dei pacchetti,
            allineato per indice con ts: cioè, sizes[i] è la dimensione del
            pacchetto ts[i]. Shape: (N,).
        ts (numpy.ndarray): Array 1D dei timestamp di arrivo dei pacchetti,
            espressi in secondi relativi all'inizio della finestra. Shape: (N,).
        plot (bool, optional): Se True, visualizza l'istogramma con matplotlib
            usando una colormap binaria (cioè, bianco se 0 pacchetti, nero se 1+
            pacchetti). Defaults to False.
        title (str, optional): Titolo della finestra di traffico, usato per il plot. Defaults to None.

    Returns:
        numpy.ndarray: Matrice 2D di shape (D, D) e dtype uint16,
            dove H[y, x] è il numero di pacchetti con dimensione y byte arrivati
            al tempo normalizzato x. La dimensione D è calcolata in base ai parametri MTU e BIN_SIZE.
    """

    #   ####################################################################    #
    #   INIZIALIZZAZIONE

    sizes = np.asarray(sizes)
    ts = np.asarray(ts)

    #   ####################################################################    #
    #   CONTROLLO DI VALIDITÀ DEGLI ARGOMENTI

    if sizes.shape[0] != ts.shape[0]:
        raise ValueError("Gli array 'ts' e 'sizes' devono avere la stessa lunghezza.")

    #   ####################################################################    #
    #   NORMALIZZAZIONE DEI TIMESTAMP
    #   L'intera durata della finestra viene sempre mappata su 1500 pixel,
    #   indipendentemente da quanto dura in secondi.

    # 1. Rende i timestamp relativi all'inizio della finestra.
    ts_norm = ts - ts[0]

    # 2. Scala nell'intervallo [0.0, 1.0].
    ts_norm = ts_norm / (ts.max() - ts.min())

    # 3. Scala nell'intervallo [0, 1500].
    ts_norm = ts_norm * MTU

    #   ####################################################################    #
    #   COSTRUZIONE DEL FLOWPIC

    H, _, _ = np.histogram2d(

        # Gli assi X e Y dell'istogramma 2D rappresentano rispettivamente le
        # dimensioni dei pacchetti e il tempo normalizzato.
        sizes,
        ts_norm,

        # Costruisce una griglia di bins 1500x1500 per l'istogramma 2D.
        # Fissa esplicitamente i limiti dei bin a [0, MTU] su entrambi gli assi.
        # Senza questo parametro il calcolo dei limiti dipende dai valori
        # min/max dei dati, portando a risultati incoerenti tra dataset
        # e istogramma.
        bins = (range(0, MTU + 1, BIN_SIZE), range(0, MTU + 1, BIN_SIZE)),
    )

    #   ####################################################################    #
    #   NORMALIZZAZIONE RISPETTO AL NUMERO DI PACCHETTI
    #   Normalizza l'istogramma in modo che la somma di tutti i pixel sia 1.

    if H.sum() > 0:
        H = H / H.sum()

    #   ####################################################################    #
    #   VISUALIZZAZIONE OPZIONALE DELL'ISTOGRAMMA

    if plot:

        H_plot = H

        # Si riduce la risoluzione dell'istogramma per il plot,
        # raggruppando i pixel in blocchi di dimensione BIN_SIZE x BIN_SIZE.
        # Dopo il reshape, le colonne 0, 2 indicizzano i blocchi originali,
        # mentre le colonne 1, 3 contano il totale dei pacchetti in ciascun blocco.
        # if BIN_SIZE > 1:
        #     n = MTU // BIN_SIZE
        #     H_plot = H[:n*BIN_SIZE, :n*BIN_SIZE] \
        #         .reshape(n, BIN_SIZE, n, BIN_SIZE) \
        #         .sum(axis=(1, 3))

        plt.pcolormesh(
            # Costruisce la griglia di coordinate per il plot.
            np.linspace(0, MTU, H_plot.shape[1] + 1),
            np.linspace(0, MTU, H_plot.shape[0] + 1),

            # L'array 2D da visualizzare, convertito in uint8 per ridurre l'uso di memoria. Sono rappresentati solo le celle contenenti almeno un pacchetto.
            (H_plot > 0).astype(np.uint8),

            # Imposta la colormap binaria inversa (bianco = 0 pacchetti, nero = 1+ pacchetti) e i limiti di visualizzazione.
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

    # Converte i conteggi in numeri a virgola mobile a 32 bit per ridurre
    # l'uso di memoria, dato che i valori sono normalizzati tra 0 e 1.
    return H.astype(np.float32)

    # end

#   ########################################################################    #
#   ALTRE FUNZIONI

def get_dataset_name(dir_path, raw_pickle_name, debug=False):
    """ Estrae i metadati dal nome del file PICKLE e costruisce il nuovo nome
    del file .npy in cui salvare il dataset di istogrammi 2D (FlowPics).

    Args:
        dir_path (str): Percorso della cartella originale in cui salvare il file .npy.
        raw_pickle_name (str): Nome del file PICKLE elaborato per costruire il dataset.
        Viene utilizzato per estrarre i metadati necessari.
        debug (bool, optional): Se True, stampa informazioni di debug durante
        il salvataggio. Defaults to False.
    """

    #   ####################################################################    #
    #   INIZIALIZZAZIONE

    if debug:
        print(f"[DEBUG] Cartella di salvataggio: {dir_path}")
        print(f"[DEBUG] Nome del file originale: {raw_pickle_name}")

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

    dataset_name = raw_pickle_name.split('.')[0]
    metadata = dataset_name.split('_')

    for idx in indices:
        if idx < len(metadata) and metadata[idx]:
            campi.append(metadata[idx])
            counter += 1
        elif debug:
            print(f"[DEBUG] Campo metadata[{idx}] non trovato.")

    if counter == len(indices):
        dataset_name = "_".join(campi)

    #   ####################################################################    #
    #   RITORNO DEL NOME DEL FILE

    if debug:
        print(f"[DEBUG] Nome del percorso completo di salvataggio: {dir_path}/{dataset_name}.npy")

    return dataset_name

    # end
