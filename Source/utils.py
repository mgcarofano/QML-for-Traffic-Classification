"""
    ## utils.py \n
    *di Mario Gabriele Carofano*

    ---

    Questo modulo contiene diverse funzioni di utilità per la gestione dell'esecuzione degli esperimenti.
"""

#	####################################################################	#
#	Librerie

from pathlib import Path
from tqdm import tqdm

import io
import json
import numpy as np
import os
import sys
import tempfile
import threading
import time
import traceback

#	####################################################################	#
#	Funzioni di conversione / verifica dei dati

def parse_seconds(seconds: float) -> str:
    # Separa i secondi interi dalla parte decimale
    sec_int = int(seconds)
    milliseconds = int(round((seconds - sec_int) * 1000))

    # Gestione dell'eventuale arrotondamento a 1000 ms
    if milliseconds >= 1000:
        sec_int += 1
        milliseconds -= 1000

    # Formatta ore, minuti e secondi con time.strftime usando l'orario UTC/epoch
    time_str = time.strftime("%H:%M:%S", time.gmtime(sec_int))

    return f"{time_str}.{milliseconds:03d}"

    # end

def check_seconds(value, name: str) -> float | None:
    """Verifica che il valore rappresenti una durata in secondi non negativa e finita.

    Args:
        value: Valore da verificare.
        name (str): Nome del parametro o contesto per i messaggi di errore.

    Raises:
        ValueError: Se il valore non rappresenta una durata valida in secondi.

    Returns:
        float | None: Durata in secondi, oppure None se il valore era None.
    """

    if value is None:
        return None

    value = float(value)

    if not np.isfinite(value) or value < 0:
        raise ValueError(
            f"Errore nella validazione del tempo ({name}): \n"
            f"Durata non finita o negativa."
        )

    return value

    # end

def check_positive_int(
        value, name: str,
        maximum: int = None,
        allow_zero: bool = False,
        allow_none: bool = False
    ) -> int | None:
    """Verifica che il valore sia un intero positivo. Può opzionalmente consentire 0 o None.

    Args:
        value: Valore da verificare.
        name (str): Nome del parametro o contesto per i messaggi di errore.
        maximum (int, optional): Valore massimo consentito. Defaults to None.
        allow_zero (bool, optional): Se True, consente anche il valore 0. Defaults to False.
        allow_none (bool, optional): Se True, consente anche il valore None. Defaults to False.

    Raises:
        TypeError: Se il valore non può essere convertito in un intero.
        ValueError: Se il valore non è un intero positivo.
        ValueError: Se il valore è negativo e allow_zero è True.
        ValueError: Se il valore è None e allow_none è False.
        ValueError: Se il valore supera il massimo consentito (maximum).
    
    Returns:
        int | None: Valore intero validato.
    """

    if value is None:
        if allow_none:
            return None

        raise ValueError(
            f"Errore nella validazione del valore '{name}': \n"
            f"Ricevuto valore None."
        )

    if isinstance(value, bool):
        raise TypeError(
            f"Errore nella validazione del valore '{name}': \n"
            f"Ricevuto valore non intero."
        )

    try:
        value = int(value)
        maximum = int(maximum) if maximum is not None else None
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Errore nella validazione del valore '{name}': \n"
            f"Impossibile convertire il valore in intero."
        ) from exc
    
    if not allow_zero and value <= 0:
        raise ValueError(
            f"Errore nella validazione del valore '{name}': \n"
            f"Ricevuto valore non positivo."
        )

    elif allow_zero and value < 0:
        raise ValueError(
            f"Errore nella validazione del valore '{name}': \n"
            f"Ricevuto valore negativo."
        )

    if maximum is not None and value > maximum:
        raise ValueError(
            f"Errore nella validazione del valore '{name}': \n"
            f"Ricevuto valore maggiore del massimo consentito ({maximum})."
        )

    return value

    # end

def check_integer_array(values, size: int, name: str, allow_duplicates: bool = False) -> np.ndarray:
    """Valida vettori 1D di interi compresi nell'intervallo [0, size) e, opzionalmente, senza duplicati.

    Args:
        values: Vettore 1D da validare.
        size (int): Dimensione massima consentita nel vettore (esclusa).
        name (str): Nome del vettore o del contesto per i messaggi di errore.
        allow_duplicates (bool, optional): Se True, consente interi duplicati. Defaults to False.

    Raises:
        ValueError: Se il vettore non è 1D, è vuoto o contiene valori non interi.
        ValueError: Se il vettore contiene valori fuori dall'intervallo [0, size).
        ValueError: Se il vettore contiene duplicati e allow_duplicates è False.

    Returns:
        np.ndarray: Restituisce una copia validata int64 del vettore di input, mantenendone l'ordine.
    """

    values = np.asarray(values)

    if values.ndim != 1 or values.size == 0 or values.dtype.kind not in "iu":
        raise ValueError(
            f"Errore nella validazione degli indici ({name}): \n"
            "Atteso: vettore 1D non vuoto di interi. \n"
            f"Ricevuto: ndim = {values.ndim} | size = {values.size} | dtype = {values.dtype}"
        )

    if np.any(values < 0) or np.any(values >= size):
        raise ValueError(
            f"Errore nella validazione degli indici ({name}): \n"
            f"Ricevuti indici fuori dal dataset."
        )

    if not allow_duplicates and np.unique(values).size != values.size:
        raise ValueError(
            f"Errore nella validazione degli indici ({name}): \n"
            f"Ricevuti indici duplicati."
        )

    return values.astype(np.int64, copy=True)

    # end

#	####################################################################	#
#	Funzioni di utilità per la scrittura atomica

def atomic_write(path, writer):
    """Crea un file temporaneo nella stessa directory della destinazione, delega la scrittura del contenuto alla funzione writer e, dopo aver sincronizzato il file, lo sposta sul percorso finale tramite os.replace.

    **Nota**
    Se la destinazione esiste già, viene sostituita.
    Se un errore interrompe la preparazione del file temporaneo prima della sostituzione, il file di destinazione preesistente non viene modificato da questa funzione.

    Args:
        path (Path): Percorso del file di destinazione.
        writer (Callable): Funzione che riceve un file binario aperto in scrittura e vi scrive il contenuto.
    """

    path = Path(path)
    temp_path = None

    try:

        # Inizializza un contesto per la creazione di un file temporaneo 'stream' nella directory di destinazione. Con l'opzione 'delete=False', il file temporaneo non viene eliminato automaticamente alla chiusura del contesto, permettendo di spostarlo successivamente sul percorso finale.
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent,
            prefix=f".{path.name}.", delete=False
        ) as stream:

            # Conserva il percorso per la sostituzione e la pulizia.
            temp_path = Path(stream.name)

            # Delega alla funzione 'writer' la scrittura del contenuto.
            writer(stream)

            # Trasferisce i dati dai buffer Python al sistema operativo, garantendo che tutto il contenuto scritto sia effettivamente presente sul disco prima della sostituzione del file.
            stream.flush()

            # Richiede la sincronizzazione del file temporaneo con il disco, assicurando che tutte le modifiche siano effettivamente scritte.
            os.fsync(stream.fileno())

            # end with

        # Sposta il file temporaneo sul percorso finale, sostituendo eventuali file esistenti.
        os.replace(temp_path, path)

    finally:

        # Sia in caso di successo che di eccezione, tenta di rimuovere il file temporaneo se esiste. L'opzione 'missing_ok=True' evita di sollevare un'eccezione se il file è già stato rimosso.
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    # end

def atomic_dump_json(path, obj):
    """Serializza un oggetto in JSON UTF-8 e lo salva in modo atomico, delegando la scrittura alla funzione atomic_write().

    Args:
        path: Percorso del file JSON di destinazione.
        obj: Oggetto da serializzare in JSON.
    """

    payload = json.dumps(
        obj,
        indent=4,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")

    atomic_write(path, lambda stream: stream.write(payload))

    # end

def atomic_dump_text(path: Path, text: str) -> None:
    """Salva uno stream di testo in modo atomico, delegando la scrittura alla funzione atomic_write().

    Args:
        path: Percorso del file di destinazione.
        text: Testo da salvare nel file.
    """

    payload = text.encode("utf-8")

    atomic_write(path, lambda stream: stream.write(payload))

    # end

#	####################################################################	#
#	Funzioni di utilità per la barra di avanzamento personalizzata

def custom_tqdm(*args, **kwargs):
    """ Crea una barra di avanzamento tqdm personalizzata che bypassa il duplicatore degli stream. Utile per visualizzare correttamente le barre di avanzamento nei terminali che utilizzano ProxyStream. """
    
    stream = kwargs.get("file") or sys.stderr
    kwargs["file"] = getattr(stream, "terminal_stream", stream)
    return tqdm(*args, **kwargs)

    # end

#	####################################################################	#
#	Classi

class TerminalLog:
    """
        Sostituisce gli stream Python 'stdout' e 'stderr' con due oggetti ProxyStream, che inoltrano ogni scrittura allo stream originale e a questo gestore.

        Prima della chiamata ad attach() in Main.py, il testo viene conservato in un buffer in memoria. Dopo l'associazione del file, il buffer iniziale viene trasferito nel log e le scritture successive vengono inviate direttamente al file. Questo consente di avere un log completo della sessione.

        Attributes:
            buffer (io.StringIO): Buffer delle scritture precedenti ad attach(). Viene chiuso dopo il trasferimento nel file.
            stream (TextIOWrapper | None): File di log aperto. È None finché il log non viene associato a un percorso.
            lock: Lock rientrante che sincronizza l'accesso al buffer e al file di log.
            stdout: Stream stdout presente all'ingresso nel contesto.
            stderr: Stream stderr presente all'ingresso nel contesto.
    """

    class ProxyStream:
        """
            Inoltra ogni scrittura sia allo stream originale che al gestore TerminalLog, tramite la ridefinizione del metodo write().

            Il proxy non si occupa di chiudere lo stream originale né di gestire il buffering; si limita a inoltrare le scritture. La gestione del buffer è responsabilità del gestore TerminalLog.

            Attributes:
                terminal_stream: Stream originale verso cui inoltrare l'output (es. terminale).
                owner: Gestore che riceve la copia del testo scritto sullo stream originale.
        """

        def __init__(self, terminal_stream, owner):

            self.terminal_stream = terminal_stream
            self.owner = owner

            # end

        def write(self, text):
            """
                Inoltra il testo allo stream originale e al log. Restituisce il numero di caratteri del testo ricevuto, se entrambe le operazioni terminano senza eccezioni.

                Le due scritture non costituiscono una transazione: se una delle due fallisce, l'altra potrebbe comunque essere stata eseguita con successo.
            """

            self.terminal_stream.write(text)
            self.owner.write(text)

            return len(text)

            # end

        def flush(self):
            """ Richiede lo svuotamento dei buffer delle due destinazioni. Inoltra flush() allo stream originale e al gestore del log, senza chiudere le risorse. """

            self.terminal_stream.flush()
            self.owner.flush()

            # end

        def __getattr__(self, name):
            """ Delega allo stream originale tutti gli altri attributi non ridefiniti in questa classe. Solleva comunque AttributeError se l'attributo richiesto non esiste nemmeno nello stream originale. """

            return getattr(self.terminal_stream, name)

            # end

    # end class ProxyStream

    def __init__(self):

        self.buffer = io.StringIO()
        self.stream = None
        self.lock = threading.RLock()

        # end

    def __enter__(self):
        """
            Attiva la duplicazione di stdout e stderr e restituisce l'istanza corrente. Conserva gli stream presenti al momento dell'ingresso e li sostituisce con due oggetti ProxyStream distinti. Questo metodo viene chiamato all'ingresso nel contesto 'with'.

            **Notes** \n
            La cattura inizia all'ingresso nel contesto. Fino ad attach(), la copia del testo rimane nel buffer in RAM. Le scritture successive vengono indirizzate direttamente al file.
        """

        self.stdout = sys.stdout
        sys.stdout = self.ProxyStream(self.stdout, self)

        self.stderr = sys.stderr
        sys.stderr = self.ProxyStream(self.stderr, self)

        return self

        # end

    def attach(self, path):
        """ Associa il log a un nuovo file e vi trasferisce il buffer. """

        with self.lock:

            if self.stream is not None:
                raise RuntimeError("Questo log è già associato a una sessione esistente.")

            # Apre il file in modalità esclusiva.
            # La modalità "x" evita di sovrascrivere un file esistente.
            self.stream = Path(path).open("x", encoding="utf-8", buffering=1)

            # Scrive il testo raccolto prima dell'associazione
            self.stream.write(self.buffer.getvalue())

            # Chiude il buffer iniziale.
            self.buffer.close()
            self.stream.flush()

        # end

    def write(self, text):
        """
            Scrive la copia del testo nel buffer o nel file di log. Se attach() non è ancora stato chiamato, conserva il testo nel buffer. Altrimenti, lo scrive direttamente nel file di log. L'accesso alla destinazione del log è protetto dal lock.

            Questo metodo non scrive sul terminale: tale operazione è gestita dai ProxyStream.
        """

        with self.lock:
            target = \
                self.stream \
                if self.stream is not None \
                else self.buffer

            target.write(text)

        # end

    def flush(self):
        """
            Svuota il buffer di scrittura del file di log, se associato. Prima di attach() non esegue operazioni: il testo iniziale è già conservato nel buffer in memoria.

            Non esegue flush() degli stream originali: tale operazione è responsabilità dei ProxyStream.
        """

        with self.lock:
            if self.stream is not None:
                self.stream.flush()

        # end

    def __exit__(self, *exc):
        """
            Finalizza il log, richiede il flush del file e ripristina gli stream originali. Se viene sollevata un'eccezione, ne registra il traceback direttamente nel file (eccetto nel caso di SystemExit). Questo metodo viene chiamato all'uscita dal contesto 'with'.

            Il traceback viene scritto direttamente nel file per non duplicarlo tramite stderr. La sua eventuale stampa sul terminale dipende dalla gestione dell'eccezione nel codice chiamante.

            Se attach() non è mai stato chiamato, il buffer viene chiuso senza essere salvato su disco.
        """

        try:
            # 'exc' è terna fornita dal protocollo dei context manager. Contiene tipo, istanza e traceback dell'eccezione. In assenza di eccezioni contiene tre None.
            exc_type = exc[0]

            if (
                exc_type is not None
                and self.stream is not None
                and not issubclass(exc_type, SystemExit)
            ):

                # Le eccezioni non intercettate saranno stampate dal chiamante sul terminale.
                traceback.print_exception(*exc, file=self.stream)

            self.flush()

        finally:
            sys.stdout, sys.stderr = self.stdout, self.stderr

            target = \
                self.stream \
                if self.stream is not None \
                else self.buffer
            
            target.close()

        # end

    # end class
