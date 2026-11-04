"""
    training_functions.py
    di Mario Gabriele Carofano

    Questo modulo contiene le funzioni per l'addestramento e la valutazione dei modelli.
"""

#   ####################################################################    #
#   Librerie

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from Source.configuration import SessionConfig
from Source.utils import \
    convert, \
    atomic_write, \
    custom_tqdm

import time
import signal
import threading

#   ####################################################################    #
#   Classi e eccezioni personalizzate

class EpochInterrupted(Exception):
    """Stop locale al training del fold, senza traceback."""

    # end class

class FoldInterrupt:
    """
        Gestisce l'interruzione del training interattivo tramite SIGINT (Ctrl+C). Sostituisce temporaneamente l'handler predefinito con un handler che registra una richiesta di stop senza terminare immediatamente il programma. All'uscita dal contesto 'with', ripristina l'handler SIGINT precedente.

        **Utilizzo** \n
        Da utilizzare con la sintassi 'with' all'interno del ciclo di addestramento / validazione. Bisogna verificare periodicamente la richiesta di interruzione tramite il metodo check(). Se lo stop è stato richiesto, check() solleva EpochInterrupted nel punto di controllo.

        Attributes:
            requested (bool): Indica se è stata ricevuta una richiesta di interruzione. Una volta impostato a True, rimane tale per tutta la durata dell'istanza.
            previous: Handler SIGINT presente all'ingresso nel contesto, conservato per ripristinarlo all'uscita.
    """

    def __init__(self):
        self.requested = False

        # end

    def __enter__(self):
        """ Attiva temporaneamente la gestione cooperativa di SIGINT e restituisce l'istanza corrente. Questo metodo viene chiamato all'ingresso nel contesto 'with'. """

        # Verifica che il contesto venga aperto nel thread principale.
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("Il training interattivo deve partire dal thread principale.")

        self.previous = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle)

        return self

        # end

    def _handle(self, signum, frame):
        """ Registra una richiesta di interruzione ricevuta tramite SIGINT. Questo metodo viene chiamato automaticamente dal sistema quando viene ricevuto il segnale SIGINT. I parametri sono richiesti dalla firma dell'handler, anche se non vengono utilizzati. """

        # Imposta requested a True senza sollevare EpochInterrupted e senza eseguire operazioni di finalizzazione.
        self.requested = True

        # end

    def check(self):
        """
            Verifica se è stata richiesta l'interruzione del fold. Consente al codice chiamante di rilevare lo stop in un punto di controllo.

            Dopo una richiesta, ogni chiamata successiva continua a sollevare EpochInterrupted. Non ripristina i pesi del modello, non salva output e non determina quale fold eseguire successivamente. Queste e altre operazioni spettano al codice che intercetta l'eccezione.
        """

        if self.requested:
            raise EpochInterrupted()

        # end

    def __exit__(self, *exc):
        """ Ripristina l'handler SIGINT presente prima del contesto. Questo metodo viene chiamato all'uscita dal contesto 'with'. """

        signal.signal(signal.SIGINT, self.previous)

        # end

    # end class

#   ####################################################################    #
#   Funzioni di addestramento e valutazione

def train_epoch(
        model, device, loader,
        criterion, optimizer = None, scheduler = None,
        stop = None,
    ) -> tuple[float, float]:
    """Addestra il modello per un'epoca sul DataLoader fornito.

    Args:
        model: Il modello da addestrare.
        device: Il device su cui eseguire l'addestramento.
        loader: Il DataLoader per il set di dati di addestramento.
        criterion: La funzione di loss.
        optimizer: L'ottimizzatore. Default è None.
        scheduler: Lo scheduler per il learning rate. Default è None.
        stop: Richiesta cooperativa di interruzione del fold (opzionale).

    Returns:
        tuple[float, float]: Loss media e percentuale di predizioni corrette sul loader.
    """

    #   ################################################################    #
    #   Inizializzazione

    # Set the model into train mode. Ensures layers like Dropout and BatchNorm behave correctly during training.
    model.train()

    loss_num = 0.0
    loss_den = 0.0
    correct = 0
    total = 0

    # Alcuni scheduler (es. OneCycleLR) sono configurati per uno step per ogni batch mentre altri sono pensati per un solo step a fine epoca. Chiamare scheduler.step() con la granularità sbagliata altera l'aggiornamento del learning rate e può compromettere l'addestramento.
    is_scheduler_per_batch = isinstance(
        scheduler, torch.optim.lr_scheduler.OneCycleLR
    )

    #   ################################################################    #
    #   Ciclo di addestramento sui batch del DataLoader.

    for batch in custom_tqdm(loader, desc="Addestramento ", leave=False):

        if stop is not None:
            stop.check()

        inputs, labels, *extra = batch
        lengths = extra[0] if extra else None

        # Sposta inputs e labels sul device di calcolo solo se necessario:
        # - Se il device è la CPU : il trasferimento è superfluo e viene evitato.
        # - Su GPU/MPS, si abilita un trasferimento asincrono, realmente efficace solo se il batch proviene da un DataLoader con pin_memory=True.
        inputs = inputs.to(device, non_blocking=True) if device.type != 'cpu' else inputs

        # Non convertire invece le etichette nello stesso dtype degli output, poiché la funzione di loss CrossEntropyLoss richiede etichette di tipo long.
        labels = labels.to(device, dtype=torch.long, non_blocking=(device.type != "cpu"))

        # Azzeramento dei gradienti dell'ottimizzatore.
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        # FORWARD PASS: calcolo delle predizioni del modello sul batch corrente.
        outputs = model(inputs, lengths)

        expected = (labels.shape[0], model.num_classes)

        if tuple(outputs.shape) != expected:
            raise ValueError(
                "Shape dei logits non valida per la classificazione: \n"
                f"Atteso {expected}. "
                f"Ricevuto {tuple(outputs.shape)}."
            )

        # BACKWARD PASS: calcolo dei gradienti e aggiornamento dei pesi.
        loss = criterion(outputs, labels)
        loss.backward()

        valid_mask = (labels != criterion.ignore_index)
        valid_labels = labels[valid_mask]

        # Se viene fornito un optimizer, si esegue un ulteriore step per aggiornare i pesi del modello.
        if optimizer is not None:
            optimizer.step()

        # Se viene fornito uno scheduler "per batch", applica l'aggiornamento
        # del learning rate secondo la policy selezionata.
        if scheduler is not None and is_scheduler_per_batch:
            scheduler.step()

        # Aggiorna le metriche di monitoraggio della loss.
        batch_den = \
            float(valid_labels.numel()) \
            if criterion.weight is None \
            else float(criterion.weight[valid_labels].sum().item())

        if batch_den <= 0:
            raise ValueError(
                "Errore nel calcolo della loss per epoca: "
                "il batch non contiene target con peso totale positivo."
            )
        
        loss_num += loss.item() * batch_den
        loss_den += batch_den

        # Aggiorna le metriche di monitoraggio dell'accuracy.
        _, predicted = outputs.max(1)
        total += valid_labels.numel()
        correct += (predicted[valid_mask].eq(valid_labels)).sum().item()

        # end for batch

    if stop is not None:
        stop.check()

    # Se viene fornito uno scheduler "per epoca", esegue un passo di aggiornamento del learning rate una sola volta alla fine dell'epoca.
    if scheduler is not None and not is_scheduler_per_batch:
        scheduler.step()

    #   ################################################################    #
    #   Ritorno dei risultati.

    if loss_den <= 0:
        raise ValueError("Errore nel calcolo della loss media finale: denominatore nullo.")

    avg_loss = loss_num / loss_den

    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

    # end

def evaluate(
        model, device, loader,
        criterion = None,
        stop = None,
    ) -> tuple[float | None, float, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Valuta il modello su un DataLoader.

    La funzione è pensata per essere utilizzabile sia in fase di training (dove la loss serve per il monitoraggio delle epoche) sia in fase di testing su un modello già addestrato e caricato da checkpoint. In questo secondo caso, calcolare l'accuracy non richiede la loss function: le due metriche sono logicamente indipendenti.

    Args:
        model: Il modello da valutare.
        device: Il device su cui eseguire la valutazione.
        loader: Il DataLoader per il set di dati di valutazione.
        criterion (opzionale): La loss function da usare per calcolare la loss media. Default è None.
        stop (opzionale): Un oggetto per il controllo dell'interruzione anticipata. Default è None.

    Returns:
        tuple: Una tupla contenente la loss media (o None se non calcolabile), l'accuracy, tutte le predizioni e tutte le etichette e tutti i vettori delle probabilità predette (softmax).
    """
    
    #   ################################################################    #
    #   Inizializzazione
    
    # Imposta il modello in modalità valutazione (inference/evaluation mode).
    # Modifica il comportamento dei layer sensibili allo stato, ad esempio:
    # Dropout: disattiva lo spegnimento casuale dei neuroni, garantendo predizioni deterministiche.
    # BatchNorm: utilizza le medie e varianze correnti (running statistics) apprese durante il training, invece di calcolare le statistiche sui singoli batch correnti.
    model.eval()

    loss_num = 0.0
    loss_den = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []
    all_probs = []

    #   ################################################################    #
    #   Ciclo di valutazione sui batch del DataLoader.

    # Il contesto "torch.no_grad()" disabilita il calcolo dei gradienti durante la fase di valutazione, migliorando le prestazioni e riducendo l'uso della memoria.
    with torch.no_grad():
        for batch in custom_tqdm(loader, desc="Valutazione ", leave=False):

            if stop is not None:
                stop.check()

            inputs, labels, *extra = batch
            lengths = extra[0] if extra else None

            inputs = inputs.to(device, non_blocking=True) if device.type != 'cpu' else inputs
            labels = labels.to(device, dtype=torch.long, non_blocking=(device.type != "cpu"))

            # Forward pass.
            outputs = model(inputs, lengths)

            expected = (labels.shape[0], model.num_classes)

            if tuple(outputs.shape) != expected:
                raise ValueError(
                    "Shape dei logits non valida per la classificazione: \n"
                    f"Atteso {expected}. "
                    f"Ricevuto {tuple(outputs.shape)}."
                )

            valid_mask = \
                (labels != criterion.ignore_index) \
                if criterion is not None \
                else torch.ones_like(labels, dtype=torch.bool)
            valid_labels = labels[valid_mask]
            n_valid = valid_labels.numel()

            # Aggiorna le metriche di monitoraggio della loss, se fornita.
            # In fase di sola valutazione (es. modello caricato da checkpoint) il criterion potrebbe non esistere, e non è comunque necessario per misurare l'accuracy.
            if criterion is not None and n_valid > 0:
                loss = criterion(outputs, labels)
                
                batch_den = \
                    float(n_valid) \
                    if criterion.weight is None \
                    else float(criterion.weight[valid_labels].sum().item())

                if batch_den <= 0:
                    raise ValueError(
                        "Errore nel calcolo della loss per epoca: "
                        "il batch non contiene target con peso totale positivo."
                    )
                
                loss_num += loss.item() * batch_den
                loss_den += batch_den

                # end if

            # Aggiorna le metriche di monitoraggio dell'accuracy.
            _, predicted = outputs.max(1)
            total += valid_labels.numel()
            correct += (predicted[valid_mask].eq(valid_labels)).sum().item()

            all_preds.append(predicted.cpu())
            all_labels.append(labels.cpu())

            #   ########################################################    #
            #   Calcolo delle probabilità per ciascuna classe.

            probabilities = torch.softmax(outputs, dim=1)

            if not torch.allclose(
                probabilities.sum(dim=1),
                torch.ones(
                    probabilities.shape[0],
                    device=probabilities.device,
                    dtype=probabilities.dtype
                ),
                atol=1e-5,
            ):
                raise ValueError("Le probabilità non sommano a 1.")
            
            all_probs.append(probabilities.cpu())

        # end for batch

    #   ################################################################    #
    #   Ritorno dei risultati.

    if stop is not None:
        stop.check()

    # Se non è stato fornito un criterion, la loss media resta None
    # invece di essere forzata a 0.0, per non far pensare che il
    # modello abbia una loss realmente nulla.
    if criterion is None:
        avg_loss = None
    else:
        if loss_den <= 0:
            raise ValueError(
                "Errore nel calcolo della loss media finale: \n"
                "denominatore nullo."
            )

        avg_loss = loss_num / loss_den

    if total <= 0:
        raise ValueError(
            "Errore nel calcolo dell'accuracy finale: \n"
            "nessun campione valido trovato."
        )

    return (
        avg_loss,
        100.0 * correct / total,
        torch.cat(all_preds),
        torch.cat(all_labels),
        torch.cat(all_probs)
    )

    # end

#   ####################################################################    #
#   Funzioni principali

def train_model(
        model, device,
        train_loader, val_loader,
        optimizer, lr_sched, criterion,
        fold_model_path: str,
        sc: SessionConfig,
    ) -> tuple[torch.nn.Module, pd.DataFrame, int | None, float | None]:
    """Addestra il modello per un numero specificato di epoche. Si può interrompere anticipatamente l'addestramento con Ctrl+C scartando l'epoca incompleta.

    ## Funzionalità
        - Monitora le metriche di addestramento e validazione.
        - Implementa early stopping e aggiorna i pesi del modello in base alla metrica di valutazione selezionata.
        - Supporta l'interruzione anticipata dell'addestramento con Ctrl+C, mantenendo i pesi aggiornati fino all'epoca migliore.

    Args:
        model: Il modello da addestrare.
        device: Il dispositivo su cui eseguire l'addestramento.
        train_loader: Il DataLoader per il set di addestramento.
        val_loader: Il DataLoader per il set di validazione.
        optimizer: L'ottimizzatore da utilizzare.
        lr_sched: Il scheduler del learning rate.
        criterion: La funzione di loss.
        fold_model_path (str): Il percorso dove viene salvato il modello aggiornato.
        sc (SessionConfig): Configurazione dei parametri di sessione per l'esperimento corrente.
    
    Returns:
        tuple: il modello addestrato, il DataFrame contenente lo storico dell'addestramento, l'epoca e il punteggio migliore (o None se non disponibili).
    """

    #	################################################################	#
    #	Inizializzazione delle metriche da monitorare.

    # Questo dizionario terrà traccia delle metriche di addestramento e validazione per ogni epoca.
    train_history_df = pd.DataFrame(columns=[
        "epoch", "time",
        "train_loss", "train_accuracy",
        "val_loss", "val_accuracy", "val_macrof1",
    ])

    # Inizializza le variabili per il monitoraggio della metrica di valutazione selezionata.
    minimize = sc.training.monitored_metric == "val_loss"
    best_score = float('inf') if minimize else -float('inf')
    best_epoch = None
    patience_counter = 0
    interrupted = False

    elapsed_time = 0.0

    #	################################################################	#
    #	Avvio del loop di addestramento.

    with FoldInterrupt() as stop:
        for epoch in range(1, sc.training.epochs +1):

            # Salva il timestamp di inizio epoca per calcolare la durata dell'epoca corrente.
            epoch_start_time = time.time()

            try:

                stop.check()

                print(f"\nEpoch {epoch}/{sc.training.epochs}")

                #	####################################################	#
                #   Calcola le metriche di addestramento per Train/Val.

                train_loss, train_acc = train_epoch(
                    model, device,
                    train_loader,
                    criterion, optimizer, lr_sched,
                    stop=stop,
                )

                val_loss, val_acc, val_preds, val_true, _ = evaluate(
                    model, device,
                    val_loader,
                    criterion,
                    stop=stop,
                )

                valid_val = val_true != criterion.ignore_index
                val_macrof1 = f1_score(
                    val_true[valid_val].numpy(),
                    val_preds[valid_val].numpy(),
                    labels= np.arange(model.num_classes),
                    average= "macro",
                    zero_division= 0,
                ) * 100

                metrics = {
                    "train_loss": float(train_loss),
                    "train_accuracy": float(train_acc),
                    "val_loss": float(val_loss),
                    "val_accuracy": float(val_acc),
                    "val_macrof1": float(val_macrof1),
                }

                #	####################################################	#
                #   Verifica la validità delle metriche calcolate.

                invalid_metrics = [name for name, value in metrics.items() if not np.isfinite(value)]
                if invalid_metrics:
                    raise RuntimeError("Errore nel calcolo delle metriche di addestramento: valori non finiti per [" + ", ".join(invalid_metrics) + "]")

                stop.check()

            except (EpochInterrupted, KeyboardInterrupt):
                interrupted = True
                print(f"\nTraining interrotto all'epoca {epoch}.")
                break

            # end try

            #	########################################################	#
            #   Salvataggio del checkpoint in caso di miglioramento.

            # Calcola il punteggio corrente in base alla metrica monitorata.
            current_score = metrics[sc.training.monitored_metric]

            # Verifica se il punteggio corrente rappresenta un miglioramento rispetto al miglior punteggio precedente.
            if best_epoch is None:
                improved = True
            elif minimize:
                improved = current_score < (best_score - sc.training.min_delta)
            else:
                improved = current_score > (best_score + sc.training.min_delta)

            if improved:
                print(
                    f"  -> '{sc.training.monitored_metric}' improved. \n"
                    f"     Model saved in {fold_model_path}."
                )

                # Salvataggio del modello corrente.
                atomic_write(fold_model_path, lambda stream: torch.save(model.state_dict(), stream))

                # Aggiornamento del miglior punteggio e reset del contatore di patience.
                best_score = current_score
                best_epoch = epoch
                patience_counter = 0

            elif sc.training.early_stopping:
                patience_counter += 1
                print(f"  -> No improvement on '{sc.training.monitored_metric}'.")
                print(f"     Patience: {patience_counter} of {sc.training.patience}")

            else:
                print(
                    f"  -> No improvement on '{sc.training.monitored_metric}'. \n" 
                    f"     Early stopping is disabled."
                )

            #	########################################################	#
            #   Finalizzazione dell'epoca.

            # Calcola la durata dell'epoca corrente e il tempo totale trascorso dall'inizio dell'addestramento.
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            elapsed_time += epoch_duration

            # Aggiorna lo storico delle metriche per l'epoca corrente.
            train_history_df.loc[len(train_history_df)] = {
                "epoch": epoch,
                "time": epoch_duration,
                **metrics
            }

            # Stampa in console le metriche dell'epoca corrente.
            print(
                f"\nTrain Loss: {train_loss:.4f} "
                f"| Train Accuracy: {train_acc:.4f}%\n"

                f"Val Loss: {val_loss:.4f} "
                f"| Val Accuracy: {val_acc:.4f}% "
                f"| Val Macro F1: {(val_macrof1):.4f}%\n"

                f"Epoch time: {convert(epoch_duration)} "
                f"| Elapsed time: {convert(elapsed_time)}"
            )

            # Controlla se è stato richiesta l'interruzione del training.
            if stop.requested:
                interrupted = True
                print("\nStop ricevuto durante la finalizzazione dell'epoca completa.")
                break

            # Se il contatore di patience raggiunge il limite si interrompe il training.
            if sc.training.early_stopping and patience_counter >= sc.training.patience:
                print("\nInterruzione dell'addestramento per early stopping.")
                break

            # end for epoch

        # end with stop

    #	################################################################	#
    #	Fine del loop di addestramento.

    if best_epoch is not None:
        # Ricarica i pesi del modello in base al miglior punteggio di validazione sulla metrica monitorata.
        model.load_state_dict(torch.load(
            fold_model_path,
            map_location=device,
            weights_only=True,
        ))

    return model, train_history_df, best_epoch, best_score, interrupted

    # end

def test_model(
        model, device,
        test_loader, metadata_test,
        criterion=None,
    ):
    """Calcola il soft-output per il test set, restituendo un Dataframe contenente il FlowID, i risultati effettivi, le predizioni e il soft-output.

    Args:
        model: Il modello da valutare.
        device: Il dispositivo su cui eseguire il calcolo.
        test_loader: Il DataLoader per il test set.
        metadata_test: Il DataFrame contenente i metadati del test set.
        criterion (opzionale): La funzione di loss da utilizzare. Default è None.

    Returns:
        pd.DataFrame: Contiene il FlowID, i risultati effettivi, le predizioni e il soft-output.
    """

    test_loss, test_acc, y_preds, y_true, y_probs = evaluate(
        model, device,
        test_loader,
        criterion
    )

    valid_mask = \
        y_true != criterion.ignore_index \
        if criterion is not None \
        else torch.ones_like(y_true, dtype=torch.bool)

    val_macrof1 = f1_score(
        y_true[valid_mask].numpy(),
        y_preds[valid_mask].numpy(),
        labels= np.arange(model.num_classes),
        average= "macro",
        zero_division= 0,
    ) * 100

    # Stampa le metriche di valutazione del modello su dati mai visti prima.
    print(f"Test Loss: {test_loss:.4f} | " if test_loss is not None else "", end="")
    print(f"Test Acc: {test_acc:.4f}%" + f" | Test Macro F1: {val_macrof1:.4f}%")

    #	################################################################	#
    #	Verifica e ritorno dei risultati.

    if not (
        len(metadata_test)
        == y_true.numel()
        == y_preds.numel()
        == y_probs.shape[0]
    ):
        raise ValueError(
            "Errore nel calcolo del soft-output: \n"
            "Disallineamento tra "
            f"metadati ({len(metadata_test)}), "
            f"etichette ({y_true.numel()}), "
            f"predizioni ({y_preds.numel()}) e, "
            f"probabilità ({y_probs.shape[0]})."
        )

    if torch.any((y_true < 0) | (y_true >= model.num_classes)):
        raise ValueError(
            "Errore nel calcolo del soft-output: \n"
            f"Il test set contiene etichette fuori dall'intervallo [0, {model.num_classes})."
        )

    soft_output = [
        ";".join(f"{probability:.8f}" for probability in row)
        for row in y_probs.tolist()
    ]

    ret = pd.DataFrame({
        "FlowID": metadata_test["FlowID"].tolist(),
        "Actual": y_true.tolist(),
        "Predicted": y_preds.tolist(),
        "Soft-output": soft_output,
    })

    return ret

    # end
