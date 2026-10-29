"""
    training_functions.py
    di Mario Gabriele Carofano

    Questo modulo contiene le funzioni per l'addestramento e la valutazione dei modelli.
"""

#   ####################################################################    #
#   Librerie

import torch
from tqdm import tqdm

#   ####################################################################    #

def train_epoch(
        model, device, loader,
        criterion, optimizer = None, scheduler = None
    ) -> tuple[float, float]:
    """Addestra il modello per un'epoca sul DataLoader fornito.

    Args:
        model: Il modello da addestrare.
        device: Il device su cui eseguire l'addestramento.
        loader: Il DataLoader per il set di dati di addestramento.
        criterion: La funzione di loss.
        optimizer: L'ottimizzatore. Default è None.
        scheduler: Lo scheduler per il learning rate. Default è None.

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

    # Alcuni scheduler (es. OneCycleLR) sono configurati per uno step per
	# ogni batch mentre altri sono pensati per un solo step a fine epoca.
    # Chiamare scheduler.step() con la granularità sbagliata altera
    # l'aggiornamento del learning rate e può compromettere l'addestramento.
    is_scheduler_per_batch = isinstance(
		scheduler, torch.optim.lr_scheduler.OneCycleLR
	)

    #   ################################################################    #
    #   Ciclo di addestramento sui batch del DataLoader.

    for batch in tqdm(loader, desc="Addestramento ", leave=False):

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
    ) -> tuple[float, float, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Valuta il modello su un DataLoader.

    La funzione è pensata per essere utilizzabile sia in fase di training (dove la loss serve per il monitoraggio delle epoche) sia in fase di testing su un modello già addestrato e caricato da checkpoint. In questo secondo caso, calcolare l'accuracy non richiede la loss function: le due metriche sono logicamente indipendenti.

    Args:
        model: Il modello da valutare.
        device: Il device su cui eseguire la valutazione.
        loader: Il DataLoader per il set di dati di valutazione.
        criterion (opzionale): La loss function da usare per calcolare la loss media. Default è None.

    Returns:
        tuple: Una tupla contenente la loss media, l'accuracy, tutte le predizioni e tutte le etichette e tutti i vettori delle probabilità predette (softmax).
    """
    
    #   ################################################################    #
    #   Inizializzazione
    
    # Disabilita dropout e batch normalization in fase di valutazione.
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

    # Si utilizza "torch.no_grad()" per disabilitare il calcolo del gradiente durante la fase di valutazione, migliorando le prestazioni e riducendo l'uso della memoria.
    with torch.no_grad():
        for batch in tqdm(loader, desc="Valutazione ", leave=False):

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

            ignore_idx = criterion.ignore_index if criterion is not None else None
            valid_mask = \
                (labels != ignore_idx) \
                if ignore_idx is not None \
                else torch.ones_like(labels, dtype=torch.bool)
            valid_labels = labels[valid_mask]

            # Aggiorna le metriche di monitoraggio della loss, se fornita.
            # In fase di sola valutazione (es. modello caricato da checkpoint) il criterion potrebbe non esistere, e non è comunque necessario per misurare l'accuracy.
            if criterion is not None:
                loss = criterion(outputs, labels)
                
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

            all_preds.append(predicted.cpu())
            all_labels.append(valid_labels.cpu())

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

    # Se non è stato fornito un criterion, la loss media resta None
    # invece di essere forzata a 0.0, per non far pensare che il
    # modello abbia una loss realmente nulla.
    if criterion is None:
        avg_loss = None
    else:
        if loss_den <= 0:
            raise ValueError("Errore nel calcolo della loss media finale: denominatore nullo.")

        avg_loss = loss_num / loss_den

    accuracy = 100.0 * correct / total

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)

    return (
        avg_loss, accuracy,
        all_preds, all_labels, all_probs
    )

    # end
