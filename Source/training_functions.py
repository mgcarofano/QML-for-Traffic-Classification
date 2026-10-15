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

    running_loss = 0.0
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

    for inputs, labels in tqdm(loader, desc="Addestramento "):

        # Sposta inputs e labels sul device di calcolo solo se necessario:
        # - Se il device è la CPU : il trasferimento è superfluo e viene evitato.
        # - Su GPU/MPS, si abilita un trasferimento asincrono, realmente efficace solo se il batch proviene da un DataLoader con pin_memory=True.
        inputs = inputs.to(device, non_blocking=True) if device.type != 'cpu' else inputs

        # Non convertire invece le etichette nello stesso dtype degli output, poiché la funzione di loss CrossEntropyLoss richiede etichette di tipo long.
        labels = labels.to(device, dtype=torch.long, non_blocking=(device.type != "cpu"))

        # Azzeramento dei gradienti dell'ottimizzatore.
        optimizer.zero_grad()

        # FORWARD PASS: calcolo delle predizioni del modello sul batch corrente.
        outputs = model(inputs)

        # BACKWARD PASS: calcolo dei gradienti e aggiornamento dei pesi.
        loss = criterion(outputs, labels)
        loss.backward()

        # Se viene fornito un optimizer, si esegue un ulteriore step per aggiornare i pesi del modello.
        if optimizer is not None:
            optimizer.step()

        # Se viene fornito uno scheduler "per batch", applica l'aggiornamento
        # del learning rate secondo la policy selezionata.
        if scheduler is not None and is_scheduler_per_batch:
            scheduler.step()

        # Aggiorna le metriche di monitoraggio della loss e dell'accuracy.
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # end for inputs, labels
    
    # Se viene fornito uno scheduler "per epoca", esegue un passo di aggiornamento del learning rate una sola volta alla fine dell'epoca.
    if scheduler is not None and not is_scheduler_per_batch:
        scheduler.step()

    #   ################################################################    #
    #   Ritorno dei risultati.

    avg_loss = (running_loss / len(loader))
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

    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []
    all_probs = []

    #   ################################################################    #
    #   Ciclo di valutazione sui batch del DataLoader.

    # Si utilizza "torch.no_grad()" per disabilitare il calcolo del gradiente durante la fase di valutazione, migliorando le prestazioni e riducendo l'uso della memoria.
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Valutazione "):

            inputs = inputs.to(device, non_blocking=True) if device.type != 'cpu' else inputs
            labels = labels.to(device, dtype=torch.long, non_blocking=(device.type != "cpu"))

            # Forward pass.
            outputs = model(inputs)

            # La loss viene calcolata solo se è stato fornito un criterion:
            # in fase di sola valutazione (es. modello caricato da
            # checkpoint) il criterion potrebbe non esistere, e non è
            # comunque necessario per misurare l'accuracy.
            if criterion is not None:
                loss = criterion(outputs, labels)
                running_loss += loss.item()

            # Aggiorna le metriche di monitoraggio della loss e dell'accuracy.
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

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

        # end for inputs, labels

    #   ################################################################    #
    #   Ritorno dei risultati.

    # Se non è stato fornito un criterion, la loss media resta None
    # invece di essere forzata a 0.0, per non far pensare che il
    # modello abbia una loss realmente nulla.
    avg_loss = (running_loss / len(loader)) if criterion is not None else None

    accuracy = 100.0 * correct / total

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)

    return (
        avg_loss, accuracy,
        all_preds, all_labels, all_probs
    )

    # end
