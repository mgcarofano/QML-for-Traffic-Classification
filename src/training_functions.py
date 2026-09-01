"""
    training_functions.py
    di Mario Gabriele Carofano

    Questo modulo contiene le funzioni di supporto
    per l'addestramento e la valutazione dei modelli.

"""

#   ####################################################################    #
#   LIBRARIES

import torch
from tqdm import tqdm
from collections import Counter

#   ####################################################################    #
#   FUNCTIONS

def train_epoch(
        model : torch.nn.Module,
        device : torch.device,
        loader : torch.utils.data.DataLoader,
        criterion : torch.nn.Module,
        optimizer : torch.optim.Optimizer = None,
        scheduler : torch.optim.lr_scheduler._LRScheduler = None,
    ) -> tuple[float, float]:
    """
    Addestra il modello per un'epoca sul DataLoader fornito.

    Args:
        model (torch.nn.Module): Il modello da addestrare.
        device (torch.device): Il device su cui eseguire l'addestramento.
        loader (torch.utils.data.DataLoader): Il DataLoader per il set di dati di addestramento.
        criterion (torch.nn.Module): La funzione di loss.
        optimizer (torch.optim.Optimizer): L'ottimizzatore.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Il scheduler per il learning rate.

    Returns:
        tuple (avg_loss, accuracy)
		avg_loss : float
			Loss media sul loader
		accuracy : float
			Percentuale di predizioni corrette sul loader (0-100).
    """

    #   ################################################################    #
    #   INIZIALIZZAZIONE

    # Set the model into train mode.
    # Ensures layers like Dropout and BatchNorm
    # behave correctly during training.
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

    for inputs, labels in tqdm(loader, desc="Addestramento ", leave=False):

        # Sposta inputs e labels sul device di calcolo solo se necessario:
        # - Se il device è la CPU : il trasferimento è superfluo e viene evitato.
        # - Su GPU/MPS, si abilita un trasferimento asincrono,
        # realmente efficace solo se il batch proviene da un DataLoader con
        # pin_memory=True.
        inputs = inputs.to(device, non_blocking=True) if device.type != 'cpu' else inputs
        labels = labels.to(device, non_blocking=True) if device.type != 'cpu' else labels

        # Si procede con il training step iniziando
        # dall'azzeramento dei gradienti e il forward pass per calcolare
        # le predizioni del modello sul batch corrente.
        optimizer.zero_grad()
        outputs = model(inputs)

        # Si calcola la loss e si procede con il backward pass
        # per aggiornare i pesi del modello.
        loss = criterion(outputs, labels)
        loss.backward()

        # Se viene fornito un optimizer, si esegue un ulteriore step
        # per aggiornare i pesi del modello.
        if optimizer is not None:
            optimizer.step()

        # Step "per batch" dello scheduler.
        # Se viene fornito uno scheduler, applica l'aggiornamento
        # del learning rate secondo la policy selezionata.
        if scheduler is not None and is_scheduler_per_batch:
            scheduler.step()

        # Aggiorna le metriche di monitoraggio della loss e dell'accuracy.
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # end for inputs, labels
    
    # Step "per epoca" dello scheduler.
    # Va eseguito una sola volta, dopo aver esaurito tutti i
    # batch dell'epoca corrente.
    if scheduler is not None and not is_scheduler_per_batch:
        scheduler.step()

    #   ################################################################    #
    #   RITORNO DEI RISULTATI

    avg_loss = (running_loss / len(loader))
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy

    # end

def evaluate(
        model : torch.nn.Module,
        device : torch.device,
        loader : torch.utils.data.DataLoader,
        criterion : torch.nn.Module = None
    ) -> tuple[float | None, float]:
    """
    Valuta il modello su un DataLoader, calcolando sempre l'accuracy e,
    solo se viene fornito un criterion, anche la loss media.

    La funzione è pensata per essere utilizzabile sia in fase di
    training (dove la loss serve per il monitoraggio delle
    epoche) sia in fase di testing su un modello già addestrato
    e caricato da checkpoint. In questo secondo caso, calcolare
    l'accuracy non richiede la loss function: le due metriche sono
    logicamente indipendenti.

    Args:
        model (torch.nn.Module): Il modello da valutare.
        device (torch.device): Il device su cui eseguire la valutazione.
        loader (torch.utils.data.DataLoader): Il DataLoader per il set di dati di valutazione.
        criterion (torch.nn.Module, opzionale):
        La loss function da usare per calcolare la loss media. Default è None.

    Returns:
        tuple (avg_loss, accuracy)
		avg_loss : float | None
			Loss media sul loader, oppure None se non è stato fornito
			un criterion.
		accuracy : float
			Percentuale di predizioni corrette sul loader (0-100).
    """

    #   ################################################################    #
    #   INIZIALIZZAZIONE

    # Disabilita dropout e batch normalization in fase di valutazione.
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    #   ################################################################    #

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Valutazione ", leave=False):
            inputs = inputs.to(device, non_blocking=True) if device.type != 'cpu' else inputs
            labels = labels.to(device, non_blocking=True) if device.type != 'cpu' else labels

            # Si calcolano le predizioni del modello sul batch corrente.
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

            # end for inputs, labels

    #   ################################################################    #
    #   RITORNO DEI RISULTATI

    # Se non è stato fornito un criterion, la loss media resta None
    # invece di essere forzata a 0.0, per non far pensare che il
    # modello abbia una loss realmente nulla.
    avg_loss = (running_loss / len(loader)) if criterion is not None else None
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy

    # end

def compute_class_weights(
        y_train : list,
        num_classes : int
    ) -> torch.FloatTensor:
    """Calcola i pesi di classe per il bilanciamento del dataset,
    da utilizzare nella definizione della loss function
    (es. WeightedCrossEntropy, Focal).

    Args:
        y_train (list): Lista delle etichette di addestramento.
        num_classes (int): Numero totale di classi.

    Returns:
        torch.FloatTensor: Tensor contenente i pesi di ciascuna classe.
    """

    class_counts = Counter(y_train)
    total_samples = sum(class_counts.values())
    weights = []
    for i in range(num_classes):
        count = class_counts.get(i, 0)
        if count > 0:
            weights.append(total_samples / (num_classes * count))
        else:
            weights.append(1.0)

    return torch.FloatTensor(weights)

    # end
