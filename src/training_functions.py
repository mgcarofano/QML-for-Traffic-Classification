"""
    training_functions.py
    di Mario Gabriele Carofano

    Questo modulo contiene le funzioni di supporto
    per l'addestramento e la valutazione dei modelli.

"""

#   ####################################################################    #
#   LIBRARIES

import torch
import torch.nn as nn
from tqdm import tqdm
from collections import Counter

#   ####################################################################    #
#   Funzioni di utilità per la scelta degli iperparametri

def build_optimizer(
        selected_optimizer : str,
        model : nn.Module,
        learning_rate : float,
        weight_decay : float,
        epsilon : float,
        momentum : float,
    ) -> torch.optim.Optimizer:

    """Costruisce l'ottimizzatore in base alla selezione dell'utente.

    Args:
        selected_optimizer (str): Tipo di ottimizzatore selezionato dall'utente.
        model (nn.Module): Modello da ottimizzare.
        learning_rate (float): Learning rate dell'ottimizzatore.
        weight_decay (float): Peso del termine di regolarizzazione L2.
        epsilon (float): Valore epsilon per ottimizzatori come AdamW.
        momentum (float): Momento per ottimizzatori come SGD e RMSprop.

    Raises:
        ValueError: Se il tipo di ottimizzatore selezionato non è supportato.

    Returns:
        torch.optim.Optimizer:
        L'ottimizzatore costruito in base alla selezione dell'utente.
    """

    if selected_optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate, weight_decay=weight_decay, momentum=momentum
        )

    elif selected_optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    elif selected_optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=learning_rate, weight_decay=weight_decay, momentum=momentum
        )

    elif selected_optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate, weight_decay=weight_decay, eps=epsilon
        )

    else:
        raise ValueError(
            f"Tipo di optimizer non supportato: {selected_optimizer}. "
            "Usare: 'Adam', 'AdamW', 'SGD' oppure 'RMSprop'."
        )

    return optimizer

    # end

def build_scheduler(
        selected_scheduler : str,
        optimizer : torch.optim.Optimizer,
        train_loader : torch.utils.data.DataLoader,
        epochs : int,
        max_lr : float = None,
        start_factor : float = None,
        step_size : int = None,
        steplr_gamma : float = None
    ) -> torch.optim.lr_scheduler._LRScheduler | None:

    """Costruisce lo scheduler di apprendimento
    in base alla selezione dell'utente.

    Args:
        selected_scheduler (str): Tipo di scheduler selezionato dall'utente.
        optimizer (torch.optim.Optimizer): Ottimizzatore da associare allo scheduler.
        train_loader (torch.utils.data.DataLoader): DataLoader del set di addestramento.
        epochs (int): Numero totale di epoche di addestramento.
        max_lr (float, opzionale): Massimo learning rate per OneCycleLR.
        start_factor (float, opzionale): Fattore iniziale per LinearLR.
        step_size (int, opzionale): Step size per StepLR.
        steplr_gamma (float, opzionale): Fattore gamma per StepLR.
    
    Raises:
        ValueError: Se il tipo di scheduler selezionato non è supportato.

    Returns:
        torch.optim.lr_scheduler._LRScheduler | None:
        Lo scheduler di apprendimento costruito in base alla selezione dell'utente,
        oppure None se 'NoSched' è selezionato.
    """

    if selected_scheduler == 'OneCycleSched':
        lr_sched = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr, steps_per_epoch=len(train_loader), epochs=epochs
        )

    elif selected_scheduler == 'LinearSched':
        lr_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=start_factor, total_iters=epochs
        )

    elif selected_scheduler == 'StepSched':
        lr_sched = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size, gamma=steplr_gamma
        )

    elif selected_scheduler == 'NoSched':
        lr_sched = None

    else:
        raise ValueError(
            f"Tipo di scheduler non supportato: {selected_scheduler}. "
            "Usare: 'OneCycleSched', 'LinearSched', 'StepSched' oppure 'NoSched'."
        )

    return lr_sched

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

def build_loss_function(
        selected_loss : str,
        y_train : torch.Tensor,
        n_classes : int,
        device : torch.device,
        alpha : float = None,
        focal_loss_gamma : float = None
    ) -> nn.Module:

    """Costruisce la funzione di loss in base alla selezione dell'utente.

    Args:
        selected_loss (str): Tipo di loss selezionata dall'utente.
        y_train (torch.Tensor): Tensor contenente le etichette di addestramento.
        n_classes (int): Numero totale di classi.
        device (torch.device): Il device su cui eseguire i calcoli.
        alpha (float, opzionale): Parametro alpha per la Focal Loss.
        focal_loss_gamma (float, opzionale): Parametro gamma per la Focal Loss.

    Raises:
        ValueError: Se il tipo di loss selezionato non è supportato.
        ValueError: Se il tipo di alpha per la Focal Loss non è supportato.

    Returns:
        nn.Module: La funzione di loss costruita.
    """

    loss_dtype = torch.float if device.type == 'mps' else torch.double

    if selected_loss == "CrossEntropy":
        criterion = nn.CrossEntropyLoss()

    elif selected_loss == "WeightedCrossEntropy":
        class_weights = compute_class_weights(
            y_train,
            n_classes
        ).to(device).to(loss_dtype)

        criterion = nn.CrossEntropyLoss(weight=class_weights)

    elif selected_loss == "Focal":

        if alpha == "class_weights":
            alpha = compute_class_weights(y_train, n_classes)
        elif alpha == "uniform":
            alpha = torch.ones(n_classes)
        elif alpha == "custom":
            alpha = torch.tensor(alpha)
        else:
            raise ValueError(
                f"Tipo di alpha non supportato: {alpha}. "
                "Usare: 'class_weights', 'uniform', oppure 'custom'."
            )

        alpha = alpha.to(device).to(loss_dtype)

        criterion = torch.hub.load(
            'adeelh/pytorch-multi-class-focal-loss',
            model='focal_loss',
            alpha=alpha,
            gamma=focal_loss_gamma,
            reduction='mean',
            device=device,
            dtype=loss_dtype,
            force_reload=False
        )

    else:
        raise ValueError(
            f"Tipo di loss non supportato: {selected_loss}. "
            "Usare: 'CrossEntropy', 'WeightedCrossEntropy' oppure 'Focal'."
        )

    return criterion

    # end

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
