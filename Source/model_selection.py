"""
    model_selection.py \n
    di Mario Gabriele Carofano

    Questo modulo contiene funzioni per la selezione dinamica del modello
    e il controllo della compatibilità dei pesi salvati.
"""

#   ####################################################################    #
#   Librerie

import torch.nn as nn

#   ####################################################################    #
#   Funzioni di utilità

def compute_model_name(
        model_class : str,
        n_qubits : int = None,
        n_layers : int = None,
        n_packets : int = None,
        n_features : int = None,
        num_classes : int = None
    ) -> str:
    """
    Restituisce una stringa con il nome del modello che riassume i parametri principali,
    se disponibili, nel formato 'class_Qn_Ln_Pn_Fn_Cn'.

    Args:
        model_class (str): Nome della classe del modello.
        n_qubits (int, optional): Numero di qubit nel circuito quantistico. Default è None.
        n_layers (int, optional): Numero di layer nel circuito quantistico. Default è None.
        n_packets (int, optional): Numero di pacchetti nell'input. Default è None.
        n_features (int, optional): Numero di feature per pacchetto. Default è None.
        num_classes (int, optional): Numero di classi per la classificazione. Default è None.

    Returns:
        str: Nome del modello.
    """

    ret = f"{model_class}"

    if n_qubits is not None:
        ret += f"_Q{n_qubits}"

    if n_layers is not None:
        ret += f"_L{n_layers}"

    if n_packets is not None:
        ret += f"_P{n_packets}"

    if n_features is not None:
        ret += f"_F{n_features}"

    if num_classes is not None:
        ret += f"_C{num_classes}"

    return ret

    # end

def get_output_dim(dimension, kernels, strides, dilatation=1, padding='same', return_paddings=False):

    def get_padding(kernel):
        pad = kernel - 1
        if padding == 'same':
            if kernel % 2:
                return pad // 2, pad // 2
            else:
                return pad // 2, pad // 2 + 1
        return 0, 0

        # end

    out_dim = dimension
    paddings = []
    if padding == 'same':
        for kernel, stride in zip(kernels, strides):
            paddings.append(get_padding(kernel))
            out_dim = (out_dim + stride - 1) // stride
    else:
        for kernel, stride in zip(kernels, strides):
            paddings.append(get_padding(kernel))
            out_dim = (out_dim - kernel + stride) // stride

    if return_paddings:
        return out_dim, paddings
    return out_dim

    # end

def weight_init(m: nn.Module) -> None:
    """Fills-in weights and biases for convolutional and linear layers. Uses Kaiming uniform initialization for weights (suitable for ReLU/non-linearities) and sets biases to zero. This function is meant to be used with model.apply(), which will call it on every sub-module.

    Args:
        m (nn.Module): a module of the model.
    """	

    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):

        # Good idea would be to use kaiming initialization of scaled orthonormal initialization.
        nn.init.kaiming_uniform_(
            m.weight,
            a=0,
            mode='fan_in',
            nonlinearity='leaky_relu'
        )

        # Check if m.bias is enabled.
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    
    return

    # end

def check_model_compatibility(model, weights):
    """Controlla la compatibilità tra il modello corrente e i pesi salvati.

    Args:
        model (torch.nn.Module): Modello corrente.
        weights (dict): Pesi salvati del modello.

    Raises:
        RuntimeError: Se le shape dei pesi salvati non corrispondono a quelle del modello corrente.
    """

    curr_weights_shape = [
        v.shape
        for k, v in model.state_dict().items()
        if "classifier" in k and k.endswith(".weight") and v.dim() == 2
    ]

    saved_weights_shape = [
        v.shape
        for k, v in weights.items()
        if "classifier" in k and k.endswith(".weight") and v.dim() == 2
    ]

    mismatch_index = next(
        (
            i
            for i, (current_shape, saved_shape) in enumerate(
                zip(curr_weights_shape, saved_weights_shape)
            )
            if current_shape != saved_shape
        ),
        None
    )

    if mismatch_index is not None:
        current_shape = curr_weights_shape[mismatch_index]
        saved_shape = saved_weights_shape[mismatch_index]
        model_name = type(model).__name__

        raise RuntimeError(
            f"Errore durante il caricamento del modello: {model_name}\n\n"
            f"Shape corrente diversa da quella salvata, in posizione ({mismatch_index}).\n"
            f"Il checkpoint è stato salvato con un numero di classi ({saved_shape}) diverso "
            f"rispetto al modello corrente ({current_shape}).\n"
            "Verifica che il dataset e il parametro num_classes usati per il "
            "training coincidano con quelli attuali."
        )

        # end if
    # end

#   ####################################################################    #
#   Selezione dinamica del modello

def select_model(
        model_name: str,
        n_features: int, n_packets: int, num_classes: int,
        n_layers: int, n_qubits: int, n_shots: int, dev_name: str,
        random_seed: int
    ):
    """Seleziona e restituisce il modello specificato.

    Args:
        model_name (str): Nome del modello da selezionare.
        n_features (int): Numero di feature per pacchetto.
        n_packets (int): Numero di pacchetti nell'input.
        num_classes (int): Numero di classi per la classificazione.
        n_layers (int): Numero di layer nel circuito quantistico (se applicabile).
        n_qubits (int): Numero di qubit nel circuito quantistico (se applicabile).
        n_shots (int): Numero di ripetizioni per il circuito quantistico (se applicabile).
        dev_name (str): Nome del dispositivo quantistico (se applicabile).
        random_seed (int): Seed per la generazione casuale (se applicabile).

    Raises:
        NotImplementedError: Se il modello specificato non è implementato.

    Returns:
        tuple: Una tupla contenente il modello selezionato e un flag booleano che indica se è quantistico.
    """

    #   -----------------------------------------------------------------   #
    #   Modelli specifici per istogrammi FlowPic.

    if model_name == "flowpic_classic":
        from Models.flowpic_models import FlowPicCNN as FModel
        is_quantum = False
        model = FModel(n_features, n_packets, num_classes)

    elif model_name == "resnet":
        from Models.flowpic_models import ResNetModel as FModel
        is_quantum = False
        model = FModel(
            n_features, n_packets, num_classes,
            n_layers
        )

    #   -----------------------------------------------------------------   #
    #   Modelli quantistici.

    elif model_name == "ampe_fixed_probs":
        from Models.quantum_models import AmpeDenseModel as QModel
        is_quantum = True
        model = QModel(
            n_features, n_packets, num_classes,
            n_layers, n_qubits, n_shots,
            random_seed, dev_name,
        )

    elif model_name == "ange_probs":
        from Models.quantum_models import AngeDenseModel as QModel
        is_quantum = True
        model = QModel(
            n_features, n_packets, num_classes,
            n_layers, n_qubits, n_shots,
            random_seed, dev_name,
        )

    elif model_name == "hybrid_ampe_fixed_probs":
        from Models.quantum_models import AmpeCNNLSTMModel as QModel
        is_quantum = True
        model = QModel(
            n_features, n_packets, num_classes,
            n_layers, n_qubits, n_shots,
            random_seed, dev_name,
        )

    elif model_name == "hybrid_ange_probs":
        from Models.quantum_models import AngeCNNLSTMModel as QModel
        is_quantum = True
        model = QModel(
            n_features, n_packets, num_classes,
            n_layers, n_qubits, n_shots,
            random_seed, dev_name,
        )

    #   -----------------------------------------------------------------   #
    #   Modelli classici.

    elif model_name == "traffic":
        from Models.classic_models import TrafficCNN as CModel
        is_quantum = False
        model = CModel(n_features, n_packets, num_classes)

    elif model_name == "baseline":
        from Models.classic_models import CNNLSTMModel as CModel
        is_quantum = False
        # model = CModel(n_features, n_packets, num_classes)
        model = CModel(n_features, n_packets, num_classes, n_layers)

    #   -----------------------------------------------------------------   #
    #   Modello non riconosciuto.

    else:
        raise NotImplementedError(f"Modello non riconosciuto: {model_name}")
    
    #   -----------------------------------------------------------------   #
    #   Ritorno del modello selezionato e del flag quantistico.

    return model, is_quantum

    # end
