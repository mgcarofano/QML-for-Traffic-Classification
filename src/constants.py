"""
    constants.py
    by Mario Gabriele Carofano

    Questo modulo contiene le costanti utilizzate nel progetto. Servono a
    configurare il comportamento del notebook e dei moduli associati, come
    il traffic converter e il modello di rete neurale.
    
    Le costanti sono organizzate in diverse sezioni, con commenti che
    spiegano il loro significato e il loro utilizzo.

    Nel notebook principale, sono presenti delle funzioni di salvataggio
    e caricamento delle costanti in un file JSON, in modo da poter riprodurre
    gli esperimenti e confrontare i risultati. Le uniche costanti che non
    vengono salvate sono quelle con il prefisso "__", che sono considerate
    private e non rilevanti per la riproducibilità degli esperimenti.
"""

#   ####################################################################    #
#   Costanti strutturali
#   Definiscono la struttura del codice stesso,

__DEBUG = False
""" Whether to enable debug mode. """

__USE_PRECOMPUTED_DATASET = True
""" Whether to load the FlowPic dataset from a precomputed .npz file. """

__USE_CONFIG_FILE = True
""" Whether to use a configuration file to load the constants. """

__CONFIG_ID = "2026-09-01_20-03"
""" Identifier for the configuration JSON file to load. """

__EXEC_MODE_TRAIN = True
""" Set this flag to choose between training a new model
or running validation on an already saved model. \n
- True  : Execute the training loop.
- False : Skip training and run validation only on a pre-saved model. \n
This allows running "Run All" in the notebook and automatically selecting
the appropriate branch without manually skipping cells. """

__SELECTED_FOLD = 0
""" Index of the fold to be used. """

__SAVE_OUTPUT = True
""" Whether to save the output of the notebook. """

#   ####################################################################    #
#   Parametri di sessione
#   Scelte sperimentali che influenzano il risultato di una run specifica.

RANDOM_SEED = 2025
""" Random seed for reproducibility. """

#   ---    #
#   Traffic converter configuration

MTU = 1500
""" Maximum Transmission Unit (Ethernet). """

BIN_SIZE = 10
""" Dimension in byte of each bin on the Y-axis of the 2D histogram.
This means that the Y-axis will be divided into 150 bins,
each representing a range of 10 bytes. """

TRAFFIC_FILTERS = {
    "min_tps": 50,
    # "min_packets": 3,
    # "min_dim": 10000,
}
""" A dictionary containing the filtering criteria for traffic flows. 
- 'min_tps' : Minimum duration of a session in seconds. Sessions shorter than this will be discarded.
- 'min_packets' : Minimum number of packets in a session. Sessions with fewer packets will be discarded.
- 'min_dim' : Minimum flow dimension (payload) in bytes. Flows smaller than this will be discarded. """

#   ---    #
#   Dataset configuration

DATA_PATH = "../dataset/mirage/2019"
""" Path to the dataset file. """

DATASET_NAME = "mirage2019_LOPEZ_lopez_lopez_100P_4F_APP_xST_PAD_metadata.pickle"
""" Name of the dataset file. """

USE_FLOWPIC_DATASET = True
""" Whether to use the FlowPic dataset or the original Mirage dataset.
If True, generate the FlowPic dataset from raw data
using the `traffic_converter` module. """

N_PACKETS = 100
""" Number of packets to consider in each flow. """

#   ---    #
#   Dataset split configuration

TRAIN_SIZE = 0.8
""" Proportion of the dataset to be used for training. """

VAL_SIZE = 0.2
""" Proportion of the dataset to be used for validation. """

TEST_FOLDS = 10
""" Number of folds for cross-validation. """

BATCH_SIZE = 128
""" Size of the mini-batches used during training. """

USE_NEW_SIZE = True
""" Whether to limit the number of training samples to a new size. """

NEW_DATASET_SIZE = 1000
""" Limit on the number of samples to use, to reduce training time and memory usage.
	If the dataset is larger than this, it will be randomly sampled down to this size. """

#   ---    #
#   Training configuration

EPOCHS = 3
""" Number of training epochs. """

LEARNING_RATE = 1e-3
""" Learning rate for the optimizer. """

EARLY_STOPPING = True
""" Whether to use early stopping during training. """

PATIENCE = max(2, EPOCHS // 10) if EPOCHS <= 100 else 10
""" Number of epochs to wait for improvement before stopping training. """

#   ---    #
#   Optimizer configuration

WEIGHT_DECAY = 1e-4
""" Weight decay (L2 penalty) for the optimizer. """

EPSILON = 1e-2
""" Epsilon value for the optimizer AdamW. """

MOMENTUM = 0.9
""" Momentum for the optimizer SGD. """

#   ---    #
#   Scheduler configuration

MAX_LR = LEARNING_RATE*3
""" Maximum learning rate for the OneCycleLR scheduler. """

START_FACTOR = 0.1
""" Starting factor for the learning rate scheduler. """

STEP_SIZE = 5
""" Step size for the StepLR scheduler. """

STEPLR_GAMMA = 0.1
""" Multiplicative factor of learning rate decay for the StepLR scheduler. """

#   ---    #
#   Loss configuration

ALPHA = "class_weights"
""" Alpha parameter for the Focal Loss.
Can be "class_weights", "uniform" or "custom". """

FOCAL_LOSS_GAMMA = 2.0
""" Gamma parameter for the Focal Loss, typically between 1 and 5. """

#   ---    #
#   Network configuration

SIMULATOR = "default.qubit"
""" Name of the quantum simulator to use with PennyLane. """

N_QUBITS = 5
""" Number of qubits in the quantum circuit. """

N_SHOTS = 20
""" Number of shots for the quantum circuit execution. """

N_LAYERS_QUANTUM = 3
""" Profondità dell'ansatz StronglyEntanglingLayers per i modelli ibridi quantistici. """

N_LAYERS_RESNET = 16
""" Variante di ResNet da istanziare per ResNetModel.
Valori possibili: 16, 18, 34. """
