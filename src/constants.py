"""
    constants.py
    by Mario Gabriele Carofano

    Questo modulo contiene le costanti utilizzate nel progetto.
"""

#   ####################################################################    #
#   Miscellaneous

RANDOM_SEED = 2025
""" Random seed for reproducibility. """

EXEC_MODE_TRAIN = True
""" Set this flag to choose between training a new model
or running validation on an already saved model. \n
- True  : Execute the training loop.
- False : Skip training and run validation only on a pre-saved model. \n
This allows running "Run All" in the notebook and automatically selecting
the appropriate branch without manually skipping cells. """

USE_FLOWPIC_DATASET = (True, True)
""" A pair of boolean values indicating whether to use the FlowPic dataset
and whether to use a precomputed version of it. \n
- (False, _) : Load the dataset from raw data using the .pickle file.
- (True, False) : Generate the FlowPic dataset from raw data using the `traffic_converter` module.
- (True, True) : Load the FlowPic dataset from a precomputed .npz file. """

OUTPUT_DIR = "../results"
""" Directory where the model and training history will be saved. """

SAVE_OUTPUT = True
""" Whether to save the output of the notebook. """

#   ####################################################################    #
#   Traffic converter configuration

TPS = 60
""" Duration of each session in seconds (TimePerSession) """

DELTA_T = 15
""" Difference in seconds between the start of two consecutive sessions (DeltaT) """

MIN_TPS = 50
""" Minimum duration of a session in seconds. Sessions shorter than this will be discarded. """

MIN_PACKETS = 3
""" Minimum number of packets in a session. Sessions with fewer packets will be discarded. """

MIN_DIM = 10000
""" Minimum flow dimension (payload) in bytes. Flows smaller than this will be discarded. """

MTU = 1500
""" Maximum Transmission Unit (Ethernet). """

BIN_SIZE = 10
""" Dimension in byte of each bin on the Y-axis of the 2D histogram. This means
    that the Y-axis will be divided into 150 bins, each representing a range of 10 bytes. """

#   ####################################################################    #
#   Dataset configuration

DATA_PATH = "../dataset/mirage/2019"
""" Path to the dataset file. """

DATASET_NAME = "mirage2019_LOPEZ_lopez_lopez_100P_4F_APP_xST_PAD_metadata.pickle"
""" Name of the dataset file. """

DATASET_PACKETS = 100
""" Number of packets available at most in each flow. """

PADDING_VALUE = -1
""" This value is used in the dataset to denote padding packets. """

N_PACKETS = 100
""" Number of packets to consider in each flow. """

FEATURES_LIST = ['DIR', 'PL', 'TCPWIN', 'IAT']
""" List of names of available features. """

N_FEATURES = len(FEATURES_LIST)
""" Number of features for each packet. """

#   ####################################################################    #
#   Dataset split configuration

TRAIN_SIZE = 0.71
""" Proportion of the dataset to be used for training. """

VAL_SIZE = 0.14
""" Proportion of the dataset to be used for validation. """

TEST_SIZE = 0.15
""" Proportion of the dataset to be used for testing. """

BATCH_SIZE = 64
""" Size of the mini-batches used during training. """

USE_NEW_SIZE = True
""" Whether to limit the number of training samples to a new size. """

NEW_DATASET_SIZE = 30000
""" Limit on the number of samples to use, to reduce training time and memory usage.
	If the dataset is larger than this, it will be randomly sampled down to this size. """

#   ####################################################################    #
#	Model selection

MODEL_REGISTRY = {
    # nn_models
    "AmplitudeEmbedding": ("../models.nn_models", "AmpHybridModel"),
    "AngleEmbedding": ("../models.nn_models", "AngleHybridModel"),
    "RingEmbedding": ("../models.nn_models", "RingHybridModel"),
    "WaterfallEmbedding": ("../models.nn_models", "WaterfallHybridModel"),
    "TrafficCNN": ("../models.nn_models", "TrafficCNN"),

    # complex_hybrid_models
    "AmpCnn": ("../models.complex_hybrid_models", "AmpCnn"),
    "ClassicalTwin": ("../models.complex_hybrid_models", "ClassicalTwinModel"),
    "ClassicalLight": ("../models.complex_hybrid_models", "ClassicalLight"),
    "CnnAmpCnn": ("../models.complex_hybrid_models", "CnnAmpCnn"),
    "Dense": ("../models.complex_hybrid_models", "Dense"),

    # flowpic_models
    "FlowPicCNN": ("../models.flowpic_models", "FlowPicCNN"),
    "ResNet": ("../models.flowpic_models", "ResNet"),
}
""" A dictionary mapping model names to their corresponding module and class names. """

MODEL_TIMESTAMP_ID = "2026-08-24_18-05"
""" Timestamp identifier for the model, used for saving and loading. """

MODEL_NAME = "15E_AdamW_OneCycleSched_WeightedCrossEntropy_ResNet"
""" Name of the model, used for saving and loading. \n
The name format is *"{epochs}E\_{loss}\_{preprocessing}\_{model}"*. """

#   ####################################################################    #
#   Training configuration

EPOCHS = 30
""" Number of training epochs. """

LEARNING_RATE = 1e-2
""" Learning rate for the optimizer. """

EARLY_STOPPING = True
""" Whether to use early stopping during training. """

PATIENCE = EPOCHS // 10 if EPOCHS <= 100 else 10
""" Number of epochs to wait for improvement before stopping training. """

#   ####################################################################    #
#   Optimizer configuration

WEIGHT_DECAY = 1e-4
""" Weight decay (L2 penalty) for the optimizer. """

EPSILON = 1e-2
""" Epsilon value for the optimizer AdamW. """

MOMENTUM = 0.9
""" Momentum for the optimizer SGD. """

#   ####################################################################    #
#   Scheduler configuration

MAX_LR = LEARNING_RATE*3
""" Maximum learning rate for the OneCycleLR scheduler. """

START_FACTOR = 0.1
""" Starting factor for the learning rate scheduler. """

STEP_SIZE = 5
""" Step size for the StepLR scheduler. """

STEPLR_GAMMA = 0.1
""" Multiplicative factor of learning rate decay for the StepLR scheduler. """

#   ####################################################################    #
#   Loss configuration

ALPHA = "class_weights"
""" Alpha parameter for the Focal Loss.
Can be "class_weights", "uniform" or "custom". """

FOCAL_LOSS_GAMMA = 2.0
""" Gamma parameter for the Focal Loss, typically between 1 and 5. """

#   ####################################################################    #
#   Network configuration

N_QUBITS = 5
""" Number of qubits in the quantum circuit. """

N_LAYERS = 3
""" Number of layers in the quantum circuit. """
