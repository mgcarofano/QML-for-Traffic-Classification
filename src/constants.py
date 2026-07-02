"""
    constants.py
    by Mario Gabriele Carofano

    Questo modulo contiene le costanti utilizzate nel progetto.
"""

#   ####################################################################    #
#   Dataset configuration

DATA_PATH = "../dataset/mirage/2019/mirage2019_LOPEZ_lopez_lopez_100P_4F_APP_xST_PAD_metadata.pickle"
""" Path to the dataset file. """

DATASET_PACKETS = 100
""" Number of packets available at most in each flow. """

PADDING_VALUE = -1
""" This value is used in the dataset to denote padding packets. """

N_PACKETS = 100
""" Number of packets to consider in each flow. """

N_FEATURES = 4
""" Number of features for each packet. """

FEATURES_LIST = ['DIR', 'PL', 'TCPWIN', 'IAT']
""" List of names of available features. """

#   ####################################################################    #
#   Dataset split configuration

TRAIN_SIZE = 0.71
""" Proportion of the dataset to be used for training. """

VAL_SIZE = 0.14
""" Proportion of the dataset to be used for validation. """

TEST_SIZE = 0.15
""" Proportion of the dataset to be used for testing. """

BATCH_SIZE = 32
""" Size of the mini-batches used during training. """

NEW_TRAIN_SIZE = 30000
""" Limit on the number of training samples to use, to reduce training time.
	If the training set is larger than this, it will be randomly sampled down to this size. """

#	Model selection
#   ####################################################################    #

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
}
""" A dictionary mapping model names to their corresponding module and class names. """

#   ####################################################################    #
#   Training configuration

LOSS_REGISTRY = {
	"CrossEntropy" : {
		"params" : {}
	},
	"WeightedCrossEntropy" : {
		"params" : {}
	},
	"Focal" : {
		"params" : {
			# ALPHA : può essere "class_weights", "uniform" o "custom".
			"ALPHA" : "class_weights",

			# GAMMA : parametro di focalizzazione, tipicamente tra 1 e 5.
			"GAMMA" : 2.0
		}
	}
}

RANDOM_SEED = 2025
""" Random seed for reproducibility. """

EPOCHS = 10
""" Number of training epochs. """

LEARNING_RATE = 1e-3
""" Learning rate for the optimizer. """

EARLY_STOPPING = True
""" Whether to use early stopping during training. """

PATIENCE = 10
""" Number of epochs to wait for improvement before stopping training. """

#   ####################################################################    #
#   Network configuration

N_QUBITS = 5
""" Number of qubits in the quantum circuit. """

N_LAYERS = 3
""" Number of layers in the quantum circuit. """