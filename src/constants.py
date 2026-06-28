"""
    constants.py
    by Mario Gabriele Carofano

    Questo modulo contiene le costanti utilizzate nel progetto.
"""

#   ####################################################################    #
#   Dataset configuration

DATA_PATH = "../dataset/mirage/2024/act/mirage2024_act_LOPEZ_lopez_lopez_36P_4F_APP_xST_PAD_cf1a9527.pickle"
""" Path to the dataset file. """

DATASET_PACKETS = 36
""" Number of packets available at most in each flow. """

PADDING_VALUE = -1
""" This value is used in the dataset to denote padding packets. """

N_PACKETS = 10
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

BATCH_SIZE = 50
""" Size of the mini-batches used during training. """

NEW_TRAIN_SIZE = 30000
""" Limit on the number of training samples to use, to reduce training time.
	If the training set is larger than this, it will be randomly sampled down to this size. """

#   Preprocessing configuration
#   ####################################################################    #

PREPROCESSING_STRATEGY = 4
""" Specifies the preprocessing strategy to apply to the dataset. \n
	Options:
	1. Masking del padding + Log1p
	2. Min-Max Scaling standard
	3. Min-Max Scaling + Fusione DIR/PL
	4. Masking del padding + Log1p + Fusione DIR/PL
"""

#   ####################################################################    #
#   Training configuration

RANDOM_SEED = 2025
""" Random seed for reproducibility. """

EPOCHS = 100
""" Number of training epochs. """

LEARNING_RATE = 1e-3
""" Learning rate for the optimizer. """

EARLY_STOPPING = False
""" Whether to use early stopping during training. """

PATIENCE = 15
""" Number of epochs to wait for improvement before stopping training. """

#   ####################################################################    #
#   Network configuration

N_QUBITS = 5
""" Number of qubits in the quantum circuit. """

N_LAYERS = 3
""" Number of layers in the quantum circuit. """