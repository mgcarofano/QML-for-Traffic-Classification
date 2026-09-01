TPS = 60
""" Duration of each session in seconds (TimePerSession) """

DELTA_T = 15
""" Difference in seconds between the start of two consecutive sessions (DeltaT) """

PADDING_VALUE = -1
""" This value is used in the dataset to denote padding packets. """

DATASET_PACKETS = 100
""" Number of packets available at most in each flow. """

#   ####################################################################    #

USE_FLOWPIC_DATASET = (True, False)
""" A pair of boolean values indicating whether to use the FlowPic dataset
and whether to use a precomputed version of it. \n
- (False, _) : Load the dataset from raw data using the .pickle file.
- (True, False) : Generate the FlowPic dataset from raw data using the `traffic_converter` module.
- (True, True) : Load the FlowPic dataset from a precomputed .npz file. """

#   ####################################################################    #

MODEL_TIMESTAMP_ID = "2026-08-25_20-32"
""" Timestamp identifier for the model, used for saving and loading. """

MODEL_NAME = "16E_AdamW_OneCycleSched_WeightedCrossEntropy_ResNet"
""" Name of the model, used for saving and loading. \n
The name format is *"{epochs}E\_{loss}\_{preprocessing}\_{model}"*. """
