"""
	constants.py \n
	by Mario Gabriele Carofano

	Questo modulo contiene le costanti utilizzate nel progetto, utili per configurare tutti gli aspetti del progetto e la riproducibilità degli esperimenti.
	
	Sono organizzate principalmente in due sezioni:
	- Costanti strutturali
	Sono considerate private (iniziano con "__") e non rilevanti per la riproducibilità degli esperimenti.

	- Parametri di sessione
	Contengono le impostazioni specifiche per una sessione di esecuzione del progetto. Sono presenti funzioni di salvataggio e caricamento in file JSON, in modo da poter riprodurre gli esperimenti e confrontare i risultati.

	Inoltre, questo modulo contiene anche altre funzioni di utilità per la gestione della configurazione e della stampa di messaggi per il debug.
"""

# TODO: trasformare il modulo delle costanti in una classe Singleton per una gestione centralizzata della configurazione.

#   ####################################################################    #
#   Librerie

import types
from typing import Any
import sys
import platform

#   ####################################################################    #
#   Funzioni di utilità

def _json_default(obj: Any) -> Any:
	"""Serializzatore di fallback per json.dump, usato per convertire tipi non nativamente serializzabili in JSON (tuple annidate, set, oggetti torch.device, classi, ecc.).
	"""

	# Caso speciale: 
	# torch.device, np.dtype e oggetti simili: si usa la rappresentazione a stringa.
	if hasattr(obj, "__class__") and obj.__class__.__name__ in ("device", "dtype"):
		return str(obj)

	if isinstance(obj, set):
		return sorted(obj)

	if isinstance(obj, (tuple,)):
		return list(obj)

	# Per le classi, si restituisce il nome della classe.
	# In Python, ogni classe è un'istanza di 'type'.
	if isinstance(obj, type):
		return obj.__name__

	# Ultima risorsa: rappresentazione testuale, per non far fallire il salvataggio.
	return str(obj)

	# end

def build_session_config() -> dict[str, Any]:
	"""Costruisce il dizionario completo di configurazione di sessione,
	combinando in un unico livello le costanti statiche di constants.py
	(chiavi MAIUSCOLE) e i metadati di ambiente (chiavi minuscole).

	Returns:
		dict: Configurazione di sessione unificata.
	"""

	config = {}

	#   ################################################################    #
	#   Costanti statiche da constants.py (chiavi MAIUSCOLE)

	import constants as constants_module
	for name, value in vars(constants_module).items():

		# Le costanti con nomi non maiuscoli non vengono incluse
		# nella configurazione finale. Questo permette di scartare
		# variabili, funzioni e moduli che non sono considerati costanti.
		if not name.isupper():
			continue

		# Le costanti con il prefisso "__" sono considerate private
		# e non vengono incluse nella configurazione finale.
		if name.startswith("__"):
			continue

		# Le costanti definite come moduli o funzioni
		# non vengono incluse nella configurazione finale.
		if isinstance(value, (
			types.ModuleType,
			types.FunctionType,
			types.BuiltinFunctionType,
			type
		)):
			continue

		config[name] = value

		# end for

	#   ################################################################    #
	#   Metadati di ambiente (chiavi minuscole)

	config["python_version"] = sys.version
	config["platform"] = platform.platform()
	config["processor"] = platform.processor()

	for lib_name in ["torch", "numpy", "pandas", "sklearn", "pennylane"]:
		try:
			module = __import__(lib_name)
			config[f"{lib_name}_version"] = getattr(module, "__version__", "unknown")
		except ImportError:
			config[f"{lib_name}_version"] = None

		# end for

	return config

	# end

def get_value_from_config(
		source: dict,
		key: str,
		default = None,
	):
	"""Recupera un valore dalla sorgente specificata.
	Se il valore non è presente, viene restituito il valore di default.

	Args:
		source (dict): Dizionario da cui recuperare il valore.
		key (str): La chiave da cercare nel dizionario di configurazione.
		default (Any, optional): Valore di default se la chiave
		non è presente. Defaults to None.

	Returns:
		Any: Il valore corrispondente alla chiave,
		o il valore di default se la chiave non è presente.
	"""

	if source is None:
		raise ValueError("Sorgente di configurazione mancante.")

	value = source.get(key, default)
	if value is None:
		if default is None:
			raise ValueError(f"Chiave '{key}' mancante.")
		
		debug_print(f"Chiave '{key}' mancante. Utilizzo '{default}' come default.")
		return default

	debug_print(f"Costante '{key}' caricata con valore: {value}")
	return value

	# end

def debug_print(message: str) -> None:
	"""Stampa un messaggio di debug se la modalità di debug è abilitata.

	Args:
		message (str): Messaggio da stampare.
	"""

	if __DEBUG:
		print(f"[DEBUG] {message}")

	# end

#   ####################################################################    #
#   Costanti strutturali
#   Controllano il flusso di esecuzione del codice.
#   Sono escluse dalla configurazione salvata per gli esperimenti.

__DEBUG = False
""" Whether to enable debug mode. """

__DEBUG_CYCLE = False
""" Whether to enable debug mode inside each cycle of the execution loop in mirage_pickle_converter(). """

__FLOWS_TO_INSPECT = None
""" List of flows to inspect during debugging in mirage_pickle_converter(). """

__USE_PRECOMPUTED_DATASET = True
""" Whether to load the FlowPic dataset from a precomputed .npz file. """

__USE_CONFIG_FILE = False
""" Whether to use a configuration file to load the constants. """

__CONFIG_ID = "2026-10-16_12-25"
""" Identifier for the configuration JSON file to load. """

__EXEC_MODE_TRAIN = True
""" Set this flag to choose between training a new model
or running validation on an already saved model. \n
- True  : Execute the training loop.
- False : Skip training and run validation only on a pre-saved model. \n
This allows running "Run All" in the notebook and automatically selecting
the appropriate branch without manually skipping cells. """

__SAVE_OUTPUT = True
""" Whether to save the output of the notebook. """

#   ####################################################################    #
#   Parametri di sessione
#   Definiscono la configurazione di una specifica esecuzione sperimentale.
#   Vengono salvati in un file JSON e ricaricati per consentire la
#   riproducibilità degli esperimenti e il confronto tra esecuzioni.

RANDOM_SEED = 2025
""" Random seed for reproducibility. """

PREPROCESSING_STRATEGY = "MinMax"
""" Strategy for preprocessing the dataset. Default is 'MinMax'. """

#   ---------------------------------------------------------------------   #
#   Dataset configuration

DATA_PATH = "Dataset/mirage/2019"
""" Path to the dataset file. """

DATASET_NAME = "mirage2019_LOPEZ_lopez_lopez_100P_4F_APP_xST_PAD_metadata.pickle"
""" Name of the dataset file. """

USE_FLOWPIC_DATASET = False
""" Whether to use the FlowPic dataset or the original Mirage dataset. If True, generate the FlowPic dataset from raw data using the `traffic_converter` module. """

N_PACKETS = 100
""" Number of packets to consider in each flow. """

N_FEATURES = 4
""" Number of features for each packet in the dataset. """

#   ---------------------------------------------------------------------   #
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

#   ---------------------------------------------------------------------   #
#   Dataset split configuration

TRAIN_SIZE = 0.8
""" Proportion of the dataset to be used for training. """

VAL_SIZE = 0.2
""" Proportion of the dataset to be used for validation. """

TEST_FOLDS = 5
""" Number of folds for cross-validation. """

SUB_SAMPLE_RATIO = 0.05
""" Ratio of the dataset to be used for sub-sampling. For example, 0.5 means using 50% of the dataset. Default should be 1.0, meaning the entire dataset is used. """

BATCH_SIZE = 50
""" Size of the mini-batches used during training. """

#   ---------------------------------------------------------------------   #
#   Training configuration

EPOCHS = 3
""" Number of training epochs. """

LEARNING_RATE = 1e-3
""" Learning rate for the optimizer. """

EARLY_STOPPING = True
""" Whether to use early stopping during training. """

PATIENCE = max(2, EPOCHS // 10) if EPOCHS <= 100 else 10
""" Number of epochs to wait for improvement before stopping training. """

#   ---------------------------------------------------------------------   #
#   Optimizer configuration

SELECTED_OPTIMIZER = "AdamW"
""" Selected optimizer for training. Default is "AdamW". """

WEIGHT_DECAY = 1e-4
""" Weight decay (L2 penalty) for the optimizer. """

EPSILON = 1e-2
""" Epsilon value for the optimizer AdamW. """

MOMENTUM = 0.9
""" Momentum for the optimizer SGD. """

#   ---------------------------------------------------------------------   #
#   Scheduler configuration

SELECTED_SCHEDULER = "OneCycleSched"
""" Selected scheduler for the learning rate. Default is "OneCycleSched". """

MAX_LR = LEARNING_RATE*3
""" Maximum learning rate for the OneCycleLR scheduler. """

START_FACTOR = 0.1
""" Starting factor for the learning rate scheduler. """

STEP_SIZE = 5
""" Step size for the StepLR scheduler. """

STEPLR_GAMMA = 0.1
""" Multiplicative factor of learning rate decay for the StepLR scheduler. """

#   ---------------------------------------------------------------------   #
#   Loss configuration

SELECTED_LOSS = "WeightedCrossEntropy"
""" Selected loss function for training. Default is "WeightedCrossEntropy". """

ALPHA = "class_weights"
""" Alpha parameter for the Focal Loss.
Can be "class_weights", "uniform" or "custom". """

FOCAL_LOSS_GAMMA = 2.0
""" Gamma parameter for the Focal Loss, typically between 1 and 5. """

#   ---------------------------------------------------------------------   #
#   Network configuration

SELECTED_MODEL = "hybrid"
""" Selected model for the network. Default is "resnet". """

SIMULATOR = "default.qubit"
""" Name of the quantum simulator to use with PennyLane. """

N_QUBITS = 5
""" Number of qubits in the quantum circuit. """

N_SHOTS = None
""" Number of shots for the quantum circuit execution. """

N_LAYERS = 18 if SELECTED_MODEL == "resnet" else 3
""" Number of layers for the selected model.
- For ResNet, possible values are 16, 18, 34. Default is 18.
- For Quantum models, this represents the depth of the StronglyEntanglingLayers ansatz. Default is 3. """

#   ---------------------------------------------------------------------   #
