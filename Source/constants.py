"""
    constants.py \n
    by Mario Gabriele Carofano

    Questo modulo contiene le costanti utilizzate nel progetto, utili per
    configurare tutti gli aspetti del progetto e la riproducibilità degli
    esperimenti.
    
    Sono organizzate principalmente in due sezioni:
    - Costanti strutturali
    Sono considerate private (iniziano con "__") e non rilevanti per la
    riproducibilità degli esperimenti.

    - Parametri di sessione
    Contengono le impostazioni specifiche per una sessione di esecuzione del
    progetto. Sono presenti funzioni di salvataggio e caricamento in file JSON,
    in modo da poter riprodurre gli esperimenti e confrontare i risultati.
"""

#   ####################################################################    #
#   Librerie

import sys
import types
from typing import Any
import platform

#   ####################################################################    #
#   Funzioni di utilità

def _json_default():
	pass

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
		default = None
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

	value = source.get(key, default)
	if value is None:
		if default is None:
			raise ValueError(f"Chiave '{key}' mancante.")
		
		print(f"Chiave '{key}' mancante. Utilizzo '{default}' come default.")
		return default

	return value

	# end

#   ####################################################################    #
#   Costanti strutturali

__DEBUG = False
""" Whether to enable debug mode. """

__USE_PRECOMPUTED_DATASET = True
""" Whether to load the FlowPic dataset from a precomputed .npz file. """

__USE_CONFIG_FILE = False
""" Whether to use a configuration file to load the constants. """

__CONFIG_ID = "2026-09-02_14-48"
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

RANDOM_SEED = 2025
""" Random seed for reproducibility. """

TEST_FOLDS = 10
""" Number of folds for cross-validation. """
