"""
	Main.py \n
	by Mario Gabriele Carofano

	Questo script rappresenta il punto di ingresso principale per l'esecuzione
	delle varie funzionalità del progetto QML-for-Traffic-Classification.
"""

#   ####################################################################    #
#	Librerie

# Importing contants values
import importlib
import sys
sys.path.insert(1, './Source/')
import constants
importlib.reload(constants)
from constants import \
	_json_default, \
	build_session_config, \
	get_value_from_config

# Dataset loading and manipulation
import numpy as np
from pathlib import Path
import os

# Preprocessing and evaluation
# ...

# Machine Learning and Quantum ML
import torch

# Visualization
# ...

# More libraries
from typing import Any
import random

#   ####################################################################    #
#   Caricamento delle costanti strutturali e dei parametri di sessione

def load_constants() -> dict[str, Any]:
	"""Carica i valori per le costanti strutturali definite in constants.py.

	Raises:
		KeyError: Se una o più costanti richieste non sono presenti
		nel namespace o nella configurazione salvata.
	"""

	#   ################################################################    #
	#	Inizializzazione

	namespace = {}

	importlib.reload(constants)
	source = vars(constants)

	names = [
		'__DEBUG',
		'__USE_PRECOMPUTED_DATASET',
		'__USE_CONFIG_FILE',
		'__CONFIG_ID',
		'__EXEC_MODE_TRAIN',
		'__SELECTED_FOLD',
		'__SAVE_OUTPUT',
	]

	#   ################################################################    #
	#   Recupero dei valori, con fallback ai valori di default

	missing = [n for n in names if n not in source]
	if missing:
		print(
			f"Costanti non trovate: \n"
			f"{missing}",
			file=sys.stderr
		)

		raise KeyError()

		# end if

	for name in names:
		namespace[name] = get_value_from_config(source, key=name,)
		print(f"Costante '{name}' caricata con valore: {namespace[name]}")

		# end for

	print()

	return namespace

	# end

def load_session_parameters() -> tuple[dict[str, Any], Path, Path]:
	"""Carica i parametri di sessione dal file di configurazione, se presente,
	oppure li inizializza con valori di default.

	Returns:
		dict[str, Any]: Dizionario dei parametri di sessione.
		Path: Percorso della directory dei risultati per la sessione corrente.
		Path: Percorso del file di configurazione.

	Raises:
		FileNotFoundError: Se la directory dei risultati o il file di configurazione non vengono trovati.
		ValueError: Se l'ID della sessione nel file di configurazione non corrisponde a quello specificato.
	"""

	#   ################################################################    #
	#	Inizializzazione

	import json
	import datetime

	global structural_constants
	use_config_file = structural_constants["__USE_CONFIG_FILE"]
	config_id = structural_constants["__CONFIG_ID"]

	#   ################################################################    #

	# Nuova sessione
	if not use_config_file:
		config = build_session_config()

		id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
		config["session_id"] = id

		output_dir = Path(f"Results/{id}/")
		output_dir.mkdir(parents=True, exist_ok=True)

		config_file_path = output_dir / "config.json"
		config_file_path.parent.mkdir(parents=True, exist_ok=True)

	# Ripristino da file di configurazione esistente
	elif use_config_file:
		output_dir = Path(f"Results/{config_id}/")
		if not output_dir.exists():
			raise FileNotFoundError(f"Directory dei risultati non trovata: {output_dir}")
		
		config_file_path = output_dir / "config.json"
		if not config_file_path.exists():
			raise FileNotFoundError(f"File di configurazione non trovato: {config_file_path}")
		
		with open(config_file_path, "r", encoding="utf-8") as f:
			config = json.load(f)

		id = get_value_from_config(config, "session_id", None)
		if id and id != config_id:
			raise ValueError(
				f"ID della sessione nel file di configurazione ({id}) "
				f"non corrisponde a quello specificato ({config_id})."
			)

		# end if
	
	return (
		config,
		output_dir,
		config_file_path
	)
	
	# end
	
#   ####################################################################    #
# 	Creazione delle directory dei risultati per il fold selezionato.

def get_fold_directory() -> str:
	"""Crea e restituisce il percorso della directory dei risultati per il fold selezionato in modalità di addestramento, altrimenti verifica che esista (in base alla directory di output fornita nel file di configurazione).

	Raises:
		FileNotFoundError: Se la directory dei risultati per il fold selezionato non viene trovata.

	Returns:
		str: Percorso della directory dei risultati per il fold selezionato.
	"""

	#   ################################################################    #
	#	Inizializzazione

	global structural_constants
	selected_fold = structural_constants["__SELECTED_FOLD"]
	exec_mode_train = structural_constants["__EXEC_MODE_TRAIN"]

	global session_config
	test_folds = session_config["TEST_FOLDS"]

	assert 0 <= selected_fold < test_folds, \
		f"__SELECTED_FOLD deve essere compreso tra 0 e {test_folds-1}."

	#   ################################################################    #
	#	Creazione della directory dei risultati per il fold selezionato.

	fold_results_dir = output_dir / f"fold{selected_fold}"

	if exec_mode_train:
		fold_results_dir.mkdir(parents=True, exist_ok=True)

	elif not exec_mode_train:
		if not fold_results_dir.exists():
			raise FileNotFoundError(
				f"Directory dei risultati per il fold {selected_fold} "
				f"non trovata: {fold_results_dir}"
			)

		# end if
	
	return fold_results_dir

	# end
	
#   ####################################################################    #
#	Selezione del dispositivo di calcolo.

def get_device() -> torch.device:
	"""Restituisce il dispositivo di calcolo per tensori e modelli.

	Se non viene specificato nel file di configurazione, seleziona il primo dispositivo disponibile secondo la priorità MPS > CUDA > CPU e ne registra il tipo nel dizionario di configurazione della sessione.

	Returns:
		torch.device: Il dispositivo selezionato (CPU, CUDA o MPS).
	"""

	#   ################################################################    #
	#	Inizializzazione

	global structural_constants
	global session_config

	use_config_file = structural_constants["__USE_CONFIG_FILE"]

	#   ################################################################    #

	if not use_config_file:
		if torch.backends.mps.is_available():
			device = torch.device("mps")
		elif torch.cuda.is_available():
			device = torch.device("cuda")
		else:
			device = torch.device("cpu")

		session_config["device"] = device.type

	elif use_config_file:
		dev_name = get_value_from_config(session_config, "device", "cpu")
		device = torch.device(dev_name)

		# end if

	return device

	# end
	
#   ####################################################################    #
#   Main

if __name__ == "__main__":

	# Caricamento delle costanti strutturali.
	structural_constants = load_constants()

	# Caricamento dei parametri di sessione.
	# Si memorizzano i valori utilizzati in questo dizionario, in modo da poterli salvare in un file JSON alla fine dell'esecuzione.
	session_config, output_dir, config_file_path = load_session_parameters()
	assert not structural_constants.keys() & session_config.keys(), \
		"Caricamento delle costanti strutturali e dei parametri di sessione fallito: ci sono chiavi comuni."
	print(f"Directory dei risultati per la sessione corrente: {output_dir}")
	print(f"Percorso del file di configurazione per la sessione corrente: {config_file_path}")

	if structural_constants["__DEBUG"]:
		print(structural_constants.keys())
		print(session_config.keys())
		print({key: structural_constants[key] for key in structural_constants.keys() & session_config.keys()})

	#	-----------------------------------------------------------------	#

	# Creazione della directory dei risultati per il fold selezionato.
	fold_results_dir = get_fold_directory()
	print(f"Directory dei risultati per il fold selezionato: {fold_results_dir}")

	# Creazione dei percorsi dei file di output.
	fold_confusion_matrix_path = fold_results_dir / "confusion_matrix.png"
	fold_history_path = fold_results_dir / "training_history.csv"
	fold_model_path = fold_results_dir / "model.pth"
	fold_plots_path = fold_results_dir / "train_val_plots.png"
	fold_predictions_path = fold_results_dir / "predictions_test.csv"
	fold_report_path = fold_results_dir / "classification_report.txt"

	#	-----------------------------------------------------------------	#

	# Configurazione per la riproducibilità.
	random.seed(session_config["RANDOM_SEED"])
	np.random.seed(session_config["RANDOM_SEED"])
	os.environ['PYTHONHASHSEED'] = str(session_config["RANDOM_SEED"])

	torch.manual_seed(session_config["RANDOM_SEED"])
	torch.cuda.manual_seed(session_config["RANDOM_SEED"])
	torch.cuda.manual_seed_all(session_config["RANDOM_SEED"])

	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

	# Selezione del dispositivo.
	device = get_device()

	print(f"Dispositivo selezionato: {device.type}")
	if device.type == "cuda":
		print(f"GPU: {torch.cuda.get_device_name(0)}")

	#	-----------------------------------------------------------------	#

	# ...

	# end
