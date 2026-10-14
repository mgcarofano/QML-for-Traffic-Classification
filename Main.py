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
	get_value_from_config, \
	debug_print

# Dataset loading and manipulation
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
from traffic_converter import \
	get_flowpic_dir, \
	mirage_pickle_converter, \
	flow_debug

# Preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

# Machine Learning and Quantum ML
import torch

# Visualization
# ...

# More libraries
from typing import Any
import random
import json
import time
from datetime import datetime

#   ####################################################################    #
#   Caricamento delle costanti strutturali e dei parametri di sessione.

from constants import \
	__DEBUG, \
	__USE_PRECOMPUTED_DATASET, \
	__USE_CONFIG_FILE, \
	__CONFIG_ID, \
	__EXEC_MODE_TRAIN, \
	__SELECTED_FOLD, \
	__SAVE_OUTPUT

def create_session() -> tuple[dict[str, Any], Path, Path]:
	"""Crea una nuova sessione caricandone i parametri da un file di configurazione esistente o inizializzandoli con valori di default.

	Returns:
		dict[str, Any]: Dizionario dei parametri di sessione.
		Path: Percorso della directory dei risultati per la sessione corrente.
		Path: Percorso del file di configurazione per la sessione corrente.

	Raises:
		FileNotFoundError: Se la directory dei risultati o il file di configurazione non vengono trovati.
		ValueError: Se l'ID della sessione nel file di configurazione non corrisponde a quello specificato.
	"""

	#   ################################################################    #

	# Nuova sessione
	if not __USE_CONFIG_FILE:
		config = build_session_config()

		id = datetime.now().strftime("%Y-%m-%d_%H-%M")
		config["session_id"] = id

		output_dir = Path(f"Results/{id}/")
		output_dir.mkdir(parents=True, exist_ok=True)

		config_file_path = output_dir / "config.json"
		config_file_path.parent.mkdir(parents=True, exist_ok=True)

	# Ripristino da file di configurazione esistente
	elif __USE_CONFIG_FILE:
		output_dir = Path(f"Results/{__CONFIG_ID}/")
		if not output_dir.exists():
			raise FileNotFoundError(f"Directory dei risultati non trovata: {output_dir}")
		
		config_file_path = output_dir / "config.json"
		if not config_file_path.exists():
			raise FileNotFoundError(f"File di configurazione non trovato: {config_file_path}")
		
		with open(config_file_path, "r", encoding="utf-8") as f:
			config = json.load(f)

		id = get_value_from_config(config, "session_id")
		if id and id != __CONFIG_ID:
			raise ValueError(
				f"ID della sessione nel file di configurazione ({id}) "
				f"non corrisponde a quello specificato ({__CONFIG_ID})."
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

	global session_config
	test_folds = get_value_from_config(session_config, "TEST_FOLDS")

	assert 0 <= __SELECTED_FOLD < test_folds, \
		f"__SELECTED_FOLD deve essere compreso tra 0 e {test_folds-1}."

	#   ################################################################    #
	#	Creazione della directory dei risultati per il fold selezionato.

	fold_results_dir = output_dir / f"fold{__SELECTED_FOLD}"

	if __EXEC_MODE_TRAIN:
		fold_results_dir.mkdir(parents=True, exist_ok=True)

	elif not __EXEC_MODE_TRAIN:
		if not fold_results_dir.exists():
			raise FileNotFoundError(
				f"Directory dei risultati per il fold {__SELECTED_FOLD} "
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

	global session_config

	#   ################################################################    #

	if not __USE_CONFIG_FILE:
		if torch.backends.mps.is_available():
			device = torch.device("mps")
		elif torch.cuda.is_available():
			device = torch.device("cuda")
		else:
			device = torch.device("cpu")

		session_config["device"] = device.type
		print(f"Dispositivo selezionato: {device.type}")

	elif __USE_CONFIG_FILE:
		dev_name = get_value_from_config(session_config, "device", "cpu")
		device = torch.device(dev_name)

		# end if

	return device

	# end
	
#   ####################################################################    #
#	Caricamento o generazione (opzionale) del dataset

def load_dataset() -> tuple[np.ndarray, np.ndarray, dict]:
	"""Carica il dataset dal file PICKLE specificato.

	Se `USE_FLOWPIC_DATASET` è impostato su True nella configurazione della sessione,
	questa funzione dovrebbe caricare il dataset FlowPic precomputato dal file .npz corrispondente.

	Args:
		file_path (str): Percorso del file PICKLE da convertire.

	Returns:
		tuple[np.ndarray, np.ndarray, dict]: Una tupla contenente i dati dei flussi (X_raw), le etichette (y_raw) e i relativi metadati.
	"""

	#   ################################################################    #
	#	Inizializzazione

	global session_config

	use_flowpic_dataset = get_value_from_config(session_config, "USE_FLOWPIC_DATASET")
	data_path = get_value_from_config(session_config, "DATA_PATH")
	dataset_name = get_value_from_config(session_config, "DATASET_NAME")

	#   ################################################################    #

	if not use_flowpic_dataset:

		# Il file pickle contiene due oggetti salvati in sequenza:
		# 1. X_raw : i dati numerici dei flussi di traffico, in formato numpy array.
		# 2. y_raw : le etichette corrispondenti ai flussi.
		# Ogni chiamata a pickle.load() legge il successivo oggetto nel file.
		with open(f"{data_path}/{dataset_name}", "rb") as f:
			X_raw = np.array(pickle.load(f), dtype=np.float32)
			y_raw = np.array(pickle.load(f))

		print(f"\nCaricato dataset (shape={X_raw.shape}) da: {data_path}/{dataset_name}")

		metadata = pd.DataFrame({
			"FlowID": np.arange(len(X_raw)),
			"DatasetID": np.arange(len(X_raw)),
			"Label": y_raw,
		})

	elif use_flowpic_dataset:

		flowpics_name = dataset_name.split('.')[0]

		flowpics_dir = get_flowpic_dir(data_path)
		flowpics_dir.mkdir(parents=True, exist_ok=True)

		npz_path = flowpics_dir / f"{flowpics_name}.npz"
		meta_path = flowpics_dir / f"{flowpics_name}.csv"

		if not __USE_PRECOMPUTED_DATASET:
			histograms, metadata = mirage_pickle_converter(
				f"{data_path}/{dataset_name}",
				session_config
			)

			# Se il flag di debug è attivo, si stampano informazioni dettagliate sui flussi specificati.
			flow_debug(histograms, metadata)

			# Salvataggio degli istogrammi 2D FlowPic in formato NumPy.
			np.savez_compressed(npz_path, data=histograms)
			print(f"\nSalvato dataset (shape={histograms.shape}) in: {npz_path}")

			# Salvataggio dei metadati in formato CSV.
			metadata.to_csv(meta_path, index=False)
			print(f"Salvati metadati (di {len(metadata)} righe) in: {meta_path} ")

		elif __USE_PRECOMPUTED_DATASET:
			if not npz_path.exists():
				raise FileNotFoundError(f"Array di istogrammi precomputati non trovato: {npz_path}")
			if not meta_path.exists():
				raise FileNotFoundError(f"DataFrame di metadati precomputati non trovato: {meta_path}")

			# Caricamento degli istogrammi 2D FlowPic precomputati.
			histograms = np.load(npz_path)['data']
			print(f"\nCaricato dataset (shape={histograms.shape}) da: {npz_path}")
	
			# Caricamento dei metadati in formato CSV.
			metadata = pd.read_csv(meta_path)
			print(f"Caricati metadati (di {len(metadata)} righe) da: {meta_path} ")

		# end if __USE_PRECOMPUTED_DATASET
	
		# Prima di procedere con split/training, si verifica che gli istogrammi e i metadati siano allineati.
		assert histograms.shape[0] == len(metadata), "Disallineamento tra istogrammi e metadati!"
		assert list(metadata["DatasetID"]) == list(range(len(metadata))), "DatasetID non contiguo!"

		X_raw = histograms
		y_raw = metadata["Label"].values

	# end if USE_FLOWPIC_DATASET

	return X_raw, y_raw, metadata

	# end

#   ####################################################################    #
#	Generazione delle etichette per le classi del dataset.

def encode_labels(y_raw, labels_map_path):
	"""Codifica le etichette del dataset utilizzando LabelEncoder.

	Args:
		y_raw (array-like): Array contenente le etichette grezze del dataset.
		labels_map_path (str): Percorso del file JSON in cui salvare o leggere la mappatura delle etichette.

	Raises:
		ValueError: Se le etichette del dataset non corrispondono a quelle salvate nella configurazione.

	Returns:
		tuple: L'array completo delle etichette codificate e un dizionario che mappa in modo univoco le etichette originali.
	"""

	le = LabelEncoder()

	y_encoded = le.fit_transform(y_raw)
	ordinal_classes = np.unique(y_encoded)
	labels_map = {}
	for l, c in zip(ordinal_classes, le.inverse_transform(ordinal_classes)):
		labels_map[int(l)] = c

	if not __USE_CONFIG_FILE:
		with open(labels_map_path, "w", encoding="utf-8") as f:
			json.dump(labels_map, f)

	elif __USE_CONFIG_FILE:
		# Se il file di configurazione esiste, si verifica che le classi nel dataset corrispondano a quelle nel file di configurazione.
		with open(labels_map_path, "r", encoding="utf-8") as f:
			config_labels_map = json.load(f)

		# Le chiavi lette da JSON sono stringhe.
		# Vengono prima convertite per essere confrontate.
		config_labels_map = {int(key): value for key, value in config_labels_map.items()}

		if config_labels_map != labels_map:
			raise ValueError(
				f"Le etichette del dataset ({labels_map}) non corrispondono "
				f"a quelle salvate nella configurazione ({config_labels_map})."
			)

		# end __USE_CONFIG_FILE
	
	assert len(X_raw) == len(y_encoded), (
		"Disallineamento tra dati e etichette!"
	)

	return y_encoded, labels_map

#   ####################################################################    #
#	Implementazione della K-fold cross-validation.

def reduce_dataset(X_raw, y_encoded, metadata) -> tuple:
	"""Riduce il dataset in base ai parametri della sessione corrente.

	Args:
		X_raw (array-like): Array contenente i dati grezzi del dataset.
		y_encoded (array-like): Array contenente le etichette codificate del dataset.
		metadata (DataFrame): DataFrame contenente i metadati associati ai campioni del dataset.

	Returns:
		tuple: Una tupla contenente i tre input ridotti, se la configurazione lo richiede. Altrimenti, restituisce gli input originali.
	"""

	#   ################################################################    #
	#	Inizializzazione

	global session_config

	sub_sample_ratio = get_value_from_config(session_config, "SUB_SAMPLE_RATIO")
	random_seed = get_value_from_config(session_config, "RANDOM_SEED")
	
	#   ################################################################    #

	# Se il rapporto di sub-sampling è inferiore a 1.0, si riduce il dataset di conseguenza.
	if sub_sample_ratio < 1.0 and len(X_raw) > 0:
		selected_indices, _ = train_test_split(
			np.arange(len(X_raw)),
			train_size=sub_sample_ratio,
			stratify=y_encoded,
			random_state=random_seed
		)

		X_raw = X_raw[selected_indices]
		y_encoded = y_encoded[selected_indices]
		metadata = metadata.iloc[selected_indices].copy().reset_index(drop=True)

		assert len(X_raw) == len(y_encoded) == len(metadata), (
			"Disallineamento dopo la riduzione del dataset!"
		)

		# end if SUB_SAMPLE_RATIO
	
	return X_raw, y_encoded, metadata

	# end

def split_dataset(X_raw, y_encoded) -> list:
	"""Suddivide il dataset in folds per la K-fold cross-validation.

	Args:
		X_raw (array-like): Array contenente i dati grezzi del dataset.
		y_encoded (array-like): Array contenente le etichette codificate del dataset.
	
	Returns:
		list: Una lista contenente le suddivisioni del dataset in folds.
	"""

	#   ################################################################    #
	#	Inizializzazione

	global session_config

	test_folds = get_value_from_config(session_config, "TEST_FOLDS")
	random_seed = get_value_from_config(session_config, "RANDOM_SEED")

	#   ################################################################    #
	#	Suddivisione del dataset in folds.

	skf = StratifiedKFold(
		n_splits = test_folds,
		shuffle = True,
		random_state = random_seed
	)

	folds = list(skf.split(X_raw, y_encoded))

	assert len(folds) > 0, "Nessun fold generato da StratifiedKFold!"
	assert len(folds) == test_folds, "Il numero di folds generati non corrisponde al numero di test_folds specificato!"
	
	return folds

	# end

def split_train_val(train_indices_temp: np.ndarray, y_temp: np.ndarray) -> tuple:
	"""Suddivide il dataset temporaneo per il training in Train e Validation in base ai parametri della sessione corrente.

	Args:
		train_indices_temp (np.ndarray): Array contenente gli indici di training temporanei.
		y_temp (np.ndarray): Array contenente le etichette codificate corrispondenti agli indici di training temporanei.

	Returns:
		tuple: Una tupla contenente gli indici di training e validation suddivisi.
	"""

	#   ################################################################    #
	#	Inizializzazione

	global session_config

	random_seed = get_value_from_config(session_config, "RANDOM_SEED")

	train_size = get_value_from_config(session_config, "TRAIN_SIZE")
	val_size = get_value_from_config(session_config, "VAL_SIZE")

	assert train_size + val_size == 1.0, \
		"Le percentuali di Train e Val devono sommare a 1.\n" \
		f"Train: {train_size:.4f}, " \
		f"Val: {val_size:.4f}, " \
		f"Total: {(train_size + val_size):.4f}"
	
	#   ################################################################    #

	val_fraction_of_temp = val_size / (train_size + val_size)
	train_indices, val_indices = train_test_split(
		train_indices_temp,
		test_size=val_fraction_of_temp,
		stratify=y_temp,
		random_state=random_seed
	)

	return train_indices, val_indices

#   ####################################################################    #
#   Main

if __name__ == "__main__":

	#	-----------------------------------------------------------------	#
	#	Caricamento dei parametri di sessione.

	# Si memorizzano i valori utilizzati in un dizionario, in modo da poterli salvare in un file JSON alla fine dell'esecuzione.
	session_config, output_dir, config_file_path = create_session()
	print(f"Directory dei risultati per la sessione corrente: {output_dir}")
	print(f"Percorso del file di configurazione per la sessione corrente: {config_file_path}")

	#	-----------------------------------------------------------------	#
	#	Creazione dei percorsi dei file di output.

	labels_map_path = output_dir / "labels_map.json"

	fold_results_dir = get_fold_directory()
	print(f"Directory dei risultati per il fold selezionato: {fold_results_dir}")

	fold_confusion_matrix_path = fold_results_dir / "confusion_matrix.png"
	fold_history_path = fold_results_dir / "training_history.csv"
	fold_model_path = fold_results_dir / "model.pth"
	fold_plots_path = fold_results_dir / "train_val_plots.png"
	fold_predictions_path = fold_results_dir / "predictions_test.csv"
	fold_report_path = fold_results_dir / "classification_report.txt"

	#	-----------------------------------------------------------------	#
	#	Configurazione per la riproducibilità.

	random_seed = get_value_from_config(session_config, "RANDOM_SEED")
	
	random.seed(random_seed)
	np.random.seed(random_seed)
	os.environ['PYTHONHASHSEED'] = str(random_seed)

	torch.manual_seed(random_seed)
	torch.cuda.manual_seed(random_seed)
	torch.cuda.manual_seed_all(random_seed)

	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

	#	-----------------------------------------------------------------	#
	# 	Selezione del dispositivo.

	device = get_device()
	if device.type == "cuda":
		print(f"GPU: {torch.cuda.get_device_name(0)}")

	#	-----------------------------------------------------------------	#
	#	Caricamento o generazione (opzionale) del dataset.

	print("\nLoading dataset...")
	print(f"Costante '__DEBUG' caricata con valore: {__DEBUG}")
	print(f"Costante '__USE_PRECOMPUTED_DATASET' caricata con valore: {__USE_PRECOMPUTED_DATASET}.")

	X_raw, y_raw, metadata = load_dataset()

	#	-----------------------------------------------------------------	#
	#	Generazione delle etichette per le classi del dataset.

	y_encoded, labels_map = encode_labels(y_raw, labels_map_path)
	num_classes = len(labels_map)

	print(f"\nNumero di classi: {num_classes}")
	print(f"Etichette delle classi generate: {labels_map.values()}")
	print()

	#	-----------------------------------------------------------------	#
	#	Implementazione della K-fold cross-validation.

	# Riduzione del dataset (opzionale).
	X_raw, y_encoded, metadata = reduce_dataset(X_raw, y_encoded, metadata)

	# Suddivisione del dataset.
	folds = split_dataset(X_raw, y_encoded)

	#	-----------------------------------------------------------------	#

	# ...

	#	-----------------------------------------------------------------	#
	#	Inizio del training / testing loop per ogni fold del dataset.

	def extract_subset(indices):
		"""Estrae feature, etichette e metadati per gli indici specificati."""
		
		return (
			X_raw[indices],
			y_encoded[indices],
			metadata.iloc[indices].copy().reset_index(drop=True)
		)
	
		# end extract_subset
	
	print(f"\n#  {'-'*20}  #")

	for fold_index, current_fold in enumerate(folds):

		print(f"\nInizio elaborazione del fold {fold_index + 1}/{len(folds)}")
		print(f"Orario corrente: {datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
		start_fold_time = time.time()

		#	-------------------------------------------------------------	#
		#	Suddivisione del dataset in Train / Val / Test.

		temp_indices, test_indices = current_fold
		train_indices, val_indices = split_train_val(temp_indices, y_encoded[temp_indices])

		X_train, y_train, metadata_train = extract_subset(train_indices)
		X_val, y_val, metadata_val = extract_subset(val_indices)
		X_test, y_test, metadata_test = extract_subset(test_indices)

		#	-------------------------------------------------------------	#
		#	Verifica della correttezza degli split.

		# Verifica della lunghezza dei set risultanti rispetto al dataset originale.
		assert len(X_train) + len(X_val) + len(X_test) == len(X_raw), \
			"La somma dei campioni Train, Val e Test non corrisponde al totale!"

		all_indices = np.arange(len(X_raw))
		split_indices = np.concatenate([train_indices, val_indices, test_indices])
		assert np.array_equal(np.sort(all_indices), np.sort(split_indices)), \
			"Gli split non coprono esattamente il dataset originale!"

		# Verifica dell'allineamento in ciascuno split.
		# TODO: vedi notebook.
		
		#	-------------------------------------------------------------	#
		#	Stampa dei risultati.

		if __DEBUG:
			print(f"\nTrain shape: {X_train.shape} ({len(X_train)/len(X_raw):.1%})")
			print(metadata_train.head())
			print(f"\nValidation shape: {X_val.shape} ({len(X_val)/len(X_raw):.1%})")
			print(metadata_val.head())
			print(f"\nTest shape: {X_test.shape} ({len(X_test)/len(X_raw):.1%})")
			print(metadata_test.head())

		#	-------------------------------------------------------------	#
		#	title

		# ...

		#	-------------------------------------------------------------	#
		#	Fine dell'elaborazione del fold corrente.

		print(f"\nFine elaborazione del fold {fold_index + 1}/{len(folds)}")
		print(f"Orario corrente: {datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")

		end_fold_time = time.time()
		print(f"Tempo di elaborazione: {end_fold_time - start_fold_time:.2f} secondi")

		print(f"\n#  {'-'*20}  #")

		# end for fold_index, current_fold

	#	-----------------------------------------------------------------	#

	# ...

	# end
