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
from classes import TrafficDataset
from torch.utils.data import DataLoader

# Preprocessing
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

# Machine Learning and Quantum ML
import torch
import torch.nn as nn
from model_selection import \
	select_model, \
	weight_init, \
	check_model_compatibility
from training_functions import \
	train_epoch, \
	evaluate

# Visualization
import matplotlib.pyplot as plt
from visualization_functions import \
	draw_train_val_plot, \
	draw_confusion_matrix
from sklearn.metrics import \
	confusion_matrix, \
	classification_report

# More libraries
from typing import Any
import random
import json
import time
from datetime import datetime
import copy

#   ####################################################################    #
#   Caricamento delle costanti strutturali e dei parametri di sessione.

from constants import \
	__DEBUG, \
	__USE_PRECOMPUTED_DATASET, \
	__USE_CONFIG_FILE, \
	__CONFIG_ID, \
	__EXEC_MODE_TRAIN, \
	__SAVE_OUTPUT

def create_session(id: str) -> tuple[dict[str, Any], Path, Path]:
	"""Crea una nuova sessione caricandone i parametri da un file di configurazione esistente o inizializzandoli con valori di default.

	Args:
		id (str): Identificatore della sessione.

	Returns:
		dict[str, Any]: Dizionario dei parametri di sessione.
		Path: Percorso della directory dei risultati per la sessione corrente.
		Path: Percorso del file di configurazione per la sessione corrente.

	Raises:
		FileNotFoundError: Se la directory dei risultati o il file di configurazione non vengono trovati.
		ValueError: Se l'ID della sessione nel file di configurazione non corrisponde a quello specificato.
	"""

	# Nuova sessione
	if not __USE_CONFIG_FILE:
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

		# end if
	
	return output_dir, config_file_path
	
	# end
	
#   ####################################################################    #
# 	Creazione delle directory dei risultati per il fold selezionato.

def get_fold_directory(selected_fold: int) -> str:
	"""Crea e restituisce il percorso della directory dei risultati per il fold selezionato in modalità di addestramento, altrimenti verifica che esista (in base alla directory di output fornita nel file di configurazione).

	Args:
		selected_fold (int): Indice del fold selezionato.

	Raises:
		FileNotFoundError: Se la directory dei risultati per il fold selezionato non viene trovata.

	Returns:
		str: Percorso della directory dei risultati per il fold selezionato.
	"""

	#   ################################################################    #
	#	Inizializzazione

	global session_config
	test_folds = get_value_from_config(session_config, "TEST_FOLDS")

	assert 0 <= selected_fold < test_folds, \
		f"selected_fold deve essere compreso tra 0 e {test_folds-1}."

	#   ################################################################    #
	#	Creazione della directory dei risultati per il fold selezionato.

	fold_results_dir = output_dir / f"fold{selected_fold}"

	if __EXEC_MODE_TRAIN:
		fold_results_dir.mkdir(parents=True, exist_ok=True)

	elif not __EXEC_MODE_TRAIN:
		if not fold_results_dir.exists():
			raise FileNotFoundError(
				f"Directory dei risultati per il fold {selected_fold} "
				f"non trovata: {fold_results_dir}"
			)

		# end if
	
	return fold_results_dir

	# end
	
#   ####################################################################    #
#	Caricamento o generazione (opzionale) del dataset.

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
#	Applicazione delle strategie di data preprocessing.

def apply_data_preprocessing(X, scaler):
	"""Applica il preprocessing ai dati utilizzando lo scaler specificato.

	Args:
		X (np.ndarray): Array contenente i dati da preprocessare.
		scaler: Oggetto scaler utilizzato per normalizzare i dati.

	Returns:
		np.ndarray: Array contenente i dati preprocessati.
	"""

	#   ################################################################    #
	#	Inizializzazione

	global session_config

	n_packets = get_value_from_config(session_config, "N_PACKETS")
	n_features = get_value_from_config(session_config, "N_FEATURES")

	#   ################################################################    #

	X_proc = X.copy().astype(np.float32)
	X_proc = X_proc[:, :n_packets, :]
	X_proc = scaler.transform(np.reshape(X_proc, [-1, n_features]))
	X_proc = np.reshape(X_proc, [-1, n_packets, n_features])

	return X_proc

	# end

#   ####################################################################    #
#   Configurazione dei componenti di addestramento.

def build_optimizer(
		selected_optimizer : str,
		model : nn.Module,
		learning_rate : float,
		weight_decay : float,
		epsilon : float,
		momentum : float,
	) -> torch.optim.Optimizer:
	"""Costruisce l'ottimizzatore in base alla selezione dell'utente.

	Args:
		selected_optimizer (str): Tipo di ottimizzatore selezionato dall'utente.
		model (nn.Module): Modello da ottimizzare.
		learning_rate (float): Learning rate dell'ottimizzatore.
		weight_decay (float): Peso del termine di regolarizzazione L2.
		epsilon (float): Valore epsilon per ottimizzatori come AdamW.
		momentum (float): Momento per ottimizzatori come SGD e RMSprop.

	Raises:
		ValueError: Se il tipo di ottimizzatore selezionato non è supportato.

	Returns:
		torch.optim.Optimizer:
		L'ottimizzatore costruito in base alla selezione dell'utente.
	"""

	if selected_optimizer == 'SGD':
		optimizer = torch.optim.SGD(
			model.parameters(),
			lr=learning_rate, weight_decay=weight_decay, momentum=momentum
		)

	elif selected_optimizer == 'Adam':
		optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

	elif selected_optimizer == 'RMSprop':
		optimizer = torch.optim.RMSprop(
			model.parameters(),
			lr=learning_rate, weight_decay=weight_decay, momentum=momentum
		)

	elif selected_optimizer == 'AdamW':
		optimizer = torch.optim.AdamW(
			model.parameters(),
			lr=learning_rate, weight_decay=weight_decay, eps=epsilon
		)

	else:
		raise ValueError(
			f"Tipo di optimizer non supportato: {selected_optimizer}. "
			"Usare: 'Adam', 'AdamW', 'SGD' oppure 'RMSprop'."
		)

	return optimizer

	# end

def build_scheduler(
		selected_scheduler : str,
		optimizer : torch.optim.Optimizer,
		train_loader : torch.utils.data.DataLoader,
		epochs : int,
		max_lr : float = None,
		start_factor : float = None,
		step_size : int = None,
		steplr_gamma : float = None
	) -> torch.optim.lr_scheduler._LRScheduler | None:
	"""Costruisce lo scheduler di apprendimento in base alla selezione dell'utente.

	Args:
		selected_scheduler (str): Tipo di scheduler selezionato dall'utente.
		optimizer (torch.optim.Optimizer): Ottimizzatore da associare allo scheduler.
		train_loader (torch.utils.data.DataLoader): DataLoader del set di addestramento.
		epochs (int): Numero totale di epoche di addestramento.
		max_lr (float, opzionale): Massimo learning rate per OneCycleLR.
		start_factor (float, opzionale): Fattore iniziale per LinearLR.
		step_size (int, opzionale): Step size per StepLR.
		steplr_gamma (float, opzionale): Fattore gamma per StepLR.

	Raises:
		ValueError: Se il tipo di scheduler selezionato non è supportato.

	Returns:
		torch.optim.lr_scheduler._LRScheduler | None:
		Lo scheduler di apprendimento costruito in base alla selezione dell'utente,
		oppure None se 'NoSched' è selezionato.
	"""

	if selected_scheduler == 'OneCycleSched':
		lr_sched = torch.optim.lr_scheduler.OneCycleLR(
			optimizer,
			max_lr=max_lr, steps_per_epoch=len(train_loader), epochs=epochs
		)

	elif selected_scheduler == 'LinearSched':
		lr_sched = torch.optim.lr_scheduler.LinearLR(
			optimizer,
			start_factor=start_factor, total_iters=epochs
		)

	elif selected_scheduler == 'StepSched':
		lr_sched = torch.optim.lr_scheduler.StepLR(
			optimizer,
			step_size=step_size, gamma=steplr_gamma
		)

	elif selected_scheduler == 'NoSched':
		lr_sched = None

	else:
		raise ValueError(
			f"Tipo di scheduler non supportato: {selected_scheduler}. "
			"Usare: 'OneCycleSched', 'LinearSched', 'StepSched' oppure 'NoSched'."
		)

	return lr_sched

	# end

def build_loss_function(
		selected_loss : str,
		y_train : torch.Tensor,
		n_classes : int,
		device : torch.device,
		dtype: torch.dtype,
		alpha : str = None,
		focal_loss_gamma : float = None
	) -> nn.Module:
	"""Costruisce la funzione di loss in base alla selezione dell'utente.

	Args:
		selected_loss (str): Tipo di loss selezionata dall'utente.
		y_train (torch.Tensor): Tensor contenente le etichette di addestramento.
		n_classes (int): Numero totale di classi.
		device (torch.device): Il device su cui eseguire i calcoli.
		alpha (str, opzionale): Tipo di alpha per la Focal Loss.
		focal_loss_gamma (float, opzionale): Parametro gamma per la Focal Loss.

	Raises:
		ValueError: Se il tipo di loss selezionato non è supportato.
		ValueError: Se il tipo di alpha per la Focal Loss non è supportato.

	Returns:
		nn.Module: La funzione di loss costruita.
	"""

	def compute_class_weights() -> torch.FloatTensor:
		""" Calcola i pesi di classe per il bilanciamento del dataset. """

		y = np.asarray(y_train.cpu() if torch.is_tensor(y_train) else y_train).astype(int)
		counts = np.bincount(y, minlength=n_classes)
		weights = np.where(counts > 0, len(y) / (n_classes * np.maximum(counts, 1)), 1.0)

		return torch.tensor(
			weights,
			dtype=dtype,
			device=device
		)

		# end

	if selected_loss == "CrossEntropy":
		return nn.CrossEntropyLoss()

	elif selected_loss == "WeightedCrossEntropy":
		w = compute_class_weights().to(device)
		return nn.CrossEntropyLoss(weight=w)

	elif selected_loss == "Focal":
		if alpha == "class_weights":
			alpha = compute_class_weights()
		elif alpha == "uniform":
			alpha_tensor = torch.ones(
                n_classes,
                dtype=dtype,
                device=device
            )
		else:
			raise ValueError(
				f"Tipo di alpha non supportato: {alpha}. "
				"Usare: 'class_weights' oppure 'uniform'."
			)

		return torch.hub.load(
			'adeelh/pytorch-multi-class-focal-loss',
			model='FocalLoss',
			alpha=alpha,
			gamma=focal_loss_gamma,
			reduction='mean',
			force_reload=False
		)

	else:
		raise ValueError(
			f"Tipo di loss non supportato: {selected_loss}. "
			"Usare: 'CrossEntropy', 'WeightedCrossEntropy' oppure 'Focal'."
		)

	# end

#   ####################################################################    #
#	Costruzione del soft-output per il Test set.

def compute_soft_output(model, device, test_loader, criterion=None):
	
	print(f"EXEC_MODE_TRAIN impostato su {__EXEC_MODE_TRAIN}.\n")
	if not __EXEC_MODE_TRAIN:
		print("Non è stato eseguito alcun addestramento.\n"
			"Il modello da valutare è stato caricato da un checkpoint salvato.")
		test_loss, test_acc, y_preds, y_true, y_probs = evaluate(
			model, device,
			test_loader,
		)

	elif __EXEC_MODE_TRAIN:
		test_loss, test_acc, y_preds, y_true, y_probs = evaluate(
			model, device,
			test_loader,
			criterion
		)

	# Calcola la loss e l'accuratezza per il test set, per valutare le prestazioni del modello su dati mai visti prima.
	print(f"Test Loss: {test_loss:.4f} | " if test_loss is not None else "", end="")
	print(f"Test Acc: {test_acc:.2f}%")

	#   ################################################################    #
	#	Ritorno dei risultati.

	soft_output = [
		";".join(f"{probability:.10f}" for probability in row)
		for row in y_probs.tolist()
	]

	ret = pd.DataFrame({
		"FlowID": metadata_test["FlowID"].tolist(),
		"Actual": y_true.tolist(),
		"Predicted": y_preds.tolist(),
		"Soft-output": soft_output,
	})

	return ret

	# end

#   ####################################################################    #
#   Main

if __name__ == "__main__":

	total_start_time = time.time()
	print(f"\nInizio elaborazione.")
	print(f"Orario corrente: {datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")

	#	-----------------------------------------------------------------	#
	# 	Selezione del dispositivo.

	if torch.backends.mps.is_available():
		device = torch.device("mps")
	elif torch.cuda.is_available():
		device = torch.device("cuda")
	else:
		device = torch.device("cpu")

	print(f"\nDispositivo selezionato: {device.type}")
	
	if device.type == "cuda":
		print(f"GPU: {torch.cuda.get_device_name(0)}")

	#	-----------------------------------------------------------------	#
	#	Caricamento e salvataggio dei parametri di sessione.

	session_id = \
		datetime.now().strftime("%Y-%m-%d_%H-%M") \
		if not __USE_CONFIG_FILE \
		else __CONFIG_ID

	# Si memorizzano i valori utilizzati in un dizionario, in modo da poterli salvare in un file JSON alla fine dell'esecuzione.
	output_dir, config_file_path = create_session(session_id)

	if __USE_CONFIG_FILE:
		with open(config_file_path, "r", encoding="utf-8") as f:
			session_config = json.load(f)

	elif not __USE_CONFIG_FILE:
		session_config = {
			"session_id": session_id,
			"device": device.type,
			**build_session_config()
		}

		with open(config_file_path, "w", encoding="utf-8") as file:
			json.dump(
				session_config,
				file, indent=4,
				default=_json_default,
				ensure_ascii=False
			)

		# end __USE_CONFIG_FILE

	print(f"Directory dei risultati per la sessione corrente: {output_dir}")
	print(f"Percorso del file di configurazione per la sessione corrente: {config_file_path}")

	# TODO: aggiungere caricamento preliminare dei singoli parametri di sessione dal dizionario session_config.
	# ...

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
	#	Caricamento o generazione (opzionale) del dataset.

	print("\nLoading dataset...")
	print(f"Costante '__DEBUG' caricata con valore: {__DEBUG}")
	print(f"Costante '__USE_PRECOMPUTED_DATASET' caricata con valore: {__USE_PRECOMPUTED_DATASET}.")

	use_flowpic_dataset = get_value_from_config(session_config, "USE_FLOWPIC_DATASET")

	X_raw, y_raw, metadata = load_dataset()

	#	-----------------------------------------------------------------	#
	#	Generazione delle etichette per le classi del dataset.

	# Creazione del percorso di output per la mappatura delle etichette.
	labels_map_path = output_dir / "labels_map.json"

	y_encoded, labels_map = encode_labels(y_raw, labels_map_path)
	num_classes = len(labels_map)

	print(f"\nNumero di classi: {num_classes}")
	print(f"Etichette delle classi generate: {labels_map.values()}\n")

	#	-----------------------------------------------------------------	#
	#	Implementazione della K-fold cross-validation.

	# Riduzione del dataset (opzionale).
	X_raw, y_encoded, metadata = reduce_dataset(X_raw, y_encoded, metadata)

	# Suddivisione del dataset.
	folds = split_dataset(X_raw, y_encoded)

	#	-----------------------------------------------------------------	#
	#	Selezione della strategia di data preprocessing.

	if use_flowpic_dataset:
		debug_print("La fase di preprocessing è disponibile solo per il dataset grezzo (Mirage).")

	elif not use_flowpic_dataset:
		preprocessing_strategy = get_value_from_config(session_config, "PREPROCESSING_STRATEGY")

		if preprocessing_strategy == "MinMax":
			scaler = MinMaxScaler(feature_range=(0, 1))

		else:
			raise ValueError(f"Strategia di preprocessing non supportata: {preprocessing_strategy}")

		# end if

	#	-----------------------------------------------------------------	#
	#	Scelta e istanziazione del modello.

	selected_model = get_value_from_config(session_config, "SELECTED_MODEL")

	n_features = get_value_from_config(session_config, "N_FEATURES")
	n_packets = get_value_from_config(session_config, "N_PACKETS")

	# TODO: modificare la selezione del numero di layer.
	num_layers = \
		get_value_from_config(session_config, "N_LAYERS_RESNET") \
		if selected_model == "resnet" \
		else get_value_from_config(session_config, "N_LAYERS_QUANTUM")

	n_qubits = get_value_from_config(session_config, "N_QUBITS")
	dev_name = get_value_from_config(session_config, "SIMULATOR")

	model, is_quantum = select_model(
		selected_model,
		n_features, n_packets, num_classes,
		num_layers, n_qubits, dev_name,
		random_seed
	)

	# Si sposta il modello sul dispositivo corretto (CPU, GPU o MPS)
	# e si imposta il tipo di dato appropriato.
	model = \
		model.to(device).float() \
		if device.type == 'mps' \
		else model.to(device).double()

	print(f"Modello selezionato: {selected_model} -> {model.get_model_name()}")

	# Stampa completa delle informazioni sul modello.
	debug_print(model)

	#	-----------------------------------------------------------------	#
	#	Scelta della strategia di ottimizzazione.

	weight_decay = get_value_from_config(session_config, "WEIGHT_DECAY")
	epsilon = get_value_from_config(session_config, "EPSILON")
	momentum = get_value_from_config(session_config, "MOMENTUM")

	if not __EXEC_MODE_TRAIN:
		print(f"EXEC_MODE_TRAIN impostato su False.")
		print("La selezione dell'optimizer è irrilevante in questa modalità.")
		selected_optimizer = None

	elif __EXEC_MODE_TRAIN:
		selected_optimizer = get_value_from_config(
			session_config,
			"SELECTED_OPTIMIZER"
		)
		
		# end __EXEC_MODE_TRAIN

	#	-----------------------------------------------------------------	#
	#	Scelta dello scheduler di apprendimento.

	max_lr = get_value_from_config(session_config, "MAX_LR")
	start_factor = get_value_from_config(session_config, "START_FACTOR")
	step_size = get_value_from_config(session_config, "STEP_SIZE")
	steplr_gamma = get_value_from_config(session_config, "STEPLR_GAMMA")

	if not __EXEC_MODE_TRAIN:
		print(f"EXEC_MODE_TRAIN impostato su False.")
		print("La selezione dello scheduler è irrilevante in questa modalità.")
		selected_scheduler = None

	elif __EXEC_MODE_TRAIN:
		selected_scheduler = get_value_from_config(
			session_config,
			"SELECTED_SCHEDULER"
		)
		
		# end __EXEC_MODE_TRAIN

	#	-----------------------------------------------------------------	#
	#	Scelta della funzione di perdita.

	alpha = get_value_from_config(session_config, "ALPHA")
	focal_loss_gamma = get_value_from_config(session_config, "FOCAL_LOSS_GAMMA")

	if not __EXEC_MODE_TRAIN:
		print(f"EXEC_MODE_TRAIN impostato su False.")
		print("La selezione della loss function è irrilevante in questa modalità.")
		selected_loss = None

	elif __EXEC_MODE_TRAIN:
		selected_loss = get_value_from_config(
			session_config,
			"SELECTED_LOSS"
		)
		
		# end __EXEC_MODE_TRAIN

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

	dataset_name = get_value_from_config(session_config, "DATASET_NAME")
	batch_size = get_value_from_config(session_config, "BATCH_SIZE")

	epochs = get_value_from_config(session_config, "EPOCHS")
	learning_rate = get_value_from_config(session_config, "LEARNING_RATE")
	early_stopping = get_value_from_config(session_config, "EARLY_STOPPING")
	patience = get_value_from_config(session_config, "PATIENCE")

	for fold_index, current_fold in enumerate(folds):

		fold_start_time = time.time()
		print(f"\nInizio elaborazione del fold {fold_index + 1}/{len(folds)}")
		print(f"Orario corrente: {datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")

		#	-----------------------------------------------------------------	#
		#	Creazione dei percorsi dei file di output.

		fold_results_dir = get_fold_directory(fold_index)
		print(f"Directory dei risultati per il fold selezionato: {fold_results_dir}")

		fold_confusion_matrix_path = fold_results_dir / "confusion_matrix.png"
		fold_history_path = fold_results_dir / "training_history.csv"
		fold_model_path = fold_results_dir / "model.pth"
		fold_plots_path = fold_results_dir / "train_val_plots.png"
		fold_predictions_path = fold_results_dir / "predictions_test.csv"
		fold_report_path = fold_results_dir / "classification_report.txt"

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

		#	-----------------------------------------------------------------	#
		#	Applicazione delle strategia di data preprocessing.

		if not use_flowpic_dataset:

			scaler.fit(np.reshape(X_train, [-1, get_value_from_config(session_config, "N_FEATURES")]))

			X_train = apply_data_preprocessing(X_train, scaler)
			X_val = apply_data_preprocessing(X_val, scaler)
			X_test = apply_data_preprocessing(X_test, scaler)

			print("Train shape:", X_train.shape)
			print("Val shape:", X_val.shape)
			print("Test shape:", X_test.shape)

		#	-------------------------------------------------------------	#
		#	Costruzione dell'oggetto Dataset per PyTorch.

		dtype = torch.float32 if device.type == "mps" else torch.float64

		train_dataset = TrafficDataset(X_train, y_train, num_classes, dtype)
		val_dataset = TrafficDataset(X_val, y_val, num_classes, dtype)
		test_dataset = TrafficDataset(X_test, y_test, num_classes, dtype)

		#	-------------------------------------------------------------	#
		#	Costruzione dei DataLoader per PyTorch.

		train_loader = DataLoader(
			train_dataset, batch_size,
			num_workers=0,
			shuffle=True, drop_last=True,
			pin_memory=(device.type == "cuda"),
		)

		val_loader = DataLoader(
			val_dataset, batch_size,
			num_workers=0,
			shuffle=False, drop_last=False,
			pin_memory=(device.type == "cuda"),
		)

		test_loader = DataLoader(
			test_dataset, batch_size,
			num_workers=0,
			shuffle=False, drop_last=False,
			pin_memory=(device.type == "cuda"),
		)

		#	-------------------------------------------------------------	#
		#	Inizializzazione dei pesi del modello.

		model.apply(weight_init)

		print(f"\nEXEC_MODE_TRAIN impostato su {__EXEC_MODE_TRAIN}.")
		if __EXEC_MODE_TRAIN:
			print(
				"Totale parametri addestrabili: "
				f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
			)

		elif not __EXEC_MODE_TRAIN:
			print(f"Caricamento modello per la valutazione da: {fold_results_dir}")
			print("Il modello non verrà addestrato, ma solo valutato sul test set.")

			# Si caricano i pesi salvati del modello dalla directory di output specificata.
			weights = torch.load(fold_model_path, map_location=device)

			# Verifica la compatibilità tra il modello corrente e i pesi salvati.
			check_model_compatibility(model, weights)

			# Carica i pesi nel modello corrente e stampa l'esito del caricamento.
			load_result = model.load_state_dict(weights)
			print(f"\nEsito caricamento: {load_result}")
			print("Modello caricato correttamente.\n")

			# Si imposta il modello in modalità di valutazione
			# per disabilitare il dropout e la batch normalization.
			model.eval()

			# Si recupera lo storico delle metriche di addestramento e validazione dal file CSV salvato.
			df_history = pd.read_csv(fold_history_path)
			history = df_history.to_dict(orient='list')

			# end __EXEC_MODE_TRAIN

		#	-------------------------------------------------------------	#
		#	Addestramento del modello.

		if not __EXEC_MODE_TRAIN:
			print(f"EXEC_MODE_TRAIN impostato su False.")
			print("Il modello non verrà addestrato.")
		
		elif __EXEC_MODE_TRAIN:

			#	-------------------------------------------------------------	#
			#	Inizializzazione delle metriche da monitorare.

			# Questo dizionario terrà traccia delle metriche di addestramento e validazione per ogni epoca.
			history = {
				'epoch': [],
				'time': [],
				'accuracy': [],
				'val_accuracy': [],
				'loss': [],
				'val_loss': [],
			}

			# Inizializza le variabili per il monitoraggio della miglior loss di validazione.
			best_val_loss = float('inf')
			patience_counter = 0

			#	-------------------------------------------------------------	#
			#	Configurazione dei componenti di addestramento.

			# Scelta dell'optimizer
			optimizer = build_optimizer(
				selected_optimizer, model,
				learning_rate, weight_decay,
				epsilon, momentum
			)

			# Scelta del learning rate scheduler
			lr_sched = build_scheduler(
				selected_scheduler, optimizer, train_loader,
				epochs, max_lr, start_factor,
				step_size, steplr_gamma
			)

			# Scelta della loss function
			criterion = build_loss_function(
				selected_loss, y_train, num_classes, device,
				next(model.parameters()).dtype,
				alpha, focal_loss_gamma
			)

			#	-------------------------------------------------------------	#
			#	Inizio del loop di addestramento.

			training_start_time = time.time()
			print(
				f"Addestramento iniziato.\n"
				f"Orario corrente: {datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}\n"
				f"[MODEL] {model.get_model_name()} | "
					f"{selected_loss} | "
					f"{selected_optimizer} | "
					f"{selected_scheduler}\n"
				f"[DATASET] {dataset_name}\n"
					f"Training samples: {len(train_loader.dataset)} | "
					f"Validation samples: {len(val_loader.dataset)} | "
					f"Batch size: {batch_size}"
			)

			#	-------------------------------------------------------------	#
			#	Esecuzione del loop di addestramento.

			for epoch in range(epochs):
				
				# Salva il timestamp di inizio epoca per calcolare la durata dell'epoca corrente.
				epoch_start_time = time.time()
				print(f"\nEpoch {epoch+1}/{epochs}\n")
				
				# Calcola la loss e l'accuratezza per il training set e il validation set.
				train_loss, train_acc = train_epoch(
					model, device,
					train_loader,
					criterion, optimizer, lr_sched
				)

				# In fase di addestramento, non serve restituire le predizioni e le etichette predette. Si inserisce '_' per ignorarle.
				val_loss, val_acc, _, _, _ = evaluate(
					model, device,
					val_loader,
					criterion
				)
				
				# Calcola la durata dell'epoca corrente e il tempo totale trascorso dall'inizio dell'addestramento.
				epoch_end_time = time.time()
				epoch_duration = epoch_end_time - epoch_start_time
				elapsed_time = epoch_end_time - training_start_time

				# Aggiorna lo storico delle metriche per l'epoca corrente.
				history['epoch'].append(epoch + 1)
				history['time'].append(epoch_duration)
				history['loss'].append(train_loss)
				history['accuracy'].append(train_acc)
				history['val_loss'].append(val_loss)
				history['val_accuracy'].append(val_acc)

				# Stampa in console le metriche dell'epoca corrente.
				print(
					f"\nLoss: {train_loss:.4f} | Acc: {train_acc:.2f}%\n"
					f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%\n"
					f"Epoch time: {epoch_duration:.2f}s | Elapsed time: {elapsed_time:.2f}s"
				)

				# CHECK : se la loss di validazione migliora,
				# salva il modello e resetta il contatore per la patience.
				if val_loss < best_val_loss:
					best_val_loss = val_loss
					torch.save(model.state_dict(), fold_model_path)
					patience_counter = 0
					print(f"  -> Validation loss improved. Model saved.")
				else:
					patience_counter += 1
					print(f"  -> No improvement. ", end="")
					print(f"Patience: {patience_counter} of {patience}" if True else "")

				# Se il contatore di patience raggiunge il limite
				# e l'early stopping è abilitato, si interrompe il training.
				if patience_counter >= patience and early_stopping:
					print("Early stopping triggered.")
					break

				# end for epoch

			#	-------------------------------------------------------------	#
			#	Fine del loop di addestramento.

			training_end_time = time.time()
			print("Addestramento completato.")
			print(f"Orario corrente: {datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
			print(f"Tempo di elaborazione: {training_end_time - training_start_time:.2f} secondi")

			#	-------------------------------------------------------------	#
			#	Salvataggio dei risultati.

			# Salva i pesi del modello con la miglior loss di validazione.
			best_model = torch.load(
				fold_model_path,
				map_location=device,
				weights_only=True,
			)
			model.load_state_dict(best_model)
			del best_model

			# Salva lo storico delle metriche di addestramento e validazione in un file CSV.
			df_history = pd.DataFrame(history)
			df_history.to_csv(fold_history_path, index=False)
			print(f"Storico delle metriche salvato in {fold_history_path}")

			# Salva i pesi del modello addestrato in un file .PTH
			torch.save(model.state_dict(), fold_model_path)
			print(f"Modello salvato in {fold_model_path}")

			#	-----------------------------------------------------------------	#
			#	Costruzione e salvataggio del soft-output per il Test set.

			# Keys: [FlowID, Actual, Predicted, Soft-output]
			predictions_df = compute_soft_output(
				model, device, test_loader, criterion
			)

			if __SAVE_OUTPUT:
				predictions_df.to_csv(
					fold_predictions_path,
					index=False, sep=",",
					encoding="utf-8", lineterminator="\n",
				)

			#	-----------------------------------------------------------------	#
			#	Disegno dei grafici di loss / accuracy su Train / Val.

			# Calcolo dell'epoca con validation loss minima per evidenziarla nel grafico.
			best_epoch = int(np.argmin(history['val_loss'])) +1

			plt.figure(figsize=(12, 5))
			loss_plot = plt.subplot(1, 2, 1)
			acc_plot = plt.subplot(1, 2, 2)

			draw_train_val_plot(
				loss_plot, 'Loss',
				history['loss'], history['val_loss'],
				best_epoch
			)
			draw_train_val_plot(
				acc_plot, 'Accuracy',
				history['accuracy'], history['val_accuracy'],
				best_epoch
			)
			plt.tight_layout()

			if __SAVE_OUTPUT:
				plt.savefig(fold_plots_path, bbox_inches='tight')

			#	-----------------------------------------------------------------	#
			#	Calcolo della matrice di confusione per il Test set.

			cm = confusion_matrix(
				predictions_df["Actual"],
				predictions_df["Predicted"],
				normalize='true'
			)

			plt.figure(figsize=(10, 8))
			plt.title("Confusion Matrix")
			plt.ylabel('True Label')
			plt.xlabel('Predicted Label')

			draw_confusion_matrix(cm, labels_map.values())
			plt.tight_layout()

			if __SAVE_OUTPUT:
				plt.savefig(fold_confusion_matrix_path, bbox_inches='tight')

			#	-----------------------------------------------------------------	#
			#	Report di classificazione completo.

			report_dict = classification_report(
				predictions_df["Actual"], predictions_df["Predicted"],
				target_names=labels_map.values(), labels=np.arange(num_classes),
				digits=4, zero_division=0
			)

			if __SAVE_OUTPUT:
				with open(fold_report_path, "w") as f:
					f.write(report_dict)

		#	-------------------------------------------------------------	#
		#	Fine dell'elaborazione del fold corrente.

		fold_end_time = time.time()
		print(f"\nFine elaborazione del fold {fold_index + 1}/{len(folds)}")
		print(f"Orario corrente: {datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
		print(f"Tempo di elaborazione: {fold_end_time - fold_start_time:.2f} secondi")

		print(f"\n#  {'-'*20}  #")

		# end for fold_index, current_fold

	#	-----------------------------------------------------------------	#
	#	title

	# TODO: creare una tabella di confronto tra modelli su fold diverse con F1-score e accuracy, media e deviazione standard.
	# ...

	#	-----------------------------------------------------------------	#
	#	Terminazione dell'elaborazione.

	total_end_time = time.time()
	print(f"\nElaborazione completata.")
	print(f"Orario corrente: {datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
	print(f"Tempo di elaborazione: {total_end_time - total_start_time:.2f} secondi")

	exit(0)

	# end
