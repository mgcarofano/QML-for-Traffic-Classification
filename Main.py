"""
	Main.py \n
	by Mario Gabriele Carofano

	Questo script rappresenta il punto di ingresso principale per l'esecuzione
	delle varie funzionalità del progetto QML-for-Traffic-Classification.
"""

#	####################################################################	#
#	Librerie

# Importing session parameters
from dataclasses import asdict
from Source.configuration import SessionConfig, RuntimeConfig, ExecutionConfig
from Source.configuration import parse_args

# Dataset loading and manipulation
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
from Source.traffic_converter import \
	get_flowpic_dir, \
	mirage_pickle_converter, \
	remove_padding
from Source.classes import TrafficDataset
from torch.utils.data import DataLoader

# Preprocessing
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

# Machine Learning and Quantum ML
import torch
import torch.nn as nn
from Source.models import _BaseModel
from Source.models import \
	select_model, \
	weight_init, \
	check_model_compatibility
from Source.training_functions import \
	train_epoch, \
	evaluate

# Visualization
import matplotlib.pyplot as plt
from Source.visualization_functions import \
	draw_quantum_circuit, \
	draw_train_val_plot, \
	draw_confusion_matrix
from sklearn.metrics import \
	f1_score, \
	confusion_matrix, \
	classification_report

# More libraries
import random
import json
import time
from datetime import datetime

#	####################################################################	#
#	Funzioni di utilità

def debug_print(message: str) -> None:
	"""Stampa un messaggio di debug se la modalità di debug è abilitata.

	Args:
		message (str): Messaggio da stampare.
	"""

	if runtime_config.flags.debug:
		print(f"\n[DEBUG] {message}")

	# end

def convert(seconds: float) -> str:
	# Separa i secondi interi dalla parte decimale
	sec_int = int(seconds)
	milliseconds = int(round((seconds - sec_int) * 1000))

	# Gestione dell'eventuale arrotondamento a 1000 ms
	if milliseconds >= 1000:
		sec_int += 1
		milliseconds -= 1000

	# Formatta ore, minuti e secondi con time.strftime usando l'orario UTC/epoch
	time_str = time.strftime("%H:%M:%S", time.gmtime(sec_int))

	return f"{time_str}.{milliseconds:03d}"

#	####################################################################	#
#	Dispositivo utilizzato per il calcolo.

def resolve_device(requested: str = "auto") -> torch.device:
	"""Calcola il dispositivo da utilizzare, risolvendo automaticamente in base alla configurazione della macchina corrente o validando un dispositivo esplicito.

	Args:
		requested (str): Nome del dispositivo richiesto. Se impostato su "auto", la funzione selezionerà automaticamente il dispositivo disponibile più appropriato sulla macchina corrente secondo l'ordine di preferenza: MPS, CUDA, CPU.

	Raises:
		RuntimeError: Se il dispositivo richiesto non è disponibile.
		NotImplementedError: Se il tipo di dispositivo richiesto non è supportato.

	Returns:
		torch.device: Il dispositivo da utilizzare per il calcolo.
	"""

	#	################################################################	#
	#	CASO SPECIALE: risoluzione automatica del dispositivo.

	if requested == "auto":
		if torch.backends.mps.is_available():
			return torch.device("mps")
		if torch.cuda.is_available():
			return torch.device("cuda", torch.cuda.current_device())
		
		return torch.device("cpu")

		# end if
	
	#	################################################################	#
	#	Validazione del dispositivo richiesto.

	device = torch.device(requested)

	if device.type == "mps":
		if not torch.backends.mps.is_available():
			raise RuntimeError("Dispositivo MPS non è disponibile sulla macchina corrente.")

	elif device.type == "cuda":
		if not torch.cuda.is_available():
			raise RuntimeError("CUDA non è disponibile sulla macchina corrente.")

		index = \
			device.index \
			if device.index is not None \
			else torch.cuda.current_device()

		count = torch.cuda.device_count()
		if not 0 <= index < count:
			raise RuntimeError(f"Dispositivo CUDA {index} non disponibile: la macchina espone {count} GPU CUDA.")

		device = torch.device("cuda", index)

	elif device.type != "cpu":
		raise NotImplementedError(
			f"Dispositivo non supportato: {device.type!r}."
			f"Dispositivi supportati: ['cpu', 'cuda', 'mps']."
		)

	return device

	# end

#	####################################################################	#
#	Caricamento o generazione (opzionale) del dataset.

def load_dataset() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
	"""Carica il dataset dal file PICKLE specificato.

	Se `USE_FLOWPIC_DATASET` è impostato su True nella configurazione della sessione,
	questa funzione dovrebbe caricare il dataset FlowPic precomputato dal file .npz corrispondente.

	Returns:
		tuple[np.ndarray, np.ndarray, pd.DataFrame]: Una tupla contenente i dati dei flussi (X_raw), le etichette (y_raw) e i relativi metadati.
	"""

	#	################################################################	#
	#	Inizializzazione.

	dataset_path = Path(f"Dataset/{session_config.dataset.path}/{session_config.dataset.name}")

	#	################################################################	#
	#	Elaborazione del dataset in formato MIRAGE (sequenza di pacchetti).

	if not session_config.dataset.is_flowpic:

		# Il file pickle contiene due oggetti salvati in sequenza:
		# 1. X_raw : i dati numerici dei flussi di traffico, in formato numpy array.
		# 2. y_raw : le etichette corrispondenti ai flussi.
		# Ogni chiamata a pickle.load() legge il successivo oggetto nel file.
		with open(dataset_path.with_suffix(".pickle"), "rb") as f:
			X_raw = np.array(pickle.load(f), dtype=np.float32)
			y_raw = np.array(pickle.load(f))

		print(f"\nCaricato dataset (shape={X_raw.shape}) da: {dataset_path}")

		metadata = pd.DataFrame({
			"FlowID": np.arange(len(X_raw)),
			"DatasetID": np.arange(len(X_raw)),
			"Label": y_raw,
			"RealPackets": np.array([len(remove_padding(flow)[0]) for flow in X_raw]),
		})

		# # Caricamento dei metadati in formato CSV.
		# metadata_flowpic = pd.read_csv("/Users/mgcarofano/Documents/Magistrale UNINA/QML-for-Traffic-Classification/Dataset/flowpic/2019/mirage2019_LOPEZ_lopez_lopez_100P_4F_APP_xST_PAD_metadata.csv")

		# assert metadata.shape == metadata_flowpic.shape, \
		# 	"I metadati non hanno la stessa forma." \
		# 	f"Mirage: {metadata.shape}, Flowpic: {metadata_flowpic.shape}"
		# diff = metadata.compare(
		# 	metadata_flowpic,
		# 	result_names=("metadata", "metadata_flowpic"),
		# )

		# print(f"Numero di righe diverse: {len(diff)}")
		# print(diff)

	#	################################################################	#
	#	Elaborazione del dataset in formato FLOWPIC (istogrammi 2D).

	elif session_config.dataset.is_flowpic:

		flowpics_dir = get_flowpic_dir(session_config)
		flowpics_dir.mkdir(parents=True, exist_ok=True)

		npz_path = flowpics_dir / f"{session_config.dataset.name}.npz"
		meta_path = flowpics_dir / f"{session_config.dataset.name}.csv"

		if not npz_path.is_file() or not meta_path.is_file():
			# Se i file precomputati non esistono, vengono generati utilizzando mirage_pickle_converter().
			histograms, metadata = mirage_pickle_converter(session_config, runtime_config)

			# Salvataggio degli istogrammi 2D FlowPic in formato NumPy.
			np.savez_compressed(npz_path, data=histograms)
			print(f"\nSalvato dataset (shape={histograms.shape}) in: {npz_path}")

			# Salvataggio dei metadati in formato CSV.
			metadata.to_csv(
				meta_path,
				index=False,
				lineterminator="\n",
			)
			print(f"Salvati metadati (di {len(metadata)} righe) in: {meta_path} ")

		else:
			# Caricamento degli istogrammi 2D FlowPic precomputati.
			with np.load(npz_path, allow_pickle=False) as archive:
				histograms = archive["data"].astype(np.float32, copy=False)
			print(f"\nCaricato dataset (shape={histograms.shape}) da: {npz_path}")

			# Caricamento dei metadati in formato CSV.
			metadata = pd.read_csv(meta_path)
			print(f"Caricati metadati (di {len(metadata)} righe) da: {meta_path} ")

		# # Prima di procedere con split/training, si verifica che gli istogrammi e i metadati siano allineati.
		# if histograms.shape[0] != len(metadata):
		# 	raise ValueError("Disallineamento tra istogrammi e metadati!")

		X_raw = histograms
		y_raw = metadata["Label"].values

	# end if USE_FLOWPIC_DATASET

	#	################################################################	#
	#	Verifica della correttezza dei dati e dei metadati.

	if len(X_raw) == 0 or len(y_raw) == 0:
		raise ValueError("Errore nel caricamento del dataset: nessun campione disponibile.")

	height, width = session_config.model_input_shape

	if not session_config.dataset.is_flowpic:
		expected = (height, width)
		if X_raw.ndim != 3 or X_raw.shape[2] != width or X_raw.shape[1] < height:
			raise ValueError(
				f"Formato del dataset Mirage non valido. \n"
				f"Atteso: (N, T >= {expected}), "
				f"Ricevuto: {X_raw.shape}."
			)
		
		X_raw = np.ascontiguousarray(X_raw[:, :height, :], dtype=np.float32)
		if not np.isfinite(X_raw).all():
			raise ValueError("Errore nel calcolo del dataset Mirage: valori non finiti.")
		
	elif session_config.dataset.is_flowpic:
		expected = (1, height, width)
		if X_raw.ndim != 4 or tuple(X_raw.shape[1:]) != expected:
			raise ValueError(
				f"Formato del dataset Flowpic non valido. \n"
				f"Atteso: (N, {expected}), "
				f"Ricevuto: {X_raw.shape}."
			)
		
		if not np.isfinite(X_raw).all() or np.any(X_raw < 0):
			raise ValueError("Errore nel calcolo del dataset Flowpic: valori non finiti o negativi.")

		# end if

	if len(X_raw) != len(y_raw) or len(X_raw) != len(metadata):
		raise ValueError("Dati, etichette e metadati non sono allineati.")
	if list(metadata["DatasetID"]) != list(range(len(metadata))):
		raise ValueError("DatasetID non contiguo!")
	
	return X_raw, y_raw, metadata

	# end

#	####################################################################	#
#	Generazione delle etichette per le classi del dataset.

def encode_labels(X_raw, y_raw, labels_map_path):
	"""Codifica le etichette del dataset utilizzando LabelEncoder.

	Args:
		X_raw (array-like): Array contenente i dati grezzi del dataset.
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

	if not runtime_config.launch.use_config_file():
		with open(labels_map_path, "w", encoding="utf-8") as f:
			json.dump(labels_map, f)

	elif runtime_config.launch.use_config_file():
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

		# end use_config_file()

	if len(X_raw) != len(y_encoded):
		raise ValueError("Disallineamento tra dati e etichette!")

	return y_encoded, labels_map

#	####################################################################	#
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

	# Se il rapporto di sub-sampling è inferiore a 1.0, si riduce il dataset di conseguenza.
	if session_config.kfold.sub_sample_ratio < 1.0 and len(X_raw) > 0:
		selected_indices, _ = train_test_split(
			np.arange(len(X_raw)),
			train_size=session_config.kfold.sub_sample_ratio,
			stratify=y_encoded,
			random_state=runtime_config.launch.random_seed
		)

		X_raw = X_raw[selected_indices]
		y_encoded = y_encoded[selected_indices]
		metadata = metadata.iloc[selected_indices].copy().reset_index(drop=True)

		if len(X_raw) != len(y_encoded) or len(X_raw) != len(metadata):
			raise ValueError("Disallineamento dopo la riduzione del dataset!")

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

	#	################################################################	#
	#	Verifica dei parametri

	if session_config.kfold.test_folds <= 0:
		raise ValueError("Il numero di folds deve essere maggiore di 0!")
	if session_config.kfold.test_folds > len(X_raw):
		raise ValueError("Il numero di folds non può superare il numero di campioni nel dataset!")

	#	################################################################	#
	#	CASO SPECIALE: unico fold

	# Nel caso in cui si vuole una singola iterazione, si restituisce l'intero dataset come unico fold. Si effettua comunque la separazione in una porzione temporanea e una porzione dedicata al Test.
	if session_config.kfold.test_folds == 1:

		if session_config.kfold.test_ratio is None:
			raise ValueError("La dimensione del test set deve essere specificata quando si utilizza un unico fold.")

		train_indices, test_indices = train_test_split(
			np.arange(len(X_raw)),
			test_size=session_config.kfold.test_ratio,
			stratify=y_encoded,
			random_state=runtime_config.launch.random_seed
		)

		return [(train_indices, test_indices)]

	#	################################################################	#
	#	Suddivisione del dataset in folds.

	skf = StratifiedKFold(
		n_splits = session_config.kfold.test_folds,
		shuffle = True,
		random_state = runtime_config.launch.random_seed
	)

	folds = list(skf.split(X_raw, y_encoded))

	#	################################################################	#
	#	Verifica e ritorno dei risultati.

	if len(folds) == 0:
		raise ValueError("Nessun fold generato da StratifiedKFold!")
	if len(folds) != session_config.kfold.test_folds:
		raise ValueError("Il numero di folds generati non corrisponde al numero di test_folds specificato!")
	
	return folds

	# end

def split_train_val(train_indices_temp, y_temp) -> tuple:
	"""Suddivide il dataset temporaneo per il training in Train e Validation in base ai parametri della sessione corrente.

	Args:
		train_indices_temp (np.ndarray): Array contenente gli indici di training temporanei.
		y_temp (np.ndarray): Array contenente le etichette codificate corrispondenti agli indici di training temporanei.

	Returns:
		tuple: Una tupla contenente gli indici di training e validation suddivisi.
	"""

	#	################################################################	#
	#	Suddivisione del dataset in Train e Validation.

	val_fraction_of_temp = session_config.kfold.val_ratio / (session_config.kfold.train_ratio + session_config.kfold.val_ratio)
	train_indices, val_indices = train_test_split(
		train_indices_temp,
		test_size=val_fraction_of_temp,
		stratify=y_temp,
		random_state=runtime_config.launch.random_seed
	)

	return train_indices, val_indices

#	####################################################################	#
#	Applicazione della strategia di data preprocessing.

def apply_data_preprocessing(X, scaler, lengths):
	"""Applica il preprocessing ai dati utilizzando lo scaler specificato, lavorando solo sulle lunghezze valide dei campioni. Valido solo per dataset di tipo Mirage (sequenze di pacchetti).

	Args:
		X (np.ndarray): Array contenente i dati da preprocessare.
		scaler: Oggetto scaler utilizzato per normalizzare i dati.
		lengths (np.ndarray): Array contenente le lunghezze valide per ciascun campione.

	Returns:
		np.ndarray: Array contenente i dati preprocessati.
	"""

	height, width = session_config.model_input_shape
	X_proc = X[:, :height, :].astype(np.float32, copy=True)
	valid = np.arange(height)[None, :] < lengths[:, None]

	X_proc = scaler.transform(X_proc.reshape(-1, width)).reshape(-1, height, width)
	X_proc[~valid] = 0.0
	
	return X_proc.astype(np.float32, copy=False)

	# end

#	####################################################################	#
#	Creazione dei percorsi dei file di output.

def get_fold_directory(output_dir: Path, fold_index: int) -> Path:
	"""Crea e restituisce il percorso della directory dei risultati per il fold selezionato in modalità di addestramento, altrimenti verifica che esista (in base alla directory di output fornita nel file di configurazione).

	Args:
		output_dir (Path): Directory di output principale.
		fold_index (int): Indice del fold selezionato.

	Raises:
		FileNotFoundError: Se la directory dei risultati per il fold selezionato non viene trovata.

	Returns:
		Path: Percorso della directory dei risultati per il fold selezionato.
	"""

	#	################################################################	#
	#	Verifica dei parametri.

	if not 0 <= fold_index < session_config.kfold.test_folds:
		raise ValueError(f"fold_index deve essere compreso tra 0 e {session_config.kfold.test_folds-1}.")

	#	################################################################	#
	#	Creazione della directory dei risultati per il fold selezionato.

	fold_results_dir = output_dir / f"fold{fold_index}"

	if runtime_config.launch.exec_mode_train:
		fold_results_dir.mkdir(parents=True, exist_ok=True)

	elif not runtime_config.launch.exec_mode_train:
		if not fold_results_dir.exists():
			raise FileNotFoundError(
				f"Directory dei risultati per il fold {fold_index} "
				f"non trovata: {fold_results_dir}"
			)

		# end if

	return fold_results_dir

	# end

#	####################################################################	#
#	Configurazione dei componenti di addestramento.

def init_model(device, fold_index: int, num_classes: int) -> _BaseModel:
	"""Inizializza e restituisce il modello selezionato.

	Args:
		device: Dispositivo su cui eseguire il modello (CPU, GPU o MPS).
		fold_index (int): Indice del fold corrente per la cross-validation.
		num_classes (int): Numero di classi per la classificazione.

	Returns:
		_BaseModel: Il modello inizializzato.
	"""

	#	################################################################	#
	#	Configurazione del seed per la riproducibilità.

	os.environ['PYTHONHASHSEED'] = str(runtime_config.launch.random_seed)

	# Si calcola il seed specifico per il fold corrente.
	fold_seed = runtime_config.launch.random_seed + fold_index

	random.seed(fold_seed)
	np.random.seed(fold_seed)

	torch.manual_seed(fold_seed)
	torch.cuda.manual_seed(fold_seed)
	torch.cuda.manual_seed_all(fold_seed)

	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

	#	################################################################	#
	#	Scelta e istanziazione del modello.

	model : _BaseModel = select_model(
		num_classes, fold_seed,
		session_config
	)

	# Scelta del tipo di dato appropriato in base al dispositivo.
	model = model.float() if device.type == "mps" else model.double()

	# Inizializzazione dei pesi del modello su CPU.
	model.apply(weight_init)

	# Trasferimento del modello sul dispositivo selezionato.
	model = model.to(device)

	return model

	# end

def build_optimizer(model):
	"""Costruisce l'ottimizzatore in base alla selezione dell'utente.

	Args:
		model (nn.Module): Modello da ottimizzare.

	Raises:
		ValueError: Se il tipo di ottimizzatore selezionato non è supportato.

	Returns:
		L'ottimizzatore costruito in base alla selezione dell'utente.
	"""

	def get_optimizer_params(names: list):
		ret = {}

		if 'lr' in names:
			ret['lr'] = session_config.optimizer.learning_rate
		if 'weight_decay' in names:
			ret['weight_decay'] = session_config.optimizer.weight_decay
		if 'eps' in names:
			ret['eps'] = session_config.optimizer.epsilon
		if 'momentum' in names:
			ret['momentum'] = session_config.optimizer.momentum
		if 'betas' in names:
			ret['betas'] = (
				session_config.optimizer.beta1,
				session_config.optimizer.beta2
			)
		if 'alpha' in names:
			ret['alpha'] = session_config.optimizer.alpha

		return ret

		# end

	if session_config.optimizer.name == 'Adam':
		optimizer = torch.optim.Adam(
			model.parameters(),
			**get_optimizer_params(['lr', 'eps', 'betas']),
		)

	elif session_config.optimizer.name == 'AdamW':
		optimizer = torch.optim.AdamW(
			model.parameters(),
			**get_optimizer_params(['lr', 'weight_decay', 'eps', 'betas']),
		)

	elif session_config.optimizer.name == 'SGD':
		optimizer = torch.optim.SGD(
			model.parameters(),
			**get_optimizer_params(['lr', 'weight_decay', 'momentum'])
		)

	elif session_config.optimizer.name == 'RMSprop':
		optimizer = torch.optim.RMSprop(
			model.parameters(),
			**get_optimizer_params(['lr', 'weight_decay', 'eps', 'momentum', 'alpha'])
		)

	else:
		# Controllo già eseguito in configuration.py
		pass

	return optimizer

	# end

def build_scheduler(optimizer, train_loader):
	"""Costruisce lo scheduler di apprendimento in base alla selezione dell'utente.

	Args:
		optimizer (torch.optim.Optimizer): Ottimizzatore da associare allo scheduler.
		train_loader (torch.utils.data.DataLoader): DataLoader del set di addestramento.

	Raises:
		ValueError: Se il tipo di scheduler selezionato non è supportato.

	Returns:
		Lo scheduler di apprendimento costruito in base alla selezione dell'utente,
		oppure None se 'NoSched' è selezionato.
	"""

	if session_config.scheduler.name == 'OneCycleSched':
		lr_sched = torch.optim.lr_scheduler.OneCycleLR(
			optimizer,
			max_lr=session_config.scheduler.max_lr,
			steps_per_epoch=len(train_loader),
			epochs=session_config.training.epochs
		)

	elif session_config.scheduler.name == 'LinearSched':
		lr_sched = torch.optim.lr_scheduler.LinearLR(
			optimizer,
			start_factor=session_config.scheduler.start_factor,
			total_iters=session_config.training.epochs
		)

	elif session_config.scheduler.name == 'StepSched':
		lr_sched = torch.optim.lr_scheduler.StepLR(
			optimizer,
			step_size=session_config.scheduler.step_size,
			gamma=session_config.scheduler.gamma
		)

	elif session_config.scheduler.name == 'NoSched':
		lr_sched = None

	else:
		# Controllo già eseguito in configuration.py
		pass

	return lr_sched

	# end

def build_loss_function(
		y_train : torch.Tensor,
		num_classes : int,
		device : torch.device,
		dtype: torch.dtype,
	) -> nn.Module:
	"""Costruisce la funzione di loss in base alla selezione dell'utente.

	Args:
		y_train (torch.Tensor): Tensor contenente le etichette di addestramento.
		num_classes (int): Numero totale di classi.
		device (torch.device): Il device su cui eseguire i calcoli.
		dtype (torch.dtype): Il tipo di dato da utilizzare per i tensori.

	Raises:
		ValueError: Se il tipo di loss selezionato non è supportato.
		ValueError: Se il tipo di alpha per la Focal Loss non è supportato.

	Returns:
		nn.Module: La funzione di loss costruita.
	"""

	def compute_class_weights() -> torch.FloatTensor:
		""" Calcola i pesi di classe per il bilanciamento del dataset. """

		y = np.asarray(y_train.cpu() if torch.is_tensor(y_train) else y_train).astype(int)
		counts = np.bincount(y, minlength=num_classes)
		weights = np.where(counts > 0, len(y) / (num_classes * np.maximum(counts, 1)), 1.0)

		return torch.tensor(
			weights,
			dtype=dtype,
			device=device
		)

		# end

	if session_config.loss.name == "CrossEntropy":
		return nn.CrossEntropyLoss()

	elif session_config.loss.name == "WeightedCrossEntropy":
		w = compute_class_weights().to(device)
		return nn.CrossEntropyLoss(weight=w)

	elif session_config.loss.name == "Focal":
		if session_config.loss.alpha == "class_weights":
			alpha_tensor = compute_class_weights()
		elif session_config.loss.alpha == "uniform":
			alpha_tensor = torch.ones(
				num_classes,
				dtype=dtype,
				device=device
			)
		else:
			# Controllo già eseguito in configuration.py
			pass

		return torch.hub.load(
			'adeelh/pytorch-multi-class-focal-loss',
			model='FocalLoss',
			alpha=alpha_tensor, gamma=session_config.loss.focal_gamma,
			reduction='mean',
			force_reload=False
		)

	else:
		# Controllo già eseguito in configuration.py
		pass

	# end

#	####################################################################	#
#	# Stampa delle informazioni sul modello e sui parametri.

def get_model_info(model) -> str:
	"""Restituisce le informazioni sul modello, tra cui il nome della classe del modello, il numero di parametri addestrabili e la struttura interna dei suoi livelli.

	Args:
		model: Il modello di cui ottenere le informazioni.

	Returns:
		str: Le informazioni sul modello come stringa.
	"""

	params_df = pd.DataFrame([
		(name, tuple(p.shape), p.numel())
		for name, p in model.named_parameters()
	], columns=["Name", "Shape", "Params"])

	return (
		"Informazioni sul dataset: \n"
		f"{session_config.dataset!r} | Classes: {model.num_classes} \n\n"

		"Modello selezionato: \n"
		f"{session_config.model!r} \n"
		f"Model input: {model.input_kind}, {model.input_height}x{model.input_width} \n\n"

		"Struttura interna: \n"
		f"{str(model)} \n\n"

		f"Numero totale di parametri addestrabili: {params_df['Params'].sum():,} \n"
		"Parametri dei livelli: \n"
		f"{params_df.to_string(index=False)} \n"
	)

	# end

#	####################################################################	#
#	Addestramento del modello.

def train_model(
		model, device,
		train_loader, val_loader,
		optimizer, lr_sched, criterion,
		fold_model_path: str,
	) -> tuple[torch.nn.Module, pd.DataFrame, int, float]:
	"""Addestra il modello per un numero specificato di epoche.

	## Funzionalità
		- Monitora le metriche di addestramento e validazione.
		- Implementa early stopping e aggiorna i pesi del modello in base alla metrica di valutazione selezionata.

	Args:
		model: Il modello da addestrare.
		device: Il dispositivo su cui eseguire l'addestramento.
		train_loader: Il DataLoader per il set di addestramento.
		val_loader: Il DataLoader per il set di validazione.
		optimizer: L'ottimizzatore da utilizzare.
		lr_sched: Il scheduler del learning rate.
		criterion: La funzione di loss.
		fold_model_path (str): Il percorso dove viene salvato il modello aggiornato.
	
	Returns:
		tuple: il modello addestrato, il DataFrame contenente lo storico dell'addestramento, l'epoca migliore e il punteggio migliore.
	"""

	#	################################################################	#
	#	Inizializzazione delle metriche da monitorare.

	# Questo dizionario terrà traccia delle metriche di addestramento e validazione per ogni epoca.
	train_history_df = pd.DataFrame(columns=[
		"epoch", "time",
		"train_loss", "train_accuracy",
		"val_loss", "val_accuracy", "val_macrof1",
	])

	# Inizializza le variabili per il monitoraggio della metrica di valutazione selezionata.
	minimize = session_config.training.monitored_metric == "val_loss"
	best_score = float('inf') if minimize else -float('inf')
	best_epoch = None
	patience_counter = 0

	#	################################################################	#
	#	Inizio del loop di addestramento.

	training_start_time = time.time()
	print(
		f"Addestramento iniziato.\n"
		"Orario corrente: "
		f"{datetime.fromtimestamp(training_start_time).strftime('%Y-%m-%d_%H:%M:%S')}\n\n"

		f"{repr(session_config)} \n"
		f"[SAMPLES] Training: {len(train_loader.dataset)} in {len(train_loader)} batches"
		f" | Validation: {len(val_loader.dataset)} in {len(val_loader)} batches"
		f" | Classes: {model.num_classes} \n"
	)

	#	################################################################	#
	#	Esecuzione del loop di addestramento.

	for epoch in range(1, session_config.training.epochs +1):

		# Salva il timestamp di inizio epoca per calcolare la durata dell'epoca corrente.
		epoch_start_time = time.time()
		print(f"\nEpoch {epoch}/{session_config.training.epochs}\n")

		# Calcola la loss e l'accuratezza per il training set e il validation set.
		train_loss, train_acc = train_epoch(
			model, device,
			train_loader,
			criterion, optimizer, lr_sched
		)

		# In fase di addestramento, non serve restituire le predizioni e le etichette predette. Si inserisce '_' per ignorarle.
		val_loss, val_acc, val_preds, val_true, _ = evaluate(
			model, device,
			val_loader,
			criterion
		)

		val_macrof1 = f1_score(
			val_true.detach().cpu().numpy(),
			val_preds.detach().cpu().numpy(),
			labels= np.arange(model.num_classes),
			average= "macro",
			zero_division= 0,
		) * 100

		metrics = {
			"train_loss": float(train_loss),
			"train_accuracy": float(train_acc),
			"val_loss": float(val_loss),
			"val_accuracy": float(val_acc),
			"val_macrof1": float(val_macrof1),
		}

		invalid_metrics = [name for name, value in metrics.items() if not np.isfinite(value)]
		if invalid_metrics:
			raise RuntimeError("Errore nel calcolo delle metriche di addestramento: valori non finiti per [" + ", ".join(invalid_metrics) + "]")

		# Calcola la durata dell'epoca corrente e il tempo totale trascorso dall'inizio dell'addestramento.
		epoch_end_time = time.time()
		epoch_duration = epoch_end_time - epoch_start_time
		elapsed_time = epoch_end_time - training_start_time

		# Aggiorna lo storico delle metriche per l'epoca corrente.
		train_history_df.loc[len(train_history_df)] = {
			"epoch": epoch,
			"time": epoch_duration,
			**metrics
		}

		# Stampa in console le metriche dell'epoca corrente.
		print(
			f"\nTrain Loss: {train_loss:.4f} "
			f"| Train Accuracy: {train_acc:.4f}%\n"

			f"Val Loss: {val_loss:.4f} "
			f"| Val Accuracy: {val_acc:.4f}% "
			f"| Val Macro F1: {(val_macrof1):.4f}%\n"

			f"Epoch time: {convert(epoch_duration)} "
			f"| Elapsed time: {convert(elapsed_time)}"
		)

		# CHECK : se il punteggio calcolato migliora, salva il modello e resetta il contatore per la patience.
		current_score = metrics[session_config.training.monitored_metric]

		if best_epoch is None:
			improved = True
		elif minimize:
			improved = current_score < (best_score - session_config.training.min_delta)
		else:
			improved = current_score > (best_score + session_config.training.min_delta)

		if improved:
			print(
				f"  -> '{session_config.training.monitored_metric}' improved. \n"
				f"     Model saved in {fold_model_path}."
			)
			torch.save(model.state_dict(), fold_model_path)

			best_score = current_score
			best_epoch = epoch
			patience_counter = 0

		elif session_config.training.early_stopping:
			patience_counter += 1
			print(f"  -> No improvement on '{session_config.training.monitored_metric}'.")
			print(f"     Patience: {patience_counter} of {session_config.training.patience}" if True else "")

		else:
			print(
				f"  -> No improvement on '{session_config.training.monitored_metric}'. \n" 
				f"     Early stopping is disabled."
			)

		# Se il contatore di patience raggiunge il limite
		# e l'early stopping è abilitato, si interrompe il training.
		if patience_counter >= session_config.training.patience and session_config.training.early_stopping:
			print("     Early stopping triggered.")
			break

		# end for epoch

	#	################################################################	#
	#	Fine del loop di addestramento.

	training_end_time = time.time()
	best_score_str = \
		f"{best_score:.4f}" \
		if session_config.training.monitored_metric == "val_loss" \
		else f"{best_score:.4f}%"

	print(
		"\nAddestramento completato. \n"
		f"Orario corrente: "
		f"{datetime.fromtimestamp(training_end_time).strftime('%Y-%m-%d_%H:%M:%S')} \n"
		f"Tempo di elaborazione: "
		f"{convert(training_end_time - training_start_time)} \n"
		f"Migliore epoca: {best_epoch} | "
		f"{session_config.training.monitored_metric}: {best_score_str}"
	)

	#	################################################################	#
	#	Salvataggio dei risultati.

	# Ricarica i pesi del modello con la miglior loss di validazione.
	best_model = torch.load(
		fold_model_path,
		map_location=device,
		weights_only=True,
	)
	model.load_state_dict(best_model)
	del best_model

	return model, train_history_df, best_epoch, best_score

	# end

#	####################################################################	#
#	Costruzione del soft-output per il Test set.

def compute_soft_output(
		model, device,
		test_loader, metadata_test,
		criterion,
	):
	"""Calcola il soft-output per il test set, restituendo un Dataframe contenente il FlowID, i risultati effettivi, le predizioni e il soft-output.

	Args:
		model: Il modello da valutare.
		device: Il dispositivo su cui eseguire il calcolo.
		test_loader: Il DataLoader per il test set.
		metadata_test: Il DataFrame contenente i metadati del test set.
		criterion (opzionale): La funzione di loss da utilizzare.

	Returns:
		pd.DataFrame: Contiene il FlowID, i risultati effettivi, le predizioni e il soft-output.
	"""

	runtime_config.print_exec_mode()
	if not runtime_config.launch.exec_mode_train:
		print("Non è stato eseguito alcun addestramento.\n"
			"Il modello da valutare è stato caricato da un checkpoint salvato.\n")
		test_loss, test_acc, y_preds, y_true, y_probs = evaluate(
			model, device,
			test_loader,
		)

	elif runtime_config.launch.exec_mode_train:
		if criterion is None:
			raise ValueError("Funzione di loss mancante!")

		test_loss, test_acc, y_preds, y_true, y_probs = evaluate(
			model, device,
			test_loader,
			criterion
		)

		# end if

	# Calcola la loss e l'accuratezza per il test set, per valutare le prestazioni del modello su dati mai visti prima.
	print(f"Test Loss: {test_loss:.4f} | " if test_loss is not None else "", end="")
	print(f"Test Acc: {test_acc:.4f}%")

	#	################################################################	#
	#	Ritorno dei risultati.

	soft_output = [
		";".join(f"{probability:.8f}" for probability in row)
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

#	####################################################################	#
#	Main

def main(cli_overrides: bool):

	#	-----------------------------------------------------------------	#
	#	Inizio dell'elaborazione.

	total_start_time = time.time()
	print(f"\nInizio elaborazione.")
	print(f"Orario corrente: {datetime.fromtimestamp(total_start_time).strftime('%Y-%m-%d_%H:%M:%S')}")

	#	-----------------------------------------------------------------	#
	# 	Calcolo dei parametri di esecuzione.

	# Dispositivo utilizzato per il calcolo.
	device = resolve_device(runtime_config.launch.requested_device)
	print(f"\nDispositivo selezionato: {device.type}")
	if device.type == "cuda":
		print(f"GPU: {torch.cuda.get_device_name(0)}")

	# Tipo di dato utilizzato per il calcolo.
	dtype = torch.float32 if device.type == "mps" else torch.float64

	# Identificatore della sessione corrente.
	# Può identificare una nuova sessione, a partire dall'orario di lancio, oppure una sessione esistente se si utilizza un file di configurazione.
	session_id = \
		datetime.fromtimestamp(total_start_time).strftime("%Y-%m-%d_%H-%M") \
		if not runtime_config.launch.use_config_file() \
		else runtime_config.launch.source_session_id

	# Directory per i risultati
	output_dir = Path(
		"Results/"
		f"{session_config.dataset.path.replace('/', '-')}/"
		f"Sample{session_config.kfold.sub_sample_ratio * 100:g}%/"
		f"{session_config.model.name}/"
		f"{session_id}/"
	)

	# Percorso per il file di configurazione.
	config_path = output_dir / "config.json"

	model_info_path = output_dir / "model_info.txt"
	q_plot_path = output_dir / "quantum_circuit.png"

	#	-----------------------------------------------------------------	#
	#	Salvataggio di tutti i parametri di configurazione.

	execution_config = \
		ExecutionConfig.from_args(session_id, device.type, str(dtype)) \
		if not runtime_config.launch.use_config_file() \
		else ExecutionConfig.from_json(config_path)

	# Verifica della compatibilità con l'ambiente corrente.
	if runtime_config.launch.use_config_file():
		if execution_config.current_device != device.type or execution_config.dtype != str(dtype):
			raise RuntimeError(
				"La configurazione attuale della macchina "
				f"({(device.type, str(dtype))}) "
				"non è compatibile con quella salvata nel file di configurazione "
				f"({(execution_config.current_device, execution_config.dtype)})!"
			)

	# Se non si utilizza un file di configurazione, si salvano i parametri correnti.
	if not runtime_config.launch.use_config_file() or cli_overrides:

		# Per una nuova sessione la directory non deve già esistere.
		# Per una sessione caricata, invece, si riutilizza quella esistente.
		if not runtime_config.launch.use_config_file():
			if output_dir.exists():
				raise FileExistsError(f"La directory dei risultati '{output_dir}' esiste già. Aspetta un minuto prima di eseguire un nuovo esperimento.")
			output_dir.mkdir(parents=True, exist_ok=False)
		
		with config_path.open("w", encoding="utf-8") as file:
			json.dump(
				{
					"execution": asdict(execution_config),
					**asdict(runtime_config),
					**asdict(session_config),
				},
				file, indent=4, ensure_ascii=False, allow_nan=False,
			)

		# end use_config_file()

	del execution_config

	print(f"Directory dei risultati per la sessione corrente: {output_dir}")
	print(f"Percorso del file di configurazione per la sessione corrente: {config_path}")

	#	-----------------------------------------------------------------	#
	#	Caricamento o generazione (opzionale) del dataset.

	print("\nLoading dataset...")
	X_raw, y_raw, metadata = load_dataset()

	#	-----------------------------------------------------------------	#
	#	Generazione delle etichette per le classi del dataset.

	# Creazione del percorso di output per la mappatura delle etichette.
	labels_map_path = output_dir / "labels_map.json"

	y_encoded, labels_map = encode_labels(X_raw, y_raw, labels_map_path)
	num_classes = len(labels_map)

	if num_classes <= 0:
		raise EnvironmentError("Errore nel caricamento delle etichette delle classi.")

	print(f"\nNumero di classi: {num_classes}")
	print(f"Etichette delle classi generate: {sorted(labels_map.values())}\n")

	#	-----------------------------------------------------------------	#
	#	Selezione della strategia di data preprocessing.

	if session_config.dataset.is_flowpic:
		debug_print("La fase di preprocessing è disponibile solo per il dataset grezzo (Mirage).")

	elif not session_config.dataset.is_flowpic:
		if session_config.preprocessing.strategy == "MinMax":
			scaler = MinMaxScaler(feature_range=(0, 1))
		else:
			raise ValueError(f"Strategia di preprocessing non supportata: {session_config.preprocessing.strategy}")

		# end if

	#	-----------------------------------------------------------------	#
	#	Implementazione della K-fold cross-validation.

	# Riduzione del dataset (opzionale).
	X_raw, y_encoded, metadata = reduce_dataset(X_raw, y_encoded, metadata)

	# Suddivisione del dataset.
	folds = split_dataset(X_raw, y_encoded)

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

		fold_start_time = time.time()
		print(f"\nInizio elaborazione del fold {fold_index + 1}/{len(folds)}")
		print(f"Orario corrente: {datetime.fromtimestamp(fold_start_time).strftime('%Y-%m-%d_%H:%M:%S')}")

		#	-------------------------------------------------------------	#
		#	Creazione dei percorsi dei file di output.

		fold_results_dir = get_fold_directory(output_dir, fold_index)
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
		if len(X_train) + len(X_val) + len(X_test) != len(X_raw):
			raise ValueError("La somma dei campioni Train, Val e Test non corrisponde al totale!")

		all_indices = np.arange(len(X_raw))
		split_indices = np.concatenate([train_indices, val_indices, test_indices])
		if not np.array_equal(np.sort(all_indices), np.sort(split_indices)):
			raise ValueError("Gli split non coprono esattamente il dataset originale!")

		# Verifica dell'allineamento in ciascuno split.
		# TODO: vedi notebook.

		#	------------------------------------------------------------	#
		#	Estrazione delle lunghezze reali dei flussi (solo per dataset Mirage).

		lengths_train = None
		lengths_val = None
		lengths_test = None

		if not session_config.dataset.is_flowpic:
			real_lengths = metadata["RealPackets"].to_numpy(dtype=np.int64)

			lengths = np.minimum(real_lengths, session_config.dataset.n_packets)
			if np.any((lengths < 1) | (lengths > session_config.dataset.n_packets)):
				raise ValueError(f"Ogni lunghezza deve essere compresa tra 1 e {session_config.dataset.n_packets}.")
			
			lengths_train = lengths[train_indices]
			lengths_val = lengths[val_indices]
			lengths_test = lengths[test_indices]

		#	------------------------------------------------------------	#
		#	Applicazione della strategia di data preprocessing (solo per dataset Mirage).

		if not session_config.dataset.is_flowpic:
			valid_train = np.arange(X_train.shape[1])[None, :] < lengths_train[:, None]
			scaler.fit(X_train[valid_train])

			X_train = apply_data_preprocessing(X_train, scaler, lengths_train)
			X_val = apply_data_preprocessing(X_val, scaler, lengths_val)
			X_test = apply_data_preprocessing(X_test, scaler, lengths_test)

		#	------------------------------------------------------------	#
		#	Stampa dei risultati.

		if runtime_config.flags.debug:
			print(f"\nTrain shape: {X_train.shape} ({len(X_train)/len(X_raw):.1%})")
			print("Head:\n", metadata_train.head())
			print(f"\nValidation shape: {X_val.shape} ({len(X_val)/len(X_raw):.1%})")
			print("Head:\n", metadata_val.head())
			print(f"\nTest shape: {X_test.shape} ({len(X_test)/len(X_raw):.1%})")
			print("Head:\n", metadata_test.head())

		#	------------------------------------------------------------	#
		#	Costruzione dell'oggetto Dataset per PyTorch.

		train_dataset = TrafficDataset(
			X_train, y_train, lengths_train,
			num_classes, dtype
		)
		
		val_dataset = TrafficDataset(
			X_val, y_val, lengths_val,
			num_classes, dtype
		)
		
		test_dataset = TrafficDataset(
			X_test, y_test, lengths_test,
			num_classes, dtype
		)

		#	------------------------------------------------------------	#
		#	Costruzione dei DataLoader per PyTorch.

		train_loader = DataLoader(
			train_dataset, session_config.training.batch_size,
			num_workers=0,
			shuffle=True, drop_last=True,
			pin_memory=(device.type == "cuda"),
		)

		val_loader = DataLoader(
			val_dataset, session_config.training.batch_size,
			num_workers=0,
			shuffle=False, drop_last=False,
			pin_memory=(device.type == "cuda"),
		)

		test_loader = DataLoader(
			test_dataset, session_config.training.batch_size,
			num_workers=0,
			shuffle=False, drop_last=False,
			pin_memory=(device.type == "cuda"),
		)

		#	------------------------------------------------------------	#
		#	Verifica della lunghezza dei Loader.

		def check_loader_length(loader, name):
			if len(loader) == 0:
				raise ValueError(f"Il DataLoader per {name} è vuoto. Controllare la configurazione del dataset.")

		check_loader_length(train_loader, "il training")
		check_loader_length(val_loader, "la validazione")
		check_loader_length(test_loader, "il test")

		#	------------------------------------------------------------	#
		#	Inizializzazione del modello.

		model = init_model(device, fold_index, num_classes)

		# Stampa e salvataggio delle informazioni sul modello.
		if fold_index == 0:
			model_info = get_model_info(model)
			debug_print(model_info)
			with open(model_info_path, "w") as f:
				f.write(model_info)

		# Visualizzazione e salvataggio del circuito quantistico.
		if model.is_quantum and runtime_config.flags.save_output:
			q_plot = draw_quantum_circuit(model, session_config)
			
			q_plot.set_size_inches(6, 7.5)
			q_plot.suptitle("Quantum Circuit")
			
			q_plot.savefig(q_plot_path, bbox_inches='tight', pad_inches=0.2,)
			print(f"\nCircuito quantistico salvato in {q_plot_path}")

			if runtime_config.flags.debug:
				q_plot.show()

		#	------------------------------------------------------------	#
		#	Addestramento del modello.

		# Per motivi di compatibilità, 'criterion' si inizializza a None.
		# Si aggiorna successivamente se la modalità di addestramento è abilitata.
		criterion = None

		runtime_config.print_exec_mode()
		if not runtime_config.launch.exec_mode_train:
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

		elif runtime_config.launch.exec_mode_train:

			#	--------------------------------------------------------	#
			#	Configurazione dei componenti di addestramento.

			optimizer = build_optimizer(model)
			if optimizer is None:
				raise ValueError("L'ottimizzatore non è stato creato correttamente.")

			lr_sched = build_scheduler(optimizer, train_loader)
			if lr_sched is None:
				raise ValueError("Lo scheduler non è stato creato correttamente.")

			criterion = build_loss_function(
				y_train, num_classes, device,
				next(model.parameters()).dtype,
			)
			if criterion is None:
				raise ValueError("La funzione di loss non è stata creata correttamente.")

			#	--------------------------------------------------------	#
			#	Esecuzione del loop di addestramento.

			model, train_history_df, best_epoch, best_score = train_model(
				model, device,
				train_loader, val_loader,
				optimizer, lr_sched, criterion,
				fold_model_path,
			)

			# Verifica la correttezza della miglior epoca e del miglior punteggio.
			selected_row = train_history_df.loc[train_history_df["epoch"] == best_epoch]
			if len(selected_row) != 1:
				raise RuntimeError(
					f"L'epoca del checkpoint ({best_epoch}) "
					"non è identificata univocamente nello storico."
				)

			history_score = float(selected_row.iloc[0][session_config.training.monitored_metric])
			if not np.isclose(history_score, best_score):
				raise RuntimeError(
					f"Il punteggio nello storico ({history_score:.2f}) "
					f"non corrisponde al miglior punteggio ({best_score:.2f})."
				)
			
			# Salva lo storico delle metriche di addestramento e validazione in un file CSV.
			if runtime_config.flags.save_output:
				train_history_df.to_csv(
					fold_history_path,
					index=False,
					lineterminator="\n",
					float_format="%.8f"
				)
				print(f"\nStorico delle metriche salvato in {fold_history_path}")

			# end exec_mode_train

		#	------------------------------------------------------------	#
		#	Disegno dei grafici di loss / accuracy su Train / Val.

		if runtime_config.launch.exec_mode_train and runtime_config.flags.save_output:

			score_text = \
				f"{best_score:.2f}%" \
				if session_config.training.monitored_metric == "val_accuracy" \
				else f"{best_score:.4f}"

			fig, axes = plt.subplots(1, 3, figsize=(18, 5), layout='constrained')
			
			fig.suptitle(
				"Metriche di addestramento | "
				f"Checkpoint: epoch {best_epoch} | "
				f"{session_config.training.monitored_metric}: {score_text}"
			)

			draw_train_val_plot(
				axes[0], "Loss",
				train_history_df["train_loss"].to_list(),
				train_history_df["val_loss"].to_list(),
				best_epoch,
			)

			draw_train_val_plot(
				axes[1], "Accuracy (%)",
				train_history_df["train_accuracy"].to_list(),
				train_history_df["val_accuracy"].to_list(),
				best_epoch,
			)

			draw_train_val_plot(
				axes[2], "Macro-F1 (%)",
				None, train_history_df["val_macrof1"].to_list(),
				best_epoch,
			)

			plt.savefig(fold_plots_path, bbox_inches='tight')
			print(f"\nGrafici salvati in {fold_plots_path}")

			if runtime_config.flags.debug:
				plt.show()

			# end save_output

		#	------------------------------------------------------------	#
		#	Costruzione del soft-output per il Test set.

		# Keys: [FlowID, Actual, Predicted, Soft-output]
		test_predictions_df = compute_soft_output(
			model, device,
			test_loader, metadata_test,
			criterion,
		)

		if runtime_config.flags.save_output:
			test_predictions_df.to_csv(
				fold_predictions_path,
				index=False,
				lineterminator="\n",
				float_format="%.8f"
			)
			print(f"\nPredizioni salvate in {fold_predictions_path}")

		#	------------------------------------------------------------	#
		#	Calcolo della matrice di confusione per il Test set.

		if runtime_config.flags.save_output:
			cm = confusion_matrix(
				test_predictions_df["Actual"],
				test_predictions_df["Predicted"],
				normalize='true'
			)

			plt.figure(figsize=(10, 8))
			plt.tight_layout()
			plt.title("Confusion Matrix")
			plt.ylabel('True Label')
			plt.xlabel('Predicted Label')

			draw_confusion_matrix(cm, labels_map.values())
			plt.savefig(fold_confusion_matrix_path, bbox_inches='tight')
			print(f"\nMatrice di confusione salvata in {fold_confusion_matrix_path}")

			if runtime_config.flags.debug:
				plt.show()

			# end save_output

		#	-------------------------------------------------------------	#
		#	Report di classificazione completo.

		if runtime_config.flags.save_output:
			report_dict = classification_report(
				test_predictions_df["Actual"], test_predictions_df["Predicted"],
				target_names=labels_map.values(), labels=np.arange(num_classes),
				digits=4, zero_division=0
			)
			print(f"\nReport di classificazione salvato in {fold_report_path}")

			with open(fold_report_path, "w") as f:
				f.write(report_dict)

			# end save_output

		#	-------------------------------------------------------------	#
		#	Fine dell'elaborazione del fold corrente.

		fold_end_time = time.time()
		print(f"\nFine elaborazione del fold {fold_index + 1}/{len(folds)}")
		print(f"Orario corrente: {datetime.fromtimestamp(fold_end_time).strftime('%Y-%m-%d_%H:%M:%S')}")
		print(f"Tempo di elaborazione: {convert(fold_end_time - fold_start_time)}")

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
	print(f"Orario d'inizio: {datetime.fromtimestamp(total_start_time).strftime('%Y-%m-%d_%H:%M:%S')}")
	print(f"Orario di fine: {datetime.fromtimestamp(total_end_time).strftime('%Y-%m-%d_%H:%M:%S')}")
	print(f"Tempo di elaborazione: {convert(total_end_time - total_start_time)}")

	return

	# end

if __name__ == "__main__":

	#	-----------------------------------------------------------------	#
	#	Creazione degli oggetti con i parametri di sessione e runtime.

	global session_config
	global runtime_config

	# Lettura degli argomenti da linea di comando.
	args = parse_args()

	session_config = SessionConfig.from_args(args)
	runtime_config = RuntimeConfig.from_args(args)

	#	-----------------------------------------------------------------	#
	#	Verifica della corretta creazione degli oggetti di configurazione.
	
	if session_config is None:
		raise ValueError("Errore nel caricamento del file di configurazione (session)!")

	if runtime_config is None:
		raise ValueError("Errore nel caricamento del file di configurazione (runtime)!")

	cli_overrides = \
		set(vars(args)) - {
			"source_session_id",
			"exec_mode_train",
			"save_output"
		}

	if vars(args).get('source_session_id') is not None:
		if cli_overrides and vars(args).get('save_output'):
			raise ValueError(
				"Modalità di valutazione attivata.\n"
				"Non è possibile sovrascrivere parametri da CLI se 'save_output' è attivo.\n"
				f"Parametri da CLI: {cli_overrides}. "
			)
	
	# Stampa di debug dei parametri di configurazione.
	debug_print(
		f"Parametri di sessione in 'session_config':" \
		f"\n{session_config}" \
		"\n\n" \
		f"Parametri di runtime in 'runtime_config':" \
		f"\n{runtime_config}"
	)

	#	-----------------------------------------------------------------	#
	#	Funzione principale per l'esecuzione del programma.

	main(cli_overrides is not None)

	exit(0)

	# end
