"""
	configuration.py
	by Mario Gabriele Carofano

	Questo modulo definisce le classi di configurazione dove vengono centralizzati tutti i parametri di sessione. Tali parametri contengono le impostazioni specifiche per una sessione di esecuzione del progetto. Sono presenti funzioni di salvataggio e caricamento in file JSON, in modo da poter riprodurre gli esperimenti e confrontare i risultati.
"""

#	####################################################################	#
#	Librerie

from dataclasses import dataclass, fields, is_dataclass, replace
from argparse import ArgumentParser, Namespace, SUPPRESS, BooleanOptionalAction
from typing import get_args, get_origin, get_type_hints

from Source.models import MODEL_REGISTRY

from pathlib import Path
import json

#	####################################################################	#
#	Funzioni di utilità

def _overrides(
		config_class,
		args: Namespace,
		prefix: str = "",
	) -> dict:
	"""Restituisce un dizionario con i valori degli argomenti della CLI che sovrascrivono i campi della classe di configurazione.

	Args:
		config_class: La classe di configurazione da cui estrarre i campi.
		args (Namespace): Gli argomenti della CLI passati al programma.
		prefix (str, optional): Prefisso da aggiungere ai nomi degli argomenti. Defaults to "".

	Returns:
		dict
	"""
	values = vars(args)
	overrides = {}

	for field in fields(config_class):
		if not field.init:
			continue

		argument_name = (
			f"{prefix}.{field.name}"
			if prefix
			else field.name
		)

		if argument_name in values:
			overrides[field.name] = values[argument_name]

	return overrides

	# end

def parse_args(argv= None) -> Namespace:
	"""Analizza gli argomenti della CLI. Utilizza la funzione add_fields() per aggiungere gli argomenti in base alle classi di configurazione definite.

	Args:
		argv (list, optional): Lista degli argomenti della CLI da analizzare. Defaults to None.

	Raises:
		TypeError: Se il tipo del campo non è supportato dalla CLI.
		TypeError: Se il tipo opzionale contiene più di un tipo non nullo.
		TypeError: Se il tipo di tuple non è supportato.

	Returns:
		Namespace: oggetto contenente i valori analizzati della CLI.
	"""
	parser = ArgumentParser(
		description="Configurazione degli esperimenti QML.",
		argument_default=SUPPRESS,
		allow_abbrev=False,
	)

	def add_fields(config_class, prefix: str = "") -> None:
		"""Aggiunge gli argomenti della CLI per i campi della classe di configurazione.

		Args:
			config_class: La classe di configurazione da cui estrarre i campi.
			prefix (str, optional): Prefisso da aggiungere ai nomi degli argomenti. Defaults to "".

		Raises:
			TypeError: Se il tipo del campo non è supportato dalla CLI.
			TypeError: Se il tipo opzionale contiene più di un tipo non nullo.
			TypeError: Se il tipo di tuple non è supportato.
		"""
		type_hints = get_type_hints(config_class)

		for field in fields(config_class):
			if not field.init:
				continue

			argument_name = (
				f"{prefix}.{field.name}"
				if prefix
				else field.name
			)

			value_type = type_hints[field.name]

			# Le sezioni composte vengono visitate ricorsivamente.
			if is_dataclass(value_type):
				add_fields(value_type, argument_name)
				continue

			# Gestione dei tipi opzionali: int | None, ecc.
			nullable = type(None) in get_args(value_type)

			if nullable:
				non_null_types = [
					item
					for item in get_args(value_type)
					if item is not type(None)
				]

				if len(non_null_types) != 1:
					raise TypeError(
						"Tipo CLI non supportato per "
						f"'{argument_name}': {value_type}"
					)

				value_type = non_null_types[0]

			options = {
				"dest": argument_name,
				"default": SUPPRESS,
			}

			# Booleani: --debug / --no-debug.
			if value_type is bool:
				parser.add_argument(
					f"--{argument_name}",
					action=BooleanOptionalAction,
					**options,
				)

			# Caso attuale: flows_to_inspect.
			elif get_origin(value_type) is tuple:
				if get_args(value_type) != (int, Ellipsis):
					raise TypeError(
						"Tipo di tuple non supportato per "
						f"'{argument_name}'."
					)

				parser.add_argument(
					f"--{argument_name}",
					type=int,
					nargs="+",
					**options,
				)

			# Tipi scalari.
			elif value_type in (str, int, float):
				converter = value_type

				if nullable:
					def optional_value(
						value,
						convert=value_type,
					):
						if value.lower() in ("none", "null"):
							return None

						return convert(value)

					converter = optional_value

				parser.add_argument(
					f"--{argument_name}",
					type=converter,
					**options,
				)

			else:
				raise TypeError(
					"Tipo CLI non supportato per "
					f"'{argument_name}': {value_type}"
				)

			# end

	# Runtime
	# Non registriamo ExecutionConfig ed EnvironmentConfig nel parser: session ID corrente, device effettivo e versioni installate devono descrivere ciò che è stato realmente rilevato, non essere metadati arbitrariamente sovrascrivibili.
	add_fields(LaunchConfig)
	add_fields(FlagsConfig)

	# Session
	add_fields(SessionConfig)

	return parser.parse_args(argv)

	# end

def find_session_config(
		source_session_id: str,
		results_dir = "Results",
	) -> Path:
	"""Trova config.json nella cartella della sessione, cercando anche nelle sottocartelle.

	Args:
		source_session_id (str): Identificatore della sessione da cercare.
		results_dir (str, optional): Directory dei risultati in cui cercare. Defaults to "Results".

	Raises:
		FileNotFoundError: Se non viene trovata alcuna configurazione per la sessione specificata.
		ValueError: Se vengono trovate più configurazioni per la stessa sessione.

	Returns:
		Path: Percorso al file config.json della sessione trovata.
	"""
	
	results_dir = Path(results_dir)

	matches = [
		path
		for path in results_dir.rglob("config.json")
		if path.parent.name == source_session_id and path.is_file()
	]

	if not matches:
		raise FileNotFoundError(f"Configurazione della sessione '{source_session_id}' non trovata in '{results_dir}'.")

	if len(matches) > 1:
		paths = "\n".join(f"  - {path}" for path in matches)
		raise ValueError(
			f"Sono state trovate più configurazioni per la sessione '{source_session_id}':\n"
			f"{paths}"
		)

	return matches[0]

	# end

def check_positive_int(name: str, value):
	"""Verifica che il valore sia un intero positivo.

	Args:
		name (str): Il nome del parametro.
		value: Il valore da verificare.

	Raises:
		TypeError: Se il valore può essere convertito in un intero.
		ValueError: Se il valore non è un intero positivo.

	Returns:
		int: Il valore convertito in intero.
	"""
	
	from numbers import Integral

	if isinstance(value, bool) or not isinstance(value, Integral):
		raise TypeError(f"Il valore '{name}' non può essere convertito in un intero.")
	
	if value <= 0:
		raise ValueError(f"Il valore '{name}' deve essere positivo.")

	# end

#	####################################################################	#
#	Configurazione dei parametri di sessione.

@dataclass(frozen=True)
class DatasetConfig:
	path: str
	""" Path to the dataset file. """
	
	name: str
	""" Name of the dataset file. """
	
	is_flowpic: bool
	""" Whether to use the FlowPic dataset or the original Mirage dataset. If True, generate the FlowPic dataset from raw data using the `traffic_converter` module. """
	
	n_packets: int
	""" Number of packets to consider in each flow. """
	
	n_features: int
	""" Number of features for each packet in the dataset. """

	@classmethod
	def from_defaults(cls) -> "DatasetConfig":
		return cls(
			path= "mirage/2019",
			name= "mirage2019_LOPEZ_lopez_lopez_100P_4F_APP_xST_PAD_metadata",
			is_flowpic= False,
			n_packets= 100,
			n_features= 4,
		)

		# end

	@classmethod
	def from_args(cls, args: Namespace) -> "DatasetConfig":
		return replace(
			cls.from_defaults(),
			**_overrides(cls, args, "dataset"),
		)

	def __post_init__(self) -> None:
		check_positive_int('n_packets', self.n_packets)
		check_positive_int('n_features', self.n_features)

		if not self.path:
			raise ValueError("Percorso del dataset non definito.")
		if not self.name:
			raise ValueError("Nome del dataset non definito.")

		# end

	def __repr__(self) -> str:
		return \
			f"[DATASET] Path: {self.path}/{self.name}\n" \
			f"          FlowPic: {self.is_flowpic} | Dimension: {self.n_packets}x{self.n_features}"

		# end

	# end class

@dataclass(frozen=True)
class PreprocessingConfig:
	strategy: str
	""" Strategy for preprocessing the dataset. Default is 'MinMax'. """

	@classmethod
	def from_defaults(cls) -> "PreprocessingConfig":
		return cls(
			strategy= "MinMax",
		)

		# end

	@classmethod
	def from_args(cls, args: Namespace) -> "PreprocessingConfig":
		return replace(
			cls.from_defaults(),
			**_overrides(cls, args, "preprocessing"),
		)

	def __post_init__(self) -> None:
		if not self.strategy:
			raise ValueError("Strategia di preprocessing non definita.")

		if self.strategy not in ["MinMax"]:
			raise NotImplementedError(f"Strategia di preprocessing non supportata: {self.strategy}")

		# end

	def __repr__(self) -> str:
		return f"[PREPROCESSING] Strategy: {self.strategy}"

		# end

	# end class

@dataclass(frozen=True)
class TrafficFiltersConfig:
	min_tps: int | None
	""" Minimum duration of a session in seconds. Sessions shorter than this will be discarded. Default is 50 if not specified. """
	
	min_packets: int | None
	""" Minimum number of packets in a session. Sessions with fewer packets will be discarded. Default is 3 if not specified. """
	
	min_dim: int | None
	""" Minimum flow dimension (payload) in bytes. Flows smaller than this will be discarded. Default is 10000 if not specified. """

	@classmethod
	def from_defaults(cls) -> "TrafficFiltersConfig":
		return cls(
			min_tps= None,
			min_packets= None,
			min_dim= None,
		)

		# end

	@classmethod
	def from_args(cls, args: Namespace) -> "TrafficFiltersConfig":
		return replace(
			cls.from_defaults(),
			**_overrides(cls, args, "traffic.filters"),
		)

	def __post_init__(self) -> None:
		if self.min_tps is not None:
			check_positive_int('min_tps', self.min_tps)
		if self.min_packets is not None:
			check_positive_int('min_packets', self.min_packets)
		if self.min_dim is not None:
			check_positive_int('min_dim', self.min_dim)

		# end

	def __repr__(self) -> str:
		return f"min_tps={self.min_tps}, min_packets={self.min_packets}, min_dim={self.min_dim}"

		# end

	# end class

@dataclass(frozen=True)
class TrafficConfig:
	mtu: int
	""" Maximum Transmission Unit (Ethernet). """
	
	bin_size: int
	""" Dimension in byte of each bin on the Y-axis of the 2D histogram. Default is 10.
	This means that the Y-axis will be divided into 150 bins, each representing a range of 10 bytes. """

	filters: TrafficFiltersConfig
	""" A custom object containing the filtering criteria for traffic flows. """

	@classmethod
	def from_defaults(cls) -> "TrafficConfig":
		return cls(
			mtu= 1500,
			bin_size= 10,
			filters=TrafficFiltersConfig.from_defaults(),
		)

		# end

	@classmethod
	def from_args(cls, args: Namespace) -> "TrafficConfig":
		changes = _overrides(cls, args, "traffic")
		changes["filters"] = TrafficFiltersConfig.from_args(args)
		return replace(
			cls.from_defaults(),
			**changes,
		)

	def __post_init__(self) -> None:
		check_positive_int('mtu', self.mtu)
		check_positive_int('bin_size', self.bin_size)

		if self.bin_size > self.mtu:
			raise ValueError("La dimensione dei bin non può essere maggiore di MTU.")

		if self.mtu % self.bin_size != 0:
			raise ValueError("La dimensione dei bin deve dividere esattamente l'MTU.")

		# end

	def __repr__(self) -> str:
		return f"[TRAFFIC] MTU: {self.mtu} | Bin size: {self.bin_size} | Filters: {self.filters}\n)"

		# end

	# end class

@dataclass(frozen=True)
class CrossValidationConfig:
	train_ratio: float
	""" Proportion of the dataset to be used for training. Default is 0.8. """
	
	val_ratio: float
	""" Proportion of the dataset to be used for validation. Default is 0.2. """
	
	test_folds: int
	""" Number of folds for cross-validation. """
	
	test_ratio: float | None
	""" Proportion of the dataset to be used for testing. Default is 0.2 if not specified. """
	
	sub_sample_ratio: float
	""" Ratio of the dataset to be used for sub-sampling. For example, 0.5 means using 50% of the dataset. Default should be 1.0, meaning the entire dataset is used. """

	@classmethod
	def from_defaults(cls, test_folds= 5) -> "CrossValidationConfig":
		return cls(
			train_ratio= 0.8,
			val_ratio= 0.2,
			test_folds= 5,
			test_ratio= None,
			sub_sample_ratio= 1.0,
		)

		# end

	@classmethod
	def from_args(cls, args: Namespace) -> "CrossValidationConfig":
		return replace(
			cls.from_defaults(),
			**_overrides(cls, args, "kfold"),
		)

		# end

	def __post_init__(self) -> None:
		from math import isclose

		if not 0 < self.train_ratio < 1:
			raise ValueError(f"La percentuale di campioni per il Train set ({self.train_ratio}) deve essere compresa tra 0 e 1.")

		if not 0 < self.val_ratio < 1:
			raise ValueError(f"La percentuale di campioni per il Validation set ({self.val_ratio}) deve essere compresa tra 0 e 1.")

		if not isclose(
			self.train_ratio + self.val_ratio, 1.0,
			rel_tol=0.0, abs_tol=1e-9
		):
			raise ValueError(
				"Le percentuali di campioni per Train e Validation set devono sommare a 1.\n" \
				f"Train: {self.train_ratio:.4f}, " \
				f"Val: {self.val_ratio:.4f}, " \
				f"Total: {(self.train_ratio + self.val_ratio):.4f}"
			)

		check_positive_int('test_folds', self.test_folds)

		if self.test_folds == 1:
			if self.test_ratio is None or not 0 < self.test_ratio < 1:
				raise ValueError("Il valore di 'test_ratio' deve essere compreso tra 0 e 1, se specificato, quando la K-fold cross-validation è disabilitata.")

		if not 0 < self.sub_sample_ratio <= 1:
			raise ValueError("La percentuale di campioni per il sub-sampling deve essere compresa tra 0 e 1.")

		# end

	def __repr__(self) -> str:
		if self.test_folds == 1:
			test_str = f"Test ratio: {self.test_ratio}"
		else:
			test_str = f"Test folds: {self.test_folds}"

		return f"[CROSS-VALIDATION] Sub-sample ratio: {self.sub_sample_ratio} | Train ratio: {self.train_ratio} | Val ratio: {self.val_ratio} | {test_str}"

		# end

	# end class

@dataclass(frozen=True)
class TrainingConfig:
	epochs: int
	""" Number of training epochs. Default is 40. """
	
	batch_size: int
	""" Size of the mini-batches used during training. Default is 50. """
	
	early_stopping: bool
	""" Whether to use early stopping during training. """
	
	patience: int
	""" Number of epochs to wait for improvement before stopping training. If None, it will be automatically calculated based on the total number of epochs. """

	@classmethod
	def from_defaults(cls, epochs = 40, patience = None) -> "TrainingConfig":

		patience = \
			patience if patience is not None \
			else max(5, epochs // 10) if epochs <= 100 else 10

		return cls(
			epochs= epochs,
			batch_size= 50,
			early_stopping= True,
			patience= patience,
		)

		# end

	@classmethod
	def from_args(cls, args: Namespace) -> "TrainingConfig":
		changes = _overrides(cls, args, "training")

		derived_inputs = {
			name: changes[name]
			for name in ("epochs", "patience")
			if name in changes
		}

		return replace(
			cls.from_defaults(**derived_inputs),
			**changes,
		)

		# end

	def __post_init__(self) -> None:
		check_positive_int('epochs', self.epochs)
		check_positive_int('batch_size', self.batch_size)
		check_positive_int('patience', self.patience)

		# end
	
	def __repr__(self) -> str:
		patience_str = f" | Patience: {self.patience}" if self.early_stopping else ""

		return f"[TRAINING] " \
			f"Epochs: {self.epochs} | Batch size: {self.batch_size}\n" \
			f"           Early stopping: {self.early_stopping}{patience_str}"

	# end class

@dataclass(frozen=True)
class OptimizerConfig:
	name: str
	""" Selected optimizer for training. Default is "AdamW". """
	
	learning_rate: float
	""" Learning rate for the optimizer. """
	
	weight_decay: float
	""" Weight decay coefficient; AdamW applies decoupled weight decay. """

	epsilon: float
	""" Numerical stability term for Adam, AdamW and RMSprop. """

	momentum: float
	""" Momentum factor for SGD and RMSprop. """

	beta1: float
	""" Exponential decay rate for the first moment in Adam and AdamW. """

	beta2: float
	""" Exponential decay rate for the second moment in Adam and AdamW. """

	alpha: float
	""" Smoothing coefficient for the moving average of squared gradients in RMSprop. Default is 0.99. """

	@classmethod
	def from_defaults(cls) -> "OptimizerConfig":
		return cls(
			name= "AdamW",
			learning_rate= 1e-3,
			weight_decay= 1e-4,
			epsilon= 1e-8,
			momentum= 0.9,
			beta1= 0.9, beta2= 0.999,
			alpha= 0.99
		)

		# end

	@classmethod
	def from_args(cls, args: Namespace) -> "OptimizerConfig":
		return replace(
			cls.from_defaults(),
			**_overrides(cls, args, "optimizer"),
		)

	def __post_init__(self) -> None:
		if self.name not in ["Adam", "AdamW", "SGD", "RMSprop"]:
			raise NotImplementedError(
				f"Tipo di optimizer non supportato: {self.name}. "
				"Usare: 'Adam', 'AdamW', 'SGD' oppure 'RMSprop'."
			)

		if self.learning_rate <= 0:
			raise ValueError("Il learning rate deve essere maggiore di zero.")

		if self.weight_decay < 0:
			raise ValueError("Il weight decay non può essere negativo.")

		if self.epsilon <= 0:
			raise ValueError("Il valore di epsilon deve essere maggiore di zero.")

		if self.momentum < 0 or self.momentum > 1:
			raise ValueError("Il valore di momentum deve essere compreso tra 0 e 1 esclusi.")

		if (not 0.0 <= self.beta1 < 1.0) or (not 0.0 <= self.beta2 < 1.0):
			raise ValueError("Il valore dei parametri Beta devono essere compresi tra 0 e 1 escluso.")

		# end

	def __repr__(self) -> str:
		return f"[OPTIMIZER] {self.name} | Learning rate: {self.learning_rate}"

		# end

	# end class

@dataclass(frozen=True)
class SchedulerConfig:
	name: str
	""" Selected scheduler for the learning rate. Default is "OneCycleSched". """
	
	max_lr: float
	""" Maximum learning rate for the OneCycleLR scheduler. If None, it will be automatically calculated based on the selected learning rate. """
	
	start_factor: float
	""" Starting factor for the learning rate scheduler. """
	
	step_size: int
	""" Step size for the StepLR scheduler. """
	
	gamma: float
	""" Multiplicative factor of learning rate decay for the StepLR scheduler. """

	@classmethod
	def from_defaults(cls, learning_rate: float) -> "SchedulerConfig":
		return cls(
			name= "OneCycleSched",
			max_lr= (learning_rate * 3),
			start_factor= 0.1,
			step_size= 5,
			gamma= 0.1,
		)

		# end

	@classmethod
	def from_args(cls, args: Namespace, learning_rate: float) -> "SchedulerConfig":
		return replace(
			cls.from_defaults(learning_rate),
			**_overrides(cls, args, "scheduler"),
		)

		# end

	def __post_init__(self) -> None:
		if self.name not in ["OneCycleSched", "LinearSched", "StepSched", "NoSched"]:
			raise NotImplementedError(
				f"Tipo di scheduler non supportato: {self.name}. "
				"Usare: 'OneCycleSched', 'LinearSched', 'StepSched' oppure 'NoSched'."
			)

		if self.name == "OneCycleSched" and self.max_lr <= 0:
			raise ValueError("Il valore di 'max_lr' deve essere positivo per OneCycleSched.")
			
		if self.start_factor <= 0 or self.start_factor > 1:
			raise ValueError("Il valore di start_factor deve essere compreso tra 0 e 1.")

		check_positive_int('step_size', self.step_size)

		if self.gamma <= 0 or self.gamma > 1:
			raise ValueError("Il valore di gamma deve essere compreso tra 0 e 1.")

		# end

	def __repr__(self) -> str:
		return f"[SCHEDULER] {self.name}"

		# end

	# end class

@dataclass(frozen=True)
class LossConfig:
	name: str
	""" Selected loss function for training. Default is "WeightedCrossEntropy". """
	
	focal_alpha: str | None
	""" Alpha parameter for the Focal Loss. Default is "class_weights". """
	
	focal_gamma: float | None
	""" Gamma parameter for the Focal Loss, typically between 1 and 5. Default is 2.0. """

	@classmethod
	def from_defaults(cls) -> "LossConfig":
		return cls(
			name= "WeightedCrossEntropy",
			focal_alpha= "class_weights",
			focal_gamma= 2.0,
		)

		# end

	@classmethod
	def from_args(cls, args: Namespace) -> "LossConfig":
		return replace(
			cls.from_defaults(),
			**_overrides(cls, args, "loss"),
		)

	def __post_init__(self) -> None:
		if self.name not in ["CrossEntropy", "WeightedCrossEntropy", "Focal"]:
			raise NotImplementedError(f"Loss {self.name} non supportata. Loss disponibili: ['CrossEntropy', 'WeightedCrossEntropy', 'Focal']")

		if self.name == "Focal":
			if self.focal_alpha is None:
				raise ValueError("Il parametro 'focal_alpha' deve essere specificato per la Focal Loss.")
			
			if self.focal_alpha not in ["class_weights", "uniform"]:
				raise NotImplementedError(f"Parametro focal_alpha '{self.focal_alpha}' non supportato. Opzioni disponibili: ['class_weights', 'uniform']")
		
			if self.focal_gamma is None:
				raise ValueError("Il parametro 'focal_gamma' deve essere specificato per la Focal Loss.")
			
			if self.focal_gamma <= 0:
				raise ValueError("Il valore di focal_gamma deve essere maggiore di zero per la Focal Loss.")

		# end

	def __repr__(self) -> str:
		return f"[LOSS] {self.name}"

		# end

	# end class

@dataclass(frozen=True)
class ModelConfig:
	name: str
	""" Selected model for the network. Default is "hybrid". """
	
	simulator: str
	""" Name of the quantum simulator to use with PennyLane. """
	
	n_qubits: int
	""" Number of qubits in the quantum circuit. """
	
	n_shots: int | None
	""" Number of shots for the quantum circuit execution. Default is 20 if not specified. """

	q_layers: int
	""" Represents the depth of the StronglyEntanglingLayers ansatz. Default is 3. """

	diff_method: str
	""" Differentiation method for the quantum circuit. """

	@classmethod
	def from_defaults(cls) -> "ModelConfig":

		return cls(
			name= "baseline",
			simulator= "default.qubit",
			q_layers= 3,
			n_qubits= 5,
			n_shots= None,
			diff_method= "backprop",
		)

		# end

	@classmethod
	def from_args(cls, args: Namespace) -> "ModelConfig":
		changes = _overrides(cls, args, "model")

		return replace(
			cls.from_defaults(),
			**changes,
		)

		# end

	def __post_init__(self) -> None:
		if self.simulator not in ["default.qubit"]:
			raise NotImplementedError(
				f"Simulatore {self.simulator} non supportato. "
				f"Simulatori disponibili: ['default.qubit']"
			)

		if self.name not in MODEL_REGISTRY:
			raise NotImplementedError(
				f"Modello '{self.name}' non supportato. "
				f"Modelli disponibili: {str(MODEL_REGISTRY.keys())}"
			)
		
		check_positive_int('n_qubits', self.n_qubits)
		check_positive_int('n_shots', self.n_shots) if self.n_shots is not None else None
		check_positive_int('q_layers', self.q_layers)

		# end

	def __repr__(self) -> str:
		quantum_str = \
			f" | Layers: {self.q_layers}" \
			f" | Simulator: {self.simulator}" \
			f" | Qubits: {self.n_qubits}" \
			f" | Shots: {self.n_shots}"

		# TODO: aggiungere "classic" o "quantum" in base al tipo di modello.

		return f"[MODEL] {self.name}{quantum_str}"

		# end

	# end class

#	####################################################################	#
#	Configurazione dei parametri di runtime.

@dataclass(frozen=True)
class FlagsConfig:
	debug: bool
	""" Whether to enable debug mode. """
	
	debug_cycle: bool
	""" Whether to enable debug mode inside each cycle of the execution loop in mirage_pickle_converter(). """
	
	save_output: bool
	""" Whether to save the output of the notebook. """

	@classmethod
	def from_defaults(cls) -> "FlagsConfig":
		return cls(
			debug= False,
			debug_cycle= False,
			save_output= True,
		)

		# end

	@classmethod
	def from_args(cls, args: Namespace) -> "FlagsConfig":
		return replace(
			cls.from_defaults(),
			**_overrides(cls, args),
		)

	def __post_init__(self) -> None:
		pass

		# end

	def __repr__(self) -> str:
		return f"\nFlagsConfig(\n\tdebug={self.debug}, \n\tdebug_cycle={self.debug_cycle}, \n\tsave_output={self.save_output}\n)"

		# end

	# end class

@dataclass(frozen=True)
class LaunchConfig:
	source_session_id: str | None
	""" Identifier for the configuration JSON file to load. """

	exec_mode_train: bool
	""" Whether to execute the training loop (True) or only run validation on a pre-saved model (False). """

	requested_device: str
	""" Name of the requested device for computation. Can be 'auto', 'cpu', 'cuda', or 'mps'. Default is 'auto'. """

	random_seed: int
	""" Random seed for reproducibility. """
	
	flows_to_inspect: tuple[int, ...] | None
	""" List of flows to inspect during debugging in mirage_pickle_converter(). """

	@classmethod
	def from_defaults(cls) -> "LaunchConfig":
		import time

		return cls(
			source_session_id= None,
			exec_mode_train= True,
			requested_device= "auto",
			random_seed= int(time.time()),
			flows_to_inspect= None,
		)

		# end

	@classmethod
	def from_args(cls, args: Namespace) -> "LaunchConfig":
		changes = _overrides(cls, args)

		if changes.get("flows_to_inspect") is not None:
			changes["flows_to_inspect"] = tuple(changes["flows_to_inspect"])

		return replace(
			cls.from_defaults(),
			**changes,
		)

		# end

	def use_config_file(self) -> bool:
		return self.source_session_id is not None

		# end
	
	def __post_init__(self) -> None:
		if self.source_session_id is not None:
			if not isinstance(self.source_session_id, str) or not self.source_session_id.strip():
				raise ValueError("Il 'source_session_id' deve essere una stringa non vuota.")

		if not self.exec_mode_train and self.source_session_id is None:
			raise ValueError("La modalità di sola valutazione richiede una sessione sorgente specificata tramite 'source_session_id'.")

		if "cuda" not in self.requested_device:
			if self.requested_device not in ("auto", "cpu", "mps"):
				raise ValueError(
					f"Dispositivo {self.requested_device} non valido.\n"
					f"Dispositivi validi: ['auto', 'cpu', 'cuda', 'mps']."
				)

		if self.random_seed < 0:
			raise ValueError("Il 'random_seed' non può essere negativo.")

		# end

	def __repr__(self) -> str:
		return f"\nLaunchConfig(\n\tsource_session_id={self.source_session_id}, \n\texec_mode_train={self.exec_mode_train}, \n\trequested_device={self.requested_device}, \n\trandom_seed={self.random_seed}, \n\tflows_to_inspect={self.flows_to_inspect}\n)"

		# end

	# end class

@dataclass(frozen=True)
class EnvironmentConfig:
	python_version: str | None
	platform: str | None
	torch_version: str | None
	numpy_version: str | None
	pandas_version: str | None
	scipy_version: str | None
	sklearn_version: str | None
	matplotlib_version: str | None
	seaborn_version: str | None
	pennylane_version: str | None

	@classmethod
	def from_environment(cls) -> "EnvironmentConfig":
		import platform

		def _get_package_version(package_name: str) -> str | None:
			from importlib.metadata import PackageNotFoundError, version

			try:
				return version(package_name)
			except PackageNotFoundError:
				return None

			# end
		
		return cls(
			python_version=platform.python_version(),
			platform=platform.platform(),
			torch_version=_get_package_version('torch'),
			numpy_version=_get_package_version('numpy'),
			pandas_version=_get_package_version('pandas'),
			scipy_version=_get_package_version('scipy'),
			sklearn_version=_get_package_version('scikit-learn'),
			matplotlib_version=_get_package_version('matplotlib'),
			seaborn_version=_get_package_version('seaborn'),
			pennylane_version=_get_package_version('pennylane'),
		)

		# end
	
	@classmethod
	def from_args(cls) -> "EnvironmentConfig":
		raise NotImplementedError("La classe EnvironmentConfig non supporta la creazione da argomenti.")

		# end

	def __repr__(self) -> str:
		return f"\nEnvironmentConfig(\n\ttorch_version={self.torch_version}, \n\tnumpy_version={self.numpy_version}, \n\tpandas_version={self.pandas_version}, \n\tscipy_version={self.scipy_version}, \n\tsklearn_version={self.sklearn_version}, \n\tmatplotlib_version={self.matplotlib_version}, \n\tseaborn_version={self.seaborn_version}, \n\tpennylane_version={self.pennylane_version}\n)"

		# end

	# end class

#	####################################################################	#
#	Classi principali

@dataclass(frozen=True)
class SessionConfig:
	dataset: DatasetConfig
	preprocessing: PreprocessingConfig
	traffic: TrafficConfig
	kfold: CrossValidationConfig
	training: TrainingConfig
	optimizer: OptimizerConfig
	scheduler: SchedulerConfig
	loss: LossConfig
	model: ModelConfig

	@classmethod
	def from_defaults(cls) -> "SessionConfig":
		optimizer = OptimizerConfig.from_defaults()

		return cls(
			dataset=DatasetConfig.from_defaults(),
			preprocessing=PreprocessingConfig.from_defaults(),
			traffic=TrafficConfig.from_defaults(),
			kfold=CrossValidationConfig.from_defaults(),
			training=TrainingConfig.from_defaults(),
			optimizer= optimizer,

			scheduler = SchedulerConfig.from_defaults(
				learning_rate=optimizer.learning_rate,
			),

			loss=LossConfig.from_defaults(),
			model=ModelConfig.from_defaults(),
		)

		# end

	@classmethod
	def from_args(cls, args: Namespace) -> "SessionConfig":
		source_session_id = getattr(args, "source_session_id", None)

		#	-------------------------------------------------------------	#
		#	Scenario 1 (caricamento da default)
		#	Scenario 2 (caricamento da argomenti con override)

		if source_session_id is None:
			optimizer = OptimizerConfig.from_args(args)

			return cls(
				dataset=DatasetConfig.from_args(args),
				preprocessing=PreprocessingConfig.from_args(args),
				traffic=TrafficConfig.from_args(args),
				kfold=CrossValidationConfig.from_args(args),
				training=TrainingConfig.from_args(args),
				optimizer=optimizer,
				scheduler=SchedulerConfig.from_args(args, optimizer.learning_rate),
				loss=LossConfig.from_args(args),
				model=ModelConfig.from_args(args),
			)

		#	-------------------------------------------------------------	#
		#	Scenario 3 (caricamento da sessione esistente)

		elif source_session_id is not None:
			section_names = {field.name for field in fields(cls)}
			more_overrides = [name for name in vars(args) if name.split(".", 1)[0] in section_names]

			if more_overrides:
				raise ValueError(
					"Non è consentito combinare source_session_id con altri parametri di sessione.\n"
					f"Parametri trovati: {sorted(more_overrides)}"
				)

			return cls.from_json(find_session_config(source_session_id))
			
		# end

	@classmethod
	def from_json(cls, session_path) -> "SessionConfig":

		def _from_dict(session_dict) -> "SessionConfig":

			if not isinstance(session_dict, dict):
				raise ValueError("La configurazione deve essere rappresentata da un oggetto JSON.")

			required_sections = {
				"dataset", "preprocessing",
				"traffic", "kfold", "training",
				"optimizer", "scheduler", "loss", "model",
			}

			missing_sections = required_sections - session_dict.keys()
			if missing_sections:
				raise ValueError(
					"Sezioni mancanti nel file di configurazione: "
					f"{sorted(missing_sections)}"
				)

			try:

				traffic_dict = dict(session_dict["traffic"])
				traffic_filters = TrafficFiltersConfig(**traffic_dict["filters"])
				traffic = TrafficConfig(
					mtu=traffic_dict["mtu"],
					bin_size=traffic_dict["bin_size"],
					filters=traffic_filters,
				)

				dataset = DatasetConfig(**session_dict["dataset"])
				preprocessing = PreprocessingConfig(**session_dict["preprocessing"])
				kfold = CrossValidationConfig(**session_dict["kfold"])
				training = TrainingConfig(**session_dict["training"])
				optimizer = OptimizerConfig(**session_dict["optimizer"])
				scheduler = SchedulerConfig(**session_dict["scheduler"])
				loss = LossConfig(**session_dict["loss"])
				model = ModelConfig(**session_dict["model"])

			except KeyError as e:
				raise ValueError(f"Chiave mancante nel file di configurazione: {e}") from e
			except TypeError as e:
				raise ValueError(f"Struttura del file di configurazione non valida: {e}") from e

			return cls(
				dataset=dataset,
				preprocessing=preprocessing,
				traffic=traffic,
				kfold=kfold,
				training=training,
				optimizer=optimizer,
				scheduler=scheduler,
				loss=loss,
				model=model,
			)

			# end

		session_path = Path(session_path)

		if not session_path.is_file():
			raise FileNotFoundError(f"File di configurazione non trovato: {session_path}")

		try:
			with open(session_path, "r", encoding="utf-8") as f:
				session_dict = json.load(f)
		except json.JSONDecodeError as error:
			raise ValueError(
				f"JSON non valido in '{session_path}': "
				f"riga {error.lineno}, colonna {error.colno}."
			) from error

		return _from_dict(session_dict)

		# end

	def __post_init__(self) -> None:
		pass

		# end
	
	# def __repr__(self) -> str:
	# 	return f"\nSessionConfig(\n{repr(self.dataset)}, \n{repr(self.preprocessing)}, \n{repr(self.traffic)}, \n{repr(self.kfold)}, \n{repr(self.training)}, \n{repr(self.optimizer)}, \n{repr(self.scheduler)}, \n{repr(self.loss)}, \n{repr(self.model)}\n)"

	def __repr__(self) -> str:
		return \
			"[SESSION INFORMATION]\n" \
			f"{self.dataset}\n" \
			f"{self.kfold}\n" \
			f"{self.training}\n" \
			f"{self.optimizer} | {self.scheduler} | {self.loss}\n" \
			f"{self.model}\n"

		# end

	# end class

@dataclass(frozen=True)
class RuntimeConfig:
	flags: FlagsConfig
	""" Configuration of the flags for the current session. """

	launch: LaunchConfig
	""" Configuration of the launch parameters for the current session. """

	environment: EnvironmentConfig
	""" Configuration of the environment for the current session. """

	@classmethod
	def from_defaults(cls) -> "RuntimeConfig":
		return cls(
			flags=FlagsConfig.from_defaults(),
			launch=LaunchConfig.from_defaults(),
			environment=EnvironmentConfig.from_environment(),
		)

		# end

	@classmethod
	def from_args(cls, args: Namespace) -> "RuntimeConfig":

		runtime_types = get_type_hints(cls)
		source_session_id = getattr(args, "source_session_id", None)
		
		#	-------------------------------------------------------------	#
		#	Scenario 1 (caricamento da default)
		#	Scenario 3 (caricamento da sessione esistente)

		if source_session_id is None:
			base_config = cls.from_defaults()
		elif source_session_id is not None:
			base_config = cls.from_json(find_session_config(source_session_id))
		
		#	-------------------------------------------------------------	#
		#	Scenario 2 (caricamento da argomenti con override)
		
		changes = {}
		derived_inputs = {
			field.name: runtime_types[field.name]
			for field in fields(cls)
			if field.init
		}

		for section_name, section_class in derived_inputs.items():
			section_overrides = _overrides(section_class, args)
			if section_overrides:
				changes[section_name] = replace(
					getattr(base_config, section_name),
					**section_overrides,
				)

		return replace(base_config, **changes)

		# end

	@classmethod
	def from_json(cls, runtime_path) -> "RuntimeConfig":

		def _from_dict(runtime_dict) -> "RuntimeConfig":

			if not isinstance(runtime_dict, dict):
				raise ValueError("La configurazione deve essere rappresentata da un oggetto JSON.")

			required_sections = {"flags", "launch", "environment"}

			missing_sections = required_sections - runtime_dict.keys()
			if missing_sections:
				raise ValueError(
					"Chiavi mancanti nel file di configurazione: "
					f"{sorted(missing_sections)}"
				)

			launch_dict = dict(runtime_dict["launch"])
			flows_to_inspect = launch_dict.get("flows_to_inspect")
			if flows_to_inspect is not None:
				launch_dict["flows_to_inspect"] = tuple(flows_to_inspect)

			try:
				flags = FlagsConfig(**runtime_dict["flags"])
				launch = LaunchConfig(**launch_dict)
				environment = EnvironmentConfig(**runtime_dict["environment"])

			except KeyError as e:
				raise ValueError(f"Chiave mancante nel file di configurazione: {e}") from e
			except TypeError as e:
				raise ValueError(f"Struttura del file di configurazione non valida: {e}") from e

			return cls(
				flags=flags,
				launch=launch,
				environment=environment,
			)

			# end

		runtime_path = Path(runtime_path)

		if not runtime_path.is_file():
			raise FileNotFoundError(f"File di configurazione non trovato: {runtime_path}")

		try:
			with open(runtime_path, "r", encoding="utf-8") as f:
				runtime_dict = json.load(f)
		except json.JSONDecodeError as error:
			raise ValueError(
				f"JSON non valido in '{runtime_path}': "
				f"riga {error.lineno}, colonna {error.colno}."
			) from error

		return _from_dict(runtime_dict)

		# end

	def __post_init__(self) -> None:
		pass

		# end

	def print_exec_mode(self) -> None:
		print(f"\nModalità di {'addestramento' if self.launch.exec_mode_train else 'valutazione'}.")

	def __repr__(self) -> str:
		return f"\nRuntimeConfig(\n{repr(self.flags)}, \n{repr(self.launch)}, \n{repr(self.environment)}\n)"

		# end

	# end class

@dataclass(frozen=True)
class ExecutionConfig:
	session_id: str
	""" Unique identifier for the session. """

	current_device: str
	""" Name of the device used for computation. """

	dtype: str
	""" Data type used for computation. """
	
	@classmethod
	def from_args(cls, session_id, device, dtype) -> "ExecutionConfig":
		return cls(
			session_id=session_id,
			current_device=device,
			dtype=dtype,
		)

		# end

	@classmethod
	def from_json(cls, execution_path) -> "ExecutionConfig":

		def _from_dict(execution_dict) -> "ExecutionConfig":

			if not isinstance(execution_dict, dict):
				raise ValueError("La configurazione deve essere rappresentata da un oggetto JSON.")

			required_keys = {"session_id", "current_device", "dtype"}

			missing_keys = required_keys - execution_dict.keys()
			if missing_keys:
				raise ValueError(
					"Chiavi mancanti nel file di configurazione: "
					f"{sorted(missing_keys)}"
				)

			try:
				session_id = execution_dict["session_id"]
				device = execution_dict["current_device"]
				dtype = execution_dict["dtype"]

			except KeyError as e:
				raise ValueError(f"Chiave mancante nel file di configurazione: {e}") from e
			except TypeError as e:
				raise ValueError(f"Struttura del file di configurazione non valida: {e}") from e

			return cls(
				session_id=session_id,
				current_device=device,
				dtype=dtype,
			)

			# end

		execution_path = Path(execution_path)

		if not execution_path.is_file():
			raise FileNotFoundError(f"File di configurazione non trovato: {execution_path}")

		try:
			with open(execution_path, "r", encoding="utf-8") as f:
				execution_dict = json.load(f)["execution"]
		except json.JSONDecodeError as error:
			raise ValueError(
				f"JSON non valido in '{execution_path}': "
				f"riga {error.lineno}, colonna {error.colno}."
			) from error
		except KeyError as e:
			raise ValueError(f"Chiave mancante nel file di configurazione: {e}") from e
		except TypeError as e:
			raise ValueError(f"Struttura del file di configurazione non valida: {e}") from e

		return _from_dict(execution_dict)

		# end

	def __post_init__(self) -> None:
		if self.session_id is None or (self.session_id.strip() == ""):
			raise ValueError("Identificatore della sessione mancante.")
		if self.current_device is None or (self.current_device.strip() == ""):
			raise ValueError("Dispositivo mancante.")
		if self.dtype is None or (self.dtype.strip() == ""):
			raise ValueError("Tipo di dato mancante.")

		# end

	def __repr__(self) -> str:
		return f"\nExecutionConfig(\n\tsession_id={self.session_id}, \n\tcurrent_device={self.current_device}, \n\tdtype={self.dtype}\n)"

		# end

	# end class
