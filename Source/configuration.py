"""
	configuration.py
	by Mario Gabriele Carofano

	Questo modulo definisce le classi di configurazione dove vengono centralizzati tutti i parametri di sessione. Tali parametri contengono le impostazioni specifiche per una sessione di esecuzione del progetto. Sono presenti funzioni di salvataggio e caricamento in file JSON, in modo da poter riprodurre gli esperimenti e confrontare i risultati.

	Inoltre contiene altre costanti, dette strutturali, che non vengono caricate / salvate tramite le classi di configurazione nel file JSON.
"""

from dataclasses import dataclass

#	####################################################################	#
#	Costanti strutturali

EXEC_MODE_TRAIN = True
""" Set this flag to choose between training a new model
or running validation on an already saved model. \n
- True  : Execute the training loop.
- False : Skip training and run validation only on a pre-saved model. """

USE_CONFIG_FILE = False
""" Whether to use a configuration file to load the constants. """

CONFIG_ID = "2026-10-18_17-21"
""" Identifier for the configuration JSON file to load. """

# TODO: rimettere SAVE_OUTPUT, DEBUG, DEBUG_CYCLE

#	####################################################################	#
#	Configurazione dei parametri di sessione.

@dataclass(frozen=True)
class ExecutionConfig:
	random_seed: int
	""" Random seed for reproducibility. """
	
	debug: bool
	""" Whether to enable debug mode. """
	
	debug_cycle: bool
	""" Whether to enable debug mode inside each cycle of the execution loop in mirage_pickle_converter(). """
	
	flows_to_inspect: tuple[int, ...] | None
	""" List of flows to inspect during debugging in mirage_pickle_converter(). """
	
	use_precomputed_dataset: bool
	""" Whether to load the FlowPic dataset from a precomputed .npz file. """
	
	save_output: bool
	""" Whether to save the output of the notebook. """

	@classmethod
	def from_constants(cls) -> "ExecutionConfig":
		return cls(
			random_seed= 2025,
			debug= False,
			debug_cycle= False,
			flows_to_inspect= None,
			use_precomputed_dataset= True,
			save_output= True,
		)

		# end

	def __repr__(self) -> str:
		return f"\nExecutionConfig(\n\trandom_seed={self.random_seed}, \n\tdebug={self.debug}, \n\tdebug_cycle={self.debug_cycle}, \n\tflows_to_inspect={self.flows_to_inspect}, \n\tuse_precomputed_dataset={self.use_precomputed_dataset}, \n\tsave_output={self.save_output}\n)"

		# end

	# end class

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
	def from_constants(cls) -> "DatasetConfig":
		return cls(
			path= "Dataset/mirage/2019",
			name= "mirage2019_LOPEZ_lopez_lopez_100P_4F_APP_xST_PAD_metadata.pickle",
			is_flowpic= False,
			n_packets= 100,
			n_features= 4,
		)

		# end

	def __repr__(self) -> str:
		return f"\nDatasetConfig(\n\tpath={self.path}, \n\tname={self.name}, \n\tis_flowpic={self.is_flowpic}, \n\tn_packets={self.n_packets}, \n\tn_features={self.n_features}\n)"

		# end

	# end class

@dataclass(frozen=True)
class PreprocessingConfig:
	strategy: str
	""" Strategy for preprocessing the dataset. Default is 'MinMax'. """

	@classmethod
	def from_constants(cls) -> "PreprocessingConfig":
		return cls(
			strategy= "MinMax",
		)

		# end

	def __repr__(self) -> str:
		return f"\nPreprocessingConfig(\n\tstrategy={self.strategy}\n)"

		# end

	# end class

@dataclass(frozen=True)
class TrafficFiltersConfig:
	min_tps: float | None = None
	""" Minimum duration of a session in seconds. Sessions shorter than this will be discarded. Default is 50. """
	
	min_packets: int | None = None
	""" Minimum number of packets in a session. Sessions with fewer packets will be discarded. Default is 3 if not specified. """
	
	min_dim: int | None = None
	""" Minimum flow dimension (payload) in bytes. Flows smaller than this will be discarded. Default is 10000 if not specified. """

	@classmethod
	def from_constants(cls) -> "TrafficFiltersConfig":

		return cls(
			min_tps= 50,
			min_packets= None,
			min_dim= None,
		)

		# end

	def __repr__(self) -> str:
		return f"TrafficFiltersConfig(\n\tmin_tps={self.min_tps}, \n\tmin_packets={self.min_packets}, \n\tmin_dim={self.min_dim}\n\t)"

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
	def from_constants(cls) -> "TrafficConfig":
		return cls(
			mtu= 1500,
			bin_size= 10,
			filters=TrafficFiltersConfig.from_constants(),
		)

		# end

	def __repr__(self) -> str:
		return f"\nTrafficConfig(\n\tmtu={self.mtu}, \n\tbin_size={self.bin_size}, \n\tfilters={self.filters}\n)"

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
	""" Proportion of the dataset to be used for testing. Default is 0.2. """
	
	sub_sample_ratio: float
	""" Ratio of the dataset to be used for sub-sampling. For example, 0.5 means using 50% of the dataset. Default should be 1.0, meaning the entire dataset is used. """

	@classmethod
	def from_constants(cls) -> "CrossValidationConfig":
		test_folds = 1

		return cls(
			train_ratio= 0.8,
			val_ratio= 0.2,
			test_folds= test_folds,
			test_ratio= None if test_folds > 1 else 0.2,
			sub_sample_ratio= 0.01,
		)

		# end

	def __repr__(self) -> str:
		return f"\nCrossValidationConfig(\n\ttrain_ratio={self.train_ratio}, \n\tval_ratio={self.val_ratio}, \n\ttest_folds={self.test_folds}, \n\ttest_ratio={self.test_ratio}, \n\tsub_sample_ratio={self.sub_sample_ratio}\n)"

		# end

	# end class

@dataclass(frozen=True)
class TrainingConfig:
	epochs: int
	""" Number of training epochs. Default is 20. """
	
	batch_size: int
	""" Size of the mini-batches used during training. """
	
	early_stopping: bool
	""" Whether to use early stopping during training. """
	
	patience: int
	""" Number of epochs to wait for improvement before stopping training. If None, it will be automatically calculated based on the total number of epochs. """

	@classmethod
	def from_constants(cls) -> "TrainingConfig":
		epochs = 20
		patience = None

		def _get_patience() -> int:
			return \
				max(2, epochs // 10) \
				if epochs <= 100 \
				else 10

			# end

		patience = (
			patience
			if patience is not None
			else _get_patience()
		)

		return cls(
			epochs= epochs,
			batch_size= 50,
			early_stopping= True,
			patience= patience,
		)

		# end
	
	def __repr__(self) -> str:
		return f"\nTrainingConfig(\n\tepochs={self.epochs}, \n\tbatch_size={self.batch_size}, \n\tearly_stopping={self.early_stopping}, \n\tpatience={self.patience}\n)"

	# end class

@dataclass(frozen=True)
class OptimizerConfig:
	name: str
	""" Selected optimizer for training. Default is "AdamW". """
	
	learning_rate: float
	""" Learning rate for the optimizer. """
	
	weight_decay: float
	""" Weight decay (L2 penalty) for the optimizer. """
	
	epsilon: float
	""" Epsilon value for the optimizer AdamW. """
	
	momentum: float
	""" Momentum for the optimizer SGD. """

	@classmethod
	def from_constants(cls) -> "OptimizerConfig":
		return cls(
			name= "AdamW",
			learning_rate= 1e-3,
			weight_decay= 1e-4,
			epsilon= 1e-2,
			momentum= 0.9,
		)

		# end

	def __repr__(self) -> str:
		return f"\nOptimizerConfig(\n\tname={self.name}, \n\tlearning_rate={self.learning_rate}, \n\tweight_decay={self.weight_decay}, \n\tepsilon={self.epsilon}, \n\tmomentum={self.momentum}\n)"

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
	def from_constants(cls, learning_rate: float) -> "SchedulerConfig":
		max_lr = None

		def _get_max_lr(learning_rate: float) -> float:
			return learning_rate * 3

		# end

		max_lr = \
			max_lr \
			if max_lr is not None \
			else _get_max_lr(learning_rate)

		return cls(
			name= "OneCycleSched",
			max_lr= max_lr,
			start_factor= 0.1,
			step_size= 5,
			gamma= 0.1,
		)

	def __repr__(self) -> str:
		return f"\nSchedulerConfig(\n\tname={self.name}, \n\tmax_lr={self.max_lr}, \n\tstart_factor={self.start_factor}, \n\tstep_size={self.step_size}, \n\tgamma={self.gamma}\n)"

		# end

	# end class

@dataclass(frozen=True)
class LossConfig:
	name: str
	""" Selected loss function for training. Default is "WeightedCrossEntropy". """
	
	alpha: str
	""" Alpha parameter for the Focal Loss. Can be "class_weights", "uniform" or "custom". """
	
	focal_gamma: float
	""" Gamma parameter for the Focal Loss, typically between 1 and 5. """

	@classmethod
	def from_constants(cls) -> "LossConfig":
		return cls(
			name= "WeightedCrossEntropy",
			alpha= "class_weights",
			focal_gamma= 2.0,
		)

	def __repr__(self) -> str:
		return f"\nLossConfig(\n\tname={self.name}, \n\talpha={self.alpha}, \n\tfocal_gamma={self.focal_gamma}\n)"

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
	
	n_shots: int
	""" Number of shots for the quantum circuit execution. """
	
	n_layers: int
	""" Number of layers for the selected model. If None, it will be automatically calculated based on the selected model.
	- For ResNet, possible values are 16, 18, 34. Default is 18.
	- For Quantum models, this represents the depth of the StronglyEntanglingLayers ansatz. Default is 3. """

	@classmethod
	def from_constants(cls) -> "ModelConfig":
		selected_model = "hybrid"
		n_layers = None

		def _get_num_layers() -> int:

			available_models = {
				"resnet" : 18,
				"hybrid" : 3,
			}

			if selected_model not in available_models:
				raise ValueError(f"Modello {selected_model} non supportato. Modelli disponibili: {available_models}")

			return available_models[selected_model]

			# end

		n_layers = \
			n_layers \
			if n_layers is not None \
			else _get_num_layers()

		return cls(
			name= selected_model,
			simulator= "default.qubit",
			n_qubits= 5,
			n_shots= 20,
			n_layers= n_layers,
		)

	def __repr__(self) -> str:
		return f"\nModelConfig(\n\tname={self.name}, \n\tsimulator={self.simulator}, \n\tn_qubits={self.n_qubits}, \n\tn_shots={self.n_shots}, \n\tn_layers={self.n_layers}\n)"

		# end

	# end class

@dataclass(frozen=True)
class LibrariesConfig:
	torch_version: str | None
	numpy_version: str | None
	pandas_version: str | None
	scipy_version: str | None
	sklearn_version: str | None
	matplotlib_version: str | None
	seaborn_version: str | None
	pennylane_version: str | None

	@classmethod
	def from_environment(cls) -> "LibrariesConfig":

		def _get_package_version(package_name: str) -> str | None:
			from importlib.metadata import PackageNotFoundError, version

			try:
				return version(package_name)
			except PackageNotFoundError:
				return None

			# end
		
		return cls(
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

	def __repr__(self) -> str:
		return f"LibrariesConfig(\n\ttorch_version={self.torch_version}, \n\tnumpy_version={self.numpy_version}, \n\tpandas_version={self.pandas_version}, \n\tscipy_version={self.scipy_version}, \n\tsklearn_version={self.sklearn_version}, \n\tmatplotlib_version={self.matplotlib_version}, \n\tseaborn_version={self.seaborn_version}, \n\tpennylane_version={self.pennylane_version}\n\t)"

		# end

	# end class

#	####################################################################	#
#	Classi principali

@dataclass(frozen=True)
class SessionConfig:
	execution: ExecutionConfig
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
	def from_constants(cls) -> "SessionConfig":
		optimizer = OptimizerConfig.from_constants()

		return cls(
			execution=ExecutionConfig.from_constants(),
			dataset=DatasetConfig.from_constants(),
			preprocessing=PreprocessingConfig.from_constants(),
			traffic=TrafficConfig.from_constants(),
			kfold=CrossValidationConfig.from_constants(),
			training=TrainingConfig.from_constants(),
			optimizer= optimizer,

			scheduler = SchedulerConfig.from_constants(
				learning_rate=optimizer.learning_rate,
			),

			loss=LossConfig.from_constants(),
			model=ModelConfig.from_constants(),
		)

		# end

	@classmethod
	def from_json(cls, session_path) -> "SessionConfig":

		def _from_dict(session_dict) -> "SessionConfig":

			assert isinstance(session_dict, dict), "La configurazione deve essere rappresentata da un oggetto JSON."

			required_sections = {
				"execution", "dataset", "preprocessing",
				"traffic", "kfold", "training",
				"optimizer", "scheduler", "loss", "model",
			}

			missing_sections = required_sections - session_dict.keys()
			assert not missing_sections, f"Sezioni mancanti nel file di configurazione: {sorted(missing_sections)}"

			try:

				execution_dict = dict(session_dict["execution"])
				flows_to_inspect = execution_dict.get("flows_to_inspect")
				if flows_to_inspect is not None:
					execution_dict["flows_to_inspect"] = tuple(flows_to_inspect)

				traffic_dict = dict(session_dict["traffic"])
				traffic_filters = TrafficFiltersConfig(**traffic_dict["filters"])
				traffic = TrafficConfig(
					mtu=traffic_dict["mtu"],
					bin_size=traffic_dict["bin_size"],
					filters=traffic_filters,
				)

				execution = ExecutionConfig(**execution_dict)
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
				execution=execution,
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

		from pathlib import Path
		import json

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
	
	def __repr__(self) -> str:
		return f"\nSessionConfig(\n\t{repr(self.execution)}, \n\t{repr(self.dataset)}, \n\t{repr(self.preprocessing)}, \n\t{repr(self.traffic)}, \n\t{repr(self.kfold)}, \n\t{repr(self.training)}, \n\t{repr(self.optimizer)}, \n\t{repr(self.scheduler)}, \n\t{repr(self.loss)}, \n\t{repr(self.model)}\n)"

		# end

	# end class

@dataclass(frozen=True)
class RuntimeConfig:
	session_id: str
	device: str
	dtype: str
	python_version: str
	platform: str
	libraries: LibrariesConfig

	@classmethod
	def from_environment(cls,
		session_id, device, dtype
	) -> "RuntimeConfig":
		import platform

		return cls(
			session_id, device, dtype,
			python_version=platform.python_version(),
			platform=platform.platform(),
			libraries=LibrariesConfig.from_environment(),
		)

		# end

	@classmethod
	def from_json(cls, runtime_path) -> "RuntimeConfig":

		def _from_dict(runtime_dict) -> "RuntimeConfig":

			assert isinstance(runtime_dict, dict), "La configurazione deve essere rappresentata da un oggetto JSON."

			required_keys = {
				"session_id", "device", "dtype",
				"python_version", "platform", "libraries"
			}

			missing_keys = required_keys - runtime_dict.keys()
			assert not missing_keys, f"Chiavi mancanti nel file di configurazione: {sorted(missing_keys)}"

			try:

				session_id = runtime_dict["session_id"]
				device = runtime_dict["device"]
				dtype = runtime_dict["dtype"]
				python_version = runtime_dict["python_version"]
				platform = runtime_dict["platform"]
				libraries = LibrariesConfig(**runtime_dict["libraries"])

			except KeyError as e:
				raise ValueError(f"Chiave mancante nel file di configurazione: {e}") from e
			except TypeError as e:
				raise ValueError(f"Struttura del file di configurazione non valida: {e}") from e

			return cls(
				session_id=session_id,
				device=device,
				dtype=dtype,
				python_version=python_version,
				platform=platform,
				libraries=libraries,
			)

			# end

		from pathlib import Path
		import json

		runtime_path = Path(runtime_path)

		if not runtime_path.is_file():
			raise FileNotFoundError(f"File di configurazione non trovato: {runtime_path}")

		try:
			with open(runtime_path, "r", encoding="utf-8") as f:
				runtime_dict = json.load(f)
				runtime_dict = runtime_dict["runtime"]
		except json.JSONDecodeError as error:
			raise ValueError(
				f"JSON non valido in '{runtime_path}': "
				f"riga {error.lineno}, colonna {error.colno}."
			) from error
		except KeyError as e:
			raise ValueError(f"Chiave mancante nel file di configurazione: {e}") from e
		except TypeError as e:
			raise ValueError(f"Struttura del file di configurazione non valida: {e}") from e

		return _from_dict(runtime_dict)

		# end

	def __repr__(self) -> str:
		return f"\nRuntimeConfig(\n\tsession_id={self.session_id}, \n\tdevice={self.device}, \n\tdtype={self.dtype}, \n\tpython_version={self.python_version}, \n\tplatform={self.platform}, \n\tlibraries={repr(self.libraries)}\n)"

		# end

	# end class
