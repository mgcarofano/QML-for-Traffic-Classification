"""
	models.py
	di Mario Gabriele Carofano

	Questo modulo centralizza la definizione, la costruzione e la selezione dei modelli utilizzati per l'addestramento e la valutazione dei dataset di traffico di rete. Riunisce le architetture classiche e le reti quantistiche (e ibride), insieme alle funzioni di utilità per la gestione e selezione dei modelli.

	I componenti comuni sono organizzati in blocchi riutilizzabili per ridurre la duplicazione del codice e separare l'estrazione delle feature, l'elaborazione quantistica e la classificazione.

	La preparazione dei dati, la configurazione degli esperimenti e i cicli di addestramento e valutazione restano esterni al modulo. La scelta del dispositivo di calcolo, della precisione numerica e dei seed globali è demandata al codice chiamante.
"""

#	####################################################################	#
#	Librerie

import torch
import torch.nn as nn
import pennylane as qml

#	####################################################################	#
#	Funzioni di utilità

def compute_model_name(
		model_class : str,
		n_qubits : int = None,
		lstm_layers : int = None,
		q_layers : int = None,
		n_packets : int = None,
		n_features : int = None,
		num_classes : int = None
	) -> str:
	"""Restituisce una stringa con il nome del modello che riassume i parametri principali,
	se disponibili, nel formato 'class_Qn_LSTMn_QLn_Pn_Fn_Cn'.

	Args:
		model_class (str): Nome della classe del modello.
		n_qubits (int, optional): Numero di qubit nel circuito quantistico. Default è None.
		lstm_layers (int, optional): Numero di layer LSTM. Default è None.
		q_layers (int, optional): Numero di layer nel circuito quantistico. Default è None.
		n_packets (int, optional): Numero di pacchetti nell'input. Default è None.
		n_features (int, optional): Numero di feature per pacchetto. Default è None.
		num_classes (int, optional): Numero di classi per la classificazione. Default è None.

	Returns:
		str: Nome del modello.
	"""

	fields = (
		('Q', n_qubits),
		('LSTM', lstm_layers),
		('QL', q_layers),
		('P', n_packets),
		('F', n_features),
		('C', num_classes)
	)

	return model_class + ''.join(
		f'_{key}{value}'
		for key, value in fields
		if value is not None
	)

	# end

def weight_init(m: nn.Module) -> None:
	"""Fills-in weights and biases for convolutional and linear layers. Uses Kaiming uniform initialization for weights (suitable for ReLU/non-linearities) and sets biases to zero. This function is meant to be used with model.apply(), which will call it on every sub-module.

	Args:
		m (nn.Module): a module of the model.
	"""	

	if isinstance(m, (nn.Conv2d, nn.Linear)):

		# Good idea would be to use kaiming initialization of scaled orthonormal initialization.
		nn.init.kaiming_uniform_(
			m.weight,
			a=0,
			mode='fan_in',
			nonlinearity='relu'
		)

		# Check if m.bias is enabled.
		if m.bias is not None:
			nn.init.zeros_(m.bias)

	return

	# end

def check_model_compatibility(model, weights):
	"""Controlla la compatibilità tra il modello corrente e i pesi caricati da un checkpoint.

	Args:
		model (torch.nn.Module): Modello corrente.
		weights (dict): Pesi salvati del modello.

	Raises:
		RuntimeError: Se le shape dei pesi salvati non corrispondono a quelle del modello corrente.
	"""

	#	################################################################	#
	#	Inizializzazione

	from torch import Tensor

	current = model.state_dict()

	missing = sorted(current.keys() - weights.keys())
	unexpected = sorted(weights.keys() - current.keys())
	invalid = sorted(key for key in weights if not isinstance(weights[key], Tensor))

	mismatches = [
		f'{key}: atteso {tuple(current[key].shape)}, ricevuto {tuple(weights[key].shape)}'
		for key in sorted(current.keys() & weights.keys())
		if isinstance(weights[key], Tensor) and current[key].shape != weights[key].shape
	]

	#	################################################################	#
	#	Costruzione dell'elenco dei problemi di compatibilità

	problems = []

	for label, values in (
		('Chiavi mancanti', missing),
		('Chiavi inattese', unexpected),
		('Valori non tensoriali', invalid),
		('Shape diverse', mismatches)
	):
		if values:
			problems.append(f'{label}:\n  ' + '\n  '.join(values))

	if problems:
		raise RuntimeError(
			f"Errore durante il caricamento del modello: {type(model).__name__}\n"
			+ '\n'.join(problems) + "\n"
			"Verifica che il dataset e il numero di classi usati per il training coincidano con quelli attuali."
		)

	# Si può ritornare senza errori se non ci sono problemi di compatibilità.
	return

	# end

def select_quantum_device(dev_name, n_qubits, random_seed):

	if dev_name == "default.qubit":
		return qml.device(dev_name, wires=n_qubits, seed=random_seed)

	elif dev_name.startswith("fake"):
		raise NotImplementedError(
			"Fake quantum devices are not yet supported."
			f"Available devices are: ['default.qubit']"
		)

	elif dev_name.startswith("ibm"):
		raise NotImplementedError(
			"IBM quantum devices are not yet supported."
			f"Available devices are: ['default.qubit']"
		)

	else:
		raise NotImplementedError(
			f"Quantum device '{dev_name}' is not supported."
			f"Available devices are: ['default.qubit']"
		)

	# end

#	####################################################################	#
#	Definizione delle classi base per i modelli

class _BaseModel(nn.Module):
	"""Base PyTorch con validazione dei metadati comuni."""

	is_quantum : bool = False

	def __init__(self, n_features: int, n_packets: int, num_classes: int):

		super().__init__()

		self.n_features = n_features
		self.n_packets = n_packets
		self.num_classes = num_classes

		# end

	def get_model_name(self) -> str:
		return compute_model_name(
			self.model_label,
			**{field: getattr(self, field) for field in self.name_fields}
		)

	# end class

class _FeatureClassifier(_BaseModel):
	"""Un solo forward per feature -> adapter -> processing -> logits. I componenti vengono assegnati una sola volta, senza alias registrati.
	"""

	def forward(self, x):
		x = self.feature_extractor(x)
		x = self.input_adapter(x)
		x = self.processing_block(x)

		return self.classifier(x)

		# end

	# end class

#	####################################################################	#
#	Definizione dell'estrattore di feature CNN + LSTM

class _CNNLSTMFeatureExtractor(nn.Module):
	"""Due Conv2d + LSTM + Linear, uscita (B, 100)."""

	output_dim = 100

	def __init__(self, n_features: int, n_packets: int, lstm_layers: int):

		super().__init__()

		self.n_features = n_features
		self.n_packets = n_packets
		self.lstm_layers = lstm_layers

		if self.n_packets < 7 or self.n_features < 3:
			raise ValueError(
				"CNNLSTM richiede n_packets >= 7 e n_features >= 3 "
				"per due Conv2d valid con kernel (4, 2)."
			)

		self.conv1 = nn.Conv2d(1, 32, (4, 2), stride=(1, 1), padding=0)
		self.bn1 = nn.BatchNorm2d(32)
		self.conv2 = nn.Conv2d(32, 64, (4, 2), stride=(1, 1), padding=0)
		self.bn2 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU()

		with torch.no_grad():
			dummy = self.conv1.weight.new_zeros(1, 1, self.n_packets, self.n_features)
			shape = self.conv2(self.conv1(dummy)).shape

		_, channels, packets, features = shape
		lstm_input_size = channels * features

		# self.lstm = nn.LSTM(lstm_input_size, 100, batch_first=True)
		# self.dropout1 = nn.Dropout(0.2)
		# self.dropout2 = nn.Dropout(0.4)

		self.lstm = nn.LSTM(
			lstm_input_size, 100,
			num_layers= lstm_layers,
			batch_first=True,
			dropout=0.2 if lstm_layers > 1 else 0.0,
		)

		self.fc1 = nn.Linear(100, 100)

		# end

	def forward(self, x):
		if x.ndim != 3 or tuple(x.shape[1:]) != (self.n_packets, self.n_features):
			raise ValueError(
				"Errore in _CNNLSTMFeatureExtractor:\n"
				f"Atteso: (B,{self.n_packets},{self.n_features})\n"
				f"Ricevuto: {tuple(x.shape)}."
			)
		
		out = self.bn1(self.relu(self.conv1(x.unsqueeze(1))))
		out = self.bn2(self.relu(self.conv2(out)))
		out = out.permute(0, 2, 1, 3).flatten(start_dim=2)

		# out, _ = self.lstm(out)
		# out = self.dropout1(out[:, -1, :])

		# return self.dropout2(self.relu(self.fc1(out)))

		_, (h_n, _) = self.lstm(out)
		return self.fc1(h_n[-1])
	
	# end class

#	####################################################################	#
#	Definizione del blocco di elaborazione quantistico

class QuantumBlock(nn.Module):
	"""Blocco di elaborazione Embedding + StronglyEntanglingLayers."""

	def __init__(
			self, embedding: str, diff_method: str,
			q_layers: int, n_qubits: int, n_shots: int,
			random_seed: int, dev_name: str
		):

		#	############################################################	#
		#	Inizializzazione degli attributi di classe.

		super().__init__()

		if embedding not in ['amplitude', 'angle']:
			raise ValueError("Il tipo di embedding deve essere 'amplitude' o 'angle'.")
		self.embedding = embedding

		if diff_method not in ['backprop', 'parameter-shift']:
			raise ValueError("Il metodo di differenziazione deve essere 'backprop' o 'parameter-shift'.")
		if diff_method == 'backprop' and n_shots is not None:
			raise ValueError("Per il metodo di differenziazione 'backprop', il valore di 'n_shots' deve essere None. Utilizzare 'parameter-shift'.")
		self.diff_method = diff_method

		self.q_layers = q_layers
		self.n_qubits = n_qubits
		self.n_shots = n_shots

		self.input_dim = 2 ** self.n_qubits if embedding == 'amplitude' else self.n_qubits
		self.output_dim = 2 ** self.n_qubits
		self.dev = select_quantum_device(dev_name, self.n_qubits, random_seed)

		#	############################################################	#
		#	Definizione del QNode.

		wires = tuple(range(self.n_qubits))

		@qml.qnode(self.dev, interface='torch', diff_method=self.diff_method)
		def qnode(inputs, weights):

			# Feature map
			if embedding == 'amplitude':
				qml.AmplitudeEmbedding(inputs, wires=wires, normalize=True, pad_with=0.0)
			else:
				qml.AngleEmbedding(inputs, wires=wires, rotation='X')

			# Ansatz (circuito parametrico)
			qml.StronglyEntanglingLayers(weights, wires=wires)

			# Measurement process
			return qml.probs(wires=wires)

			# end

		#	############################################################	#
		#	Definizione del circuito quantistico.

		qnode = qml.set_shots(qnode, shots=self.n_shots)
		if self.diff_method == 'parameter-shift':
			qnode = qml.transforms.broadcast_expand(qnode)

		weight_shapes = {'weights': qml.StronglyEntanglingLayers.shape(
			n_layers=self.q_layers, n_wires=self.n_qubits
		)}

		self.torch_layer = qml.qnn.TorchLayer(qnode, weight_shapes)

		# end

	@property
	def qnode(self):
		return self.torch_layer.qnode

		# end

	def forward(self, x):
		if x.ndim != 2 or x.shape[-1] != self.input_dim:
			raise ValueError(
				"Errore in _CNNLSTMFeatureExtractor:\n"
				f"Atteso: (B,{self.input_dim})\n"
				f"Ricevuto: {tuple(x.shape)}."
			)

		return self.torch_layer(x)

	# end class

class _QuantumModel(_FeatureClassifier):
	"""Modello quantistico di base."""

	name_fields = [
		"n_qubits", "lstm_layers", "q_layers",
		"n_packets", "n_features", "num_classes"
	]
	embedding: str
	encoder_type: str
	is_quantum : bool = True

	def __init__(
			self,
			n_features: int, n_packets: int, num_classes: int,
			lstm_layers: int, q_layers: int, n_qubits: int, n_shots: int,
			random_seed: int, dev_name: str,
			diff_method: str
		):

		super().__init__(n_features, n_packets, num_classes)

		self.n_features = n_features
		self.n_packets = n_packets
		self.num_classes = num_classes

		self.n_qubits = n_qubits
		self.lstm_layers = lstm_layers
		self.q_layers = q_layers
		self.n_shots = n_shots

		self.random_seed = random_seed
		self.dev_name = dev_name

		self.diff_method = diff_method

		projection = None
		if self.encoder_type == "dense":
			self.feature_extractor = nn.Flatten()
			input_dim = 2 ** self.n_qubits if self.embedding == 'amplitude' else self.n_qubits
			feature_dim = self.n_packets * self.n_features
			projection = nn.Linear(feature_dim, input_dim)
		else:
			self.feature_extractor = _CNNLSTMFeatureExtractor(
				self.n_features, self.n_packets, self.lstm_layers
			)
			feature_dim = self.feature_extractor.output_dim

		block = QuantumBlock(
			self.embedding, self.diff_method,
			self.q_layers, self.n_qubits, self.n_shots,
			random_seed, dev_name
		)

		if projection is None:
			projection = nn.Linear(feature_dim, block.input_dim)

		self.input_adapter = nn.Sequential(projection, self.input_activation())
		self.processing_block = block
		self.classifier = nn.Linear(block.output_dim, self.num_classes)

		# end

	# end class

#	####################################################################	#
#	Definizione dei modelli

class TrafficCNN(_FeatureClassifier):
	"""Baseline CNN1D."""

	model_label = "TrafficCNN"
	name_fields = ["n_features", "n_packets", "num_classes"]
	output_dim = 256
	is_quantum : bool = False

	def __init__(self, n_features: int, n_packets: int, num_classes: int):

		super().__init__(n_features, n_packets, num_classes)

		self.feature_extractor = nn.Sequential(
            nn.Conv1d(self.n_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
		)

		self.input_adapter = nn.Identity()
		self.processing_block = nn.Identity()

		self.classifier = nn.Sequential(
			nn.Flatten(), nn.Linear(256, 128), nn.ReLU(),
			nn.Dropout(0.5), nn.Linear(128, self.num_classes),
		)

		# end

	# end class

class CNNLSTMModel(_FeatureClassifier):
	"""Baseline originale."""

	model_label = "CNNLSTM"
	name_fields = ["n_features", "n_packets", "num_classes", "lstm_layers"]
	is_quantum : bool = False

	def __init__(
		self,
		n_features: int, n_packets: int, num_classes: int,
		lstm_layers: int
	):

		super().__init__(n_features, n_packets, num_classes)
		self.lstm_layers = lstm_layers

		self.feature_extractor = _CNNLSTMFeatureExtractor(
			self.n_features, self.n_packets, self.lstm_layers
		)

		self.input_adapter = nn.Identity()
		self.processing_block = nn.Identity()

		self.classifier = nn.Linear(
			self.feature_extractor.output_dim,
			self.num_classes
		)

		# end

	# end class

class AmpeDenseModel(_QuantumModel):
	"""Flatten -> Linear -> Sigmoid -> Amplitude/QNN/probs -> Linear."""

	model_label = 'AmpeDense'
	embedding = 'amplitude'
	encoder_type = 'dense'
	input_activation = nn.Sigmoid

	# end class

class AngeDenseModel(_QuantumModel):
	"""Flatten -> Linear -> ReLU -> Angle(X)/QNN/probs -> Linear."""

	model_label = 'AngeDense'
	embedding = 'angle'
	encoder_type = 'dense'
	input_activation = nn.ReLU

	# end class

class AmpeCNNLSTMModel(_QuantumModel):
	"""CNNLSTM -> Linear -> Sigmoid -> Amplitude/QNN/probs -> Linear."""

	model_label = 'AmpeCNNLSTM'
	embedding = 'amplitude'
	encoder_type = 'cnnlstm'
	input_activation = nn.Sigmoid

	# end class

class AngeCNNLSTMModel(_QuantumModel):
	"""CNNLSTM -> Linear -> Sigmoid -> Angle(X)/QNN/probs -> Linear."""
	
	model_label = 'AngeCNNLSTM'
	embedding = 'angle'
	encoder_type = 'cnnlstm'
	input_activation = nn.ReLU

	# end class

#	####################################################################	#
#	Selezione dinamica del modello

MODEL_REGISTRY = {
	"traffic": TrafficCNN,
	"baseline": CNNLSTMModel,
	# "resnet": ResNetModel,
	"ampe_fixed_probs": AmpeDenseModel,
	"ange_probs": AngeDenseModel,
	"hybrid_ampe_fixed_probs": AmpeCNNLSTMModel,
	"hybrid_ange_probs": AngeCNNLSTMModel,
}

def select_model(
		model_name: str,
		n_features: int, n_packets: int, num_classes: int,
		lstm_layers: int, q_layers: int, n_qubits: int, n_shots: int,
		random_seed: int, dev_name: str, diff_method: str = None
	) -> nn.Module:
	"""Seleziona e restituisce il modello specificato.

	Args:
		model_name (str): Nome del modello da selezionare.
		n_features (int): Numero di feature per pacchetto.
		n_packets (int): Numero di pacchetti nell'input.
		num_classes (int): Numero di classi per la classificazione.
		lstm_layers (int): Numero di layer LSTM (se applicabile).
		q_layers (int): Numero di layer nel circuito quantistico (se applicabile).
		n_qubits (int): Numero di qubit nel circuito quantistico (se applicabile).
		n_shots (int): Numero di ripetizioni per il circuito quantistico (se applicabile).
		random_seed (int): Seed per la generazione casuale (se applicabile).
		dev_name (str): Nome del dispositivo quantistico (se applicabile).
		diff_method (str): Metodo di differenziazione per il circuito quantistico (se applicabile). Default è None.

	Raises:
		NotImplementedError: Se il modello specificato non è supportato.

	Returns:
		nn.Module: Il modello selezionato.
	"""

	params = {
		"n_features": n_features,
		"n_packets": n_packets,
		"num_classes": num_classes,
	}

	try:
		model : _BaseModel = MODEL_REGISTRY[model_name]
	except KeyError as exc:
		raise NotImplementedError(
			f'Modello {model_name!r} non supportato.\n'
			f'Modelli disponibili: {", ".join(tuple(MODEL_REGISTRY))}.') from exc

	if model_name == "baseline":
		params.update({
			"lstm_layers": lstm_layers,
		})

	if model.is_quantum:
		params.update({
			"n_qubits": n_qubits,
			"lstm_layers": lstm_layers,
			"q_layers": q_layers,
			"n_shots": n_shots,
			"random_seed": random_seed,
			"dev_name": dev_name,
			"diff_method": diff_method,
		})
	
	return model(**params)

	# end
