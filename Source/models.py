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

from typing import TYPE_CHECKING
if TYPE_CHECKING:
	from Source.configuration import SessionConfig

#	####################################################################	#
#	Funzioni di utilità

@torch.no_grad()
def weight_init(m: nn.Module) -> None:
	"""Inizializzazione esplicita dei layer classici. Applica schemi di inizializzazione specifici per convoluzioni, lineari, batch norm e LSTM. La funzione non intercetta qml.qnn.TorchLayer, quindi non sovrascrive i pesi del circuito quantistico."""

	# Inizializzazione per layer convoluzionali.
	if isinstance(m, (nn.Conv1d, nn.Conv2d)):
		nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")

		if m.bias is not None:
			nn.init.zeros_(m.bias)

	# Inizializzazione per layer lineari.
	elif isinstance(m, nn.Linear):
		nonlinearity = getattr(m, "_init_nonlinearity", "linear",)

		if nonlinearity == "relu":
			nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
		else:
			nn.init.xavier_uniform_(m.weight, gain=1.0)

		if m.bias is not None:
			nn.init.zeros_(m.bias)

	# Inizializzazione per layer di normalizzazione batch.
	elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
		if m.affine:
			nn.init.ones_(m.weight)
			nn.init.zeros_(m.bias)

		if m.track_running_stats:
			m.reset_running_stats()

	# Inizializzazione per layer LSTM.
	elif isinstance(m, nn.LSTM):
		for name, parameter in m.named_parameters(recurse=False):

			# La presenza di due bias separati (bias_ih, bias_hh) richiede un'inizializzazione specifica per ciascun gate: Xavier sui pesi di input, ortogonale per gate sui pesi ricorrenti.
			if name.startswith("weight_ih"):
				for gate in parameter.chunk(4, dim=0):
					nn.init.xavier_uniform_(gate)

			elif name.startswith("weight_hh"):
				for gate in parameter.chunk(4, dim=0):
					nn.init.orthogonal_(gate)

			elif name.startswith("bias"):
				nn.init.zeros_(parameter)

				# Gate order: input, forget, cell, output.
				# Only bias_ih gets +1: total forget bias is 1, not 2.
				if name.startswith("bias_ih"):
					hidden_size = m.hidden_size
					parameter[hidden_size : 2 * hidden_size].fill_(1.0)

	else:
		# Per altri tipi di layer, non viene applicata alcuna inizializzazione specifica.
		pass

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
			"Fake quantum devices are not yet supported. \n"
			f"Available devices are: ['default.qubit']"
		)

	elif dev_name.startswith("ibm"):
		raise NotImplementedError(
			"IBM quantum devices are not yet supported. \n"
			f"Available devices are: ['default.qubit']"
		)

	else:
		raise NotImplementedError(
			f"Quantum device '{dev_name}' is not supported. \n"
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

	# end class

class _FeatureClassifier(_BaseModel):
	"""Un solo forward per feature -> adapter -> processing -> logits. I componenti vengono assegnati una sola volta, senza alias registrati.
	"""

	uses_lengths: bool = False

	def forward(self, x, lengths):

		if self.uses_lengths:
			x = self.feature_extractor(x, lengths)
		else:
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

	def __init__(self, n_features: int, n_packets: int, n_dim: int = 2):

		super().__init__()

		self.n_dim = n_dim
		self.n_features = n_features
		self.n_packets = n_packets

		if self.n_dim not in [1, 2]:
			raise ValueError("Il valore di n_dim può essere solo 1 o 2.")

		# Using padding='same' with even kernel lengths and odd dilation may require a zero-padded copy of the input be created.
		self.pad2d = nn.ZeroPad2d((0, 1, 1, 2))

		self.conv1 = \
			nn.Conv2d(1, 32, (4, 2), stride=1, padding=0) \
			if self.n_dim == 2 \
			else nn.Conv1d(self.n_features, 32, kernel_size=5, padding=2, bias=False)
		
		self.bn1 = nn.BatchNorm2d(32) if self.n_dim == 2 else nn.BatchNorm1d(32)

		self.conv2 = \
			nn.Conv2d(32, 64, (4, 2), stride=1, padding=0, bias=False) \
			if self.n_dim == 2 \
			else nn.Conv1d(32, 64, kernel_size=3, padding=1, bias=False)
		
		self.bn2 = nn.BatchNorm2d(64) if self.n_dim == 2 else nn.BatchNorm1d(64)

		self.relu = nn.ReLU()

		if self.n_dim == 2:
			with torch.no_grad():
				dummy = self.conv1.weight.new_zeros(1, 1, self.n_packets, self.n_features)
				dummy = self.conv1(self.pad2d(dummy))
				dummy = self.conv2(self.pad2d(dummy))

				batches, channels, packets, features = dummy.shape
				lstm_input_size = channels * features

			self.lstm = nn.LSTM(lstm_input_size, 100, batch_first=True)
			self.fc1 = nn.Linear(100, self.output_dim)

		# Per la TrafficCNN con n_dim == 1, non si utilizza la rete LSTM.
		else:
			self.fc1 = nn.Linear(64, self.output_dim)

		# Evita di applicare Kaiming anche al classificatore finale e alle proiezioni verso gli embedding.
		self.fc1._init_nonlinearity = "relu"

		# end

	def forward(self, x, lengths):

		#	############################################################	#
		#	Inizializzazione e validazione dell'input.

		if x.ndim != 3 or tuple(x.shape[1:]) != (self.n_packets, self.n_features):
			raise ValueError(
				"Errore in _CNNLSTMFeatureExtractor:\n"
				f"Atteso: (B,{self.n_packets},{self.n_features})\n"
				f"Ricevuto: {tuple(x.shape)}."
			)

		lengths = torch.as_tensor(lengths, dtype=torch.long, device=x.device)
		valid = torch.arange(x.size(1), device=x.device)[None, :] < lengths[:, None]

		# Azzera anche eventuali valori di padding modificati dal preprocessing.
		x = x.masked_fill(~valid.unsqueeze(-1), 0.0)

		#	############################################################	#
		#	Esecuzione del forward pass.

		out = x.unsqueeze(1) \
			if self.n_dim == 2 \
			else x.transpose(1, 2)

		conv_mask = valid[:, None, :, None] \
			if self.n_dim == 2 \
			else valid[:, None, :]

		# Dopo l'esecuzione di ogni blocco convoluzionale, applica la maschera per azzerare i valori di padding.
		out = self.pad2d(out) if self.n_dim == 2 else out
		out = self.conv1(out)
		out = self.relu(self.bn1(out))
		out = out.masked_fill(~conv_mask, 0.0)

		out = self.pad2d(out) if self.n_dim == 2 else out
		out = self.conv2(out)
		out = self.relu(self.bn2(out))
		out = out.masked_fill(~conv_mask, 0.0)

		if self.n_dim == 2:
			out = out.permute(0, 2, 1, 3).flatten(start_dim=2)
			out, _ = self.lstm(out)
			batch_indices = torch.arange(out.size(0), device=out.device)
			out = out[batch_indices, lengths - 1]
		else:
			# out = out.transpose(1, 2)
			out = out.sum(dim=-1) / lengths.to(out.dtype).unsqueeze(1)

		out = self.relu(self.fc1(out))

		return out
	
	# end class

#	####################################################################	#
#	Definizione del blocco di elaborazione quantistico

class QuantumBlock(nn.Module):
	"""Il blocco codifica un vettore classico in uno stato quantistico, applica un ansatz parametrico e restituisce le probabilità di misura sulla base computazionale, riscalate da un fattore apprendibile.

	## Struttura:
	x -> Embedding(x) -> StronglyEntanglingLayers(weights) -> probs -> probs * logit_scale
	"""

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

		# Controllo già eseguito in configuration.py
		self.diff_method = diff_method

		self.q_layers = q_layers
		self.n_qubits = n_qubits
		self.n_shots = n_shots

		self.input_dim = 2 ** self.n_qubits if embedding == 'amplitude' else self.n_qubits
		self.output_dim = 2 ** self.n_qubits

		# Fattore di scala apprendibile applicato alle probabilità in uscita dal circuito.
		self.logit_scale = nn.Parameter(torch.tensor(float(self.output_dim)))

		#	############################################################	#
		#	Definizione del QNode.

		self.dev = select_quantum_device(dev_name, self.n_qubits, random_seed)
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

	def extra_repr(self) -> str:

		def _format_op(op):
			args = []
			for p in getattr(op, "parameters", []):
				if hasattr(p, "shape"):
					args.append(f"shape={tuple(p.shape)}")
				else:
					args.append(repr(p))

			wires = list(op.wires)
			args.append(f"wires={wires}")

			# Aggiunge eventuali iperparametri (es. rotation='X' in AngleEmbedding)
			for k, v in getattr(op, "hyperparameters", {}).items():
				if k != "wires":
					args.append(f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}")

			join_str = ', \n    '
			return f"{op.name}(" \
				"\n    " \
				f"{join_str.join(args)}" \
				"\n  )"

			# end

		# 1. Recupero parametri registrati in PyTorch (weight_shapes)
		shapes = {name: tuple(p.shape) for name, p in self.torch_layer.named_parameters()}

		# 2. Ispezione dinamica del QNode tramite QuantumTape
		qnode = self.torch_layer.qnode
		dev = qnode.device

		# Costruisce il tape usando dummy tensor per ispezionare il circuito reale
		dummy_inputs = torch.full(
			(self.input_dim,),
			self.input_dim ** -0.5,
			dtype=torch.float32,
		)
		dummy_weights = {
			k: torch.zeros(p.shape, dtype=torch.float32) 
			for k, p in self.torch_layer.named_parameters()
		}

		try:
			tape = qnode.construct((dummy_inputs,), dummy_weights)
		except Exception as e:
			return f"<repr non disponibile: {e}>"

		# Estrae la feature map (prima operazione) e l'ansatz (operazioni successive)
		ops = tape.operations
		feature_map_str = _format_op(ops[0]) if ops else "None"
		ansatz_ops = [_format_op(op) for op in ops[1:]]
		ansatz_str = ", ".join(ansatz_ops) if ansatz_ops else "None"

		# Estrae il processo di misurazione
		meas_str = ", ".join(repr(m) for m in tape.measurements)

		dev_name = dev.short_name if hasattr(dev, "short_name") else getattr(dev, "name", str(dev))
		shots = tape.shots
		diff_method = getattr(qnode, "diff_method", self.diff_method)

		return (
			f"device='{dev_name}', "
			f"diff_method='{diff_method}', "
			f"shots={shots}, \n"
			f"  feature_map={feature_map_str}, \n"
			f"  ansatz={ansatz_str}, \n"
			f"  measurement={meas_str}, \n"
			f"  weight_shapes={shapes}"
		)

		# end

	@property
	def qnode(self):
		return self.torch_layer.qnode

		# end

	def forward(self, x):
		if x.ndim != 2 or x.shape[-1] != self.input_dim:
			raise ValueError(
				"Errore in QuantumBlock:\n"
				f"Atteso: (B,{self.input_dim})\n"
				f"Ricevuto: {tuple(x.shape)}."
			)

		# NOTA
		# Dato che le probabilità di uscita dal blocco quantistico sommano a 1 (media 1 / 2^n_qubits), senza riscalamento il classificatore lineare successivo riceve ingressi molto piccoli e i logit iniziali restano quasi uguali tra le classi.
		# Per ovviare a questo problema, moltiplichiamo le probabilità di uscita per un fattore di scala apprendibile (logit_scale), cioè che può poi essere adattato dall'ottimizzatore.

		return self.torch_layer(x) * self.logit_scale

	# end class

class _QuantumModel(_FeatureClassifier):
	"""Modello quantistico di base."""

	embedding: str
	is_quantum : bool = True

	def __init__(
			self,
			n_features: int, n_packets: int, num_classes: int,
			q_layers: int, n_qubits: int, n_shots: int,
			random_seed: int, dev_name: str,
			diff_method: str
		):

		super().__init__(n_features, n_packets, num_classes)

		self.uses_lengths = self.encoder_type == "cnnlstm"

		self.n_qubits = n_qubits
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
				self.n_features, self.n_packets, n_dim=2
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

class AngleActivation(nn.Module):
	"""Map classical features to angles in (-pi, pi)."""

	def forward(self, x):
		from math import pi

		return pi * torch.tanh(x)

		# end

	# end class

#	####################################################################	#
#	Definizione dei modelli

class TrafficCNN(_FeatureClassifier):
	"""Baseline CNN1D."""

	model_label = "TrafficCNN"
	encoder_type = 'cnn'
	uses_lengths = True
	output_dim = 100
	is_quantum : bool = False

	def __init__(self, n_features: int, n_packets: int, num_classes: int):

		super().__init__(n_features, n_packets, num_classes)

		self.feature_extractor = _CNNLSTMFeatureExtractor(
			self.n_features, self.n_packets, n_dim=1
		)

		self.input_adapter = nn.Identity()
		self.processing_block = nn.Identity()

		self.classifier = nn.Linear(
			self.feature_extractor.output_dim,
			self.num_classes
		)

		# end

	# end class

class CNNLSTMModel(_FeatureClassifier):
	"""Baseline originale."""

	model_label = "CNNLSTM"
	encoder_type = 'cnnlstm'
	uses_lengths = True
	is_quantum : bool = False

	def __init__(self, n_features: int, n_packets: int, num_classes: int):

		super().__init__(n_features, n_packets, num_classes)

		self.feature_extractor = _CNNLSTMFeatureExtractor(
			self.n_features, self.n_packets, n_dim=2
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

class AmpeCNNLSTMModel(_QuantumModel):
	"""CNNLSTM -> Linear -> Sigmoid -> Amplitude/QNN/probs -> Linear."""

	model_label = 'AmpeCNNLSTM'
	embedding = 'amplitude'
	encoder_type = 'cnnlstm'
	input_activation = nn.Sigmoid

	# end class

class AngeDenseModel(_QuantumModel):
	"""Flatten -> Linear -> pi*tanh -> Angle(X)/QNN/probs -> Linear."""

	model_label = 'AngeDense'
	embedding = 'angle'
	encoder_type = 'dense'
	input_activation = AngleActivation

	# end class

class AngeCNNLSTMModel(_QuantumModel):
	"""CNNLSTM -> Linear -> pi*tanh -> Angle(X)/QNN/probs -> Linear."""
	
	model_label = 'AngeCNNLSTM'
	embedding = 'angle'
	encoder_type = 'cnnlstm'
	input_activation = AngleActivation

	# end class

#	####################################################################	#
#	Selezione dinamica del modello

MODEL_REGISTRY = {
	"traffic": TrafficCNN,
	"baseline": CNNLSTMModel,
	# "resnet": ResNetModel,
	"ampe_fixed_probs": AmpeDenseModel,
	"hybrid_ampe_fixed_probs": AmpeCNNLSTMModel,
	"ange_probs": AngeDenseModel,
	"hybrid_ange_probs": AngeCNNLSTMModel,
}

def select_model(
		num_classes: int,
		random_seed: int,
		sc: "SessionConfig"
	) -> nn.Module:
	"""Seleziona e restituisce il modello specificato.

	Args:
		num_classes (int): Numero di classi per la classificazione.
		random_seed (int): Il seme casuale da utilizzare per la riproducibilità.
		sc (SessionConfig): Configurazione dei parametri di sessione per l'esperimento corrente.

	Raises:
		NotImplementedError: Se il modello specificato non è supportato.

	Returns:
		nn.Module: Il modello selezionato.
	"""

	params = {
		"n_features": sc.dataset.n_features,
		"n_packets": sc.dataset.n_packets,
		"num_classes": num_classes,
	}

	try:
		model : _BaseModel = MODEL_REGISTRY[sc.model.name]
	except KeyError as exc:
		raise NotImplementedError(
			f'Modello {sc.model.name} non supportato.\n'
			f'Modelli disponibili: {", ".join(tuple(MODEL_REGISTRY))}.') from exc

	if model.is_quantum:
		params.update({
			"random_seed": random_seed,
			"n_qubits": sc.model.quantum.n_qubits,
			"q_layers": sc.model.quantum.q_layers,
			"n_shots": sc.model.quantum.n_shots,
			"dev_name": sc.model.quantum.simulator,
			"diff_method": sc.model.quantum.diff_method,
		})
	
	return model(**params)

	# end
