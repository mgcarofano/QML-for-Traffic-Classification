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
	from Source.configuration import SessionConfig, RuntimeConfig

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
	elif isinstance(m, nn.GroupNorm):
		if m.affine:
			nn.init.ones_(m.weight)
			nn.init.zeros_(m.bias)

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
			+ '\n'.join(problems)
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

def get_model_input_shape(
		sc: "SessionConfig",
		rc: "RuntimeConfig"
	) -> tuple[int, int]:
	"""Restituisce la forma dell'input del modello come (altezza, larghezza).

	Args:
		sc (SessionConfig): Configurazione dei parametri di sessione per l'esperimento corrente.
		rc (RuntimeConfig): Configurazione dei parametri di runtime per l'esperimento corrente.

	Returns:
		tuple[int, int]: La forma dell'input del modello (altezza, larghezza).
	"""

	# Per dataset di tipo Flowpic composto da istogrammi 2D, la dimensione dell'input è determinata dal rapporto tra MTU e bin_size.
	if rc.flags.is_flowpic:
		dim = sc.traffic.mtu // sc.traffic.bin_size
		return (dim, dim)

	# Per dataset di tipo Mirage composto da sequenze di pacchetti, la dimensione dell'input è determinata dal numero di pacchetti e dal numero di caratteristiche per pacchetto del biflusso.
	return (sc.dataset.n_packets, sc.dataset.n_features)

	# end

def mask_rows(x, lengths, n_dim=2):
	"""Applica la maschera per gestire i padding nelle sequenze di pacchetti.

	Args:
		x (torch.Tensor): Il tensore di input da mascherare.
		lengths (torch.Tensor): Il vettore delle lunghezze effettive delle sequenze.
		n_dim (int): Il numero di dimensioni spaziali del tensore (1 o 2). Default è 2.

	Returns:
		torch.Tensor: Il tensore mascherato, con i valori di padding azzerati.
	"""

	if lengths is None:
		return x

	valid = torch.arange(x.shape[2], device=x.device)[None, :] < lengths[:, None]
	mask = valid[:, None, :, None] if n_dim == 2 else valid[:, None, :]

	return x.masked_fill(~mask, 0.0)

	# end

#	####################################################################	#
#	Definizione delle classi base per i modelli

class _BaseModel(nn.Module):
	"""Base PyTorch con validazione dei metadati comuni."""

	is_quantum : bool = False
	supported_inputs = frozenset({"mirage", "flowpic"})

	def __init__(self, input_height, input_width, input_kind, num_classes):

		super().__init__()

		self.input_height = input_height
		self.input_width = input_width
		self.num_classes = num_classes

		if input_kind not in self.supported_inputs:
			raise ValueError(f"{type(self).__name__} non supporta {input_kind}.")
		self.input_kind = input_kind

	def _prepare_input(self, x, lengths):
		"""Prepara l'input per il modello, applicando eventuale padding e mascheramento.

		Args:
			x: il dataset in input, come tensore PyTorch.
			lengths: le lunghezze reali dei flussi Mirage, necessarie per il padding corretto.

		## Raises (per tutti i tipi di dataset):
			ValueError: Se il formato del dataset in input non è valido.
			ValueError: Se il batch è vuoto o non è floating point.

		## Raises (solo per dataset di tipo Mirage):
			ValueError: Se il vettore delle lunghezze è None.
			ValueError: Se il vettore delle lunghezze non ha la shape corretta.
			ValueError: Se le lunghezze non sono comprese nell'intervallo valido.

		Returns:
			tuple: Una tupla contenente il tensore preparato e le lunghezze (None per FlowPic).
		"""

		expected = \
			(self.input_height, self.input_width) \
			if self.input_kind == "mirage" \
			else (1, self.input_height, self.input_width)

		# CHECK 1: verifica tipo e forma del dataset in input.
		if not isinstance(x, torch.Tensor) or tuple(x.shape[1:]) != expected:
			raise ValueError(
				f"Formato del dataset in input non valido. \n"
				f"Atteso: (B, {expected}), "
				f"Ricevuto: {x.shape}."
			)

		# CHECK 2: verifica che il batch non sia vuoto e che sia floating point.
		if x.shape[0] == 0 or not x.is_floating_point():
			raise ValueError(
				"Errore nel caricamento del dataset in input: \n"
				"Il batch deve essere non vuoto e floating point."
			)

		# Per dataset di tipo FlowPic, non è necessario applicare padding o mascheramento in quanto ogni istogramma contiene sia i pacchetti reali che quelli di padding.
		if self.input_kind == "flowpic":
			return x, None

		# Invece per dataset di tipo Mirage, si applica padding e mascheramento per gestire le lunghezze variabili dei flussi e migliorare l'efficienza e l'accuratezza del modello.
		elif self.input_kind == "mirage":

			# CHECK 3: verifica che il vettore delle lunghezze sia valido.
			if lengths is None:
				raise ValueError(
					"Errore nel caricamento di 'lengths': \n"
					"Il vettore delle lunghezze è richiesto per i dataset di tipo Mirage."
				)

			lengths = torch.as_tensor(lengths, device=x.device, dtype=torch.long)

			# CHECK 4: verifica la forma del vettore delle lunghezze.
			if tuple(lengths.shape) != (x.shape[0],):
				raise ValueError(
					"Errore nel caricamento di 'lengths': \n"
					f"Atteso: ({x.shape[0]},), "
					f"Ricevuto: {lengths.shape}."
				)

			# CHECK 5: verifica che le lunghezze siano comprese nell'intervallo valido.
			if torch.any((lengths < 1) | (lengths > self.input_height)):
				raise ValueError(
					"Errore nel caricamento di 'lengths': \n"
					f"Trovate lunghezze non valide. Devono essere comprese in [1,{self.input_height}]."
				)

			x = mask_rows(x.unsqueeze(1), lengths, n_dim=2)

			# mask = torch.arange(self.input_height, device=x.device)[None, :] < lengths[:, None]
			# x = x.masked_fill(~mask[:, :, None], 0.0).unsqueeze(1)

			return x, lengths

			# end if self.input_kind

		# end

	# end class

class _FeatureClassifier(_BaseModel):
	"""Un solo forward per feature -> adapter -> processing -> logits. I componenti vengono assegnati una sola volta, senza alias registrati.
	"""

	def forward(self, x, lengths=None):

		input, lengths = self._prepare_input(x, lengths)

		features = self.feature_extractor(input, lengths)
		features = self.input_adapter(features)
		features = self.processing_block(features)

		return self.classifier(features)

		# end

	# end class

#	####################################################################	#
#	Definizione di blocchi BatchNorm sicuri
#	Consente l'utilizzo di batch normalization manuale quando il batch è troppo piccolo per evitare problemi di stima della media e della varianza.

class _SafeBatchNorm1d(nn.BatchNorm1d):

	def forward(self, x):

		if self.training and x.numel() // x.shape[1] == 1:
			return nn.functional.batch_norm(
				x, self.running_mean, self.running_var,
				self.weight, self.bias, training=False,
				momentum=0.0, eps=self.eps
			)

		return super().forward(x)

		# end

	# end class

class _SafeBatchNorm2d(nn.BatchNorm2d):

	def forward(self, x):

		if self.training and x.numel() // x.shape[1] == 1:
			return nn.functional.batch_norm(
				x, self.running_mean, self.running_var,
				self.weight, self.bias, training=False,
				momentum=0.0, eps=self.eps
			)

		return super().forward(x)

		# end

	# end class

#	####################################################################	#
#	Definizione degli estrattori di feature

class _DenseFeatureExtractor(nn.Module):

	def forward(self, input, lengths=None):

		out = input.flatten(start_dim=1)
		return out

		# end

	# end class

class _CNNLSTMFeatureExtractor(nn.Module):
	"""Due Conv2d + LSTM + Linear, uscita (B, 100)."""

	output_dim = 100

	def __init__(self, input_height, input_width, input_kind, n_dim=2):

		super().__init__()

		if n_dim not in (1, 2):
			raise ValueError("Il valore di n_dim può essere solo 1 o 2.")
		if n_dim == 1 and input_kind != "mirage":
			raise ValueError("La CNN1D e' riservata alle sequenze Mirage.")
		self.n_dim = n_dim
		self.input_kind = input_kind

		# Using padding='same' with even kernel lengths and odd dilation may require a zero-padded copy of the input be created.
		self.pad2d = nn.ZeroPad2d((0, 1, 1, 2))

		spatial_width = input_width if input_kind == "mirage" else input_height

		# Configurazione del modello per convoluzioni 2D (es. CNNLSTMModel)
		if n_dim == 2:
			self.conv1 = nn.Conv2d(1, 32, (4, 2), bias=True)
			self.bn1 = _SafeBatchNorm2d(32)
			self.conv2 = nn.Conv2d(32, 64, (4, 2), bias=False)
			self.bn2 = _SafeBatchNorm2d(64)
			self.lstm = nn.LSTM(64 * spatial_width, 100, batch_first=True)
			self.fc1 = nn.Linear(100, self.output_dim)

		# Configurazione del modello per convoluzioni 1D senza LSTM (es. TrafficCNN)
		elif n_dim == 1:
			self.conv1 = nn.Conv1d(input_width, 32, 5, padding=2, bias=False)
			self.bn1 = _SafeBatchNorm1d(32)
			self.conv2 = nn.Conv1d(32, 64, 3, padding=1, bias=False)
			self.bn2 = _SafeBatchNorm1d(64)
			self.fc1 = nn.Linear(64, self.output_dim)

		self.relu = nn.ReLU()

		# Evita di applicare Kaiming anche al classificatore finale e alle proiezioni verso gli embedding.
		self.fc1._init_nonlinearity = "relu"

		# end

	def forward(self, input, lengths=None):

		#	############################################################	#
		#	Inizializzazione.

		# Trasformazione dell'input in base al tipo di input e al numero di dimensioni (1D o 2D).
		if self.n_dim == 1:
			out = input[:, 0].transpose(1, 2)
		elif self.input_kind == "flowpic":
			out = input.transpose(-1, -2).contiguous()
		else:
			out = input

		# mask = None
		# if self.input_kind == "mirage" and lengths is not None:
		# 	valid = torch.arange(out.shape[2], device=out.device)[None, :] < lengths[:, None]
		# 	mask = valid[:, None, :, None] if self.n_dim == 2 else valid[:, None, :]

		#	############################################################	#
		#	Esecuzione del forward pass.

		for conv, bn in ((self.conv1, self.bn1), (self.conv2, self.bn2)):

			# Applica il padding solo per convoluzioni 2D.
			if self.n_dim == 2:
				out = self.pad2d(out)

			out = conv(out)
			out = bn(out)
			out = self.relu(out)

			# Dopo l'esecuzione di ogni blocco convoluzionale, applica la maschera per azzerare i valori di padding.
			# if mask is not None:
			# 	out = out.masked_fill(~mask, 0.0)
			out = mask_rows(out, lengths, n_dim=self.n_dim)

			# end for conv, bn

		# Esecuzione del blocco LSTM per convoluzioni 2D o aggregazione per convoluzioni 1D.
		if self.n_dim == 2:
			out = out.permute(0, 2, 1, 3).flatten(start_dim=2)
			out, _ = self.lstm(out)

			# Se è disponibile il vettore delle lunghezze, taglia l'output alla lunghezza effettiva di ciascuna sequenza.
			# Migliora l'efficienza evitando di considerare i pacchetti di padding per la classificazione finale.
			if lengths is not None:
				batch_indices = torch.arange(out.size(0), device=out.device)
				out = out[batch_indices, lengths - 1]
			else:
				out = out[:, -1, :]

		elif self.n_dim == 1:
			out = out.sum(dim=-1) / lengths.to(out.dtype).unsqueeze(1)

			# end if

		out = self.relu(self.fc1(out))

		return out

		# end

	# end class

class _ResidualBlock2D(nn.Module):
	"""Due Conv2D 3x3 + ReLU. Utilizza una shortcut 1x1 se cambia la shape dell'input."""

	def __init__(self, in_channels, out_channels, stride=1):

		super().__init__()

		self.stride = stride

		self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
		self.norm1 = nn.GroupNorm(8, out_channels)
		self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
		self.norm2 = nn.GroupNorm(8, out_channels)

		self.shortcut = (
			nn.Identity() if stride == 1 and in_channels == out_channels
			else nn.Sequential(
				nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
				nn.GroupNorm(8, out_channels)
			)
		)

		self.relu = nn.ReLU()

		# end

	def forward(self, x, lengths=None):

		next_lengths = (lengths + self.stride - 1) // self.stride if lengths is not None else None

		out = self.conv1(x)
		out = self.norm1(out)
		out = self.relu(out)
		out = mask_rows(out, next_lengths)

		out = self.conv2(out)
		out = self.norm2(out)
		out = self.relu(out + self.shortcut(x))
		out = mask_rows(out, next_lengths)

		return out, next_lengths

		# end

	# end class

class _ResNetFeatureExtractor(nn.Module):
	"""ResNet18 adattata: un canale, stem 3x3/s1, no maxpool, GroupNorm."""

	output_dim = 512

	def __init__(self):
		super().__init__()

		self.stem = nn.Sequential(
			nn.Conv2d(1, 64, 3, padding=1, bias=False),
			nn.GroupNorm(8, 64), nn.ReLU()
		)

		blocks = []
		current = 64
		for stage, channels in enumerate((64, 128, 256, 512)):
			for block_index in range(2):

				is_first_block = stage > 0 and block_index == 0
				stride = 2 if is_first_block else 1

				blocks.append(_ResidualBlock2D(current, channels, stride))
				current = channels

				# end for

		self.blocks = nn.ModuleList(blocks)
		self.pool = nn.AdaptiveAvgPool2d((1, 1))

		# end

	def forward(self, input, lengths=None):

		out = self.stem(input)
		out = mask_rows(out, lengths)

		for block in self.blocks:
			out, lengths = block(out, lengths)

		if lengths is None:
			return self.pool(out).flatten(start_dim=1)

		# Media solo sulle righe valide e su tutta la larghezza finale.
		spatial_width = out.shape[3]
		out = out.sum(dim=(2, 3))
		valid_counts = lengths.to(out.dtype) * spatial_width
		return out / valid_counts[:, None]

		# end

	# end class

#	####################################################################	#
#	Definizione del blocco di elaborazione quantistico

class QuantumBlock(nn.Module):
	"""Il blocco codifica un vettore classico in uno stato quantistico, applica un ansatz parametrico e restituisce le probabilità di misura sulla base computazionale, riscalate da un fattore apprendibile.

	## Struttura:
	x -> Embedding(x) -> StronglyEntanglingLayers(weights) -> probs -> probs * logit_scale
	"""

	def __init__(
			self, embedding, diff_method,
			q_layers, n_qubits, n_shots,
			random_seed, dev_name
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
			dtype=self.logit_scale.dtype,
			device=self.logit_scale.device,
		)
		dummy_weights = {
			k: torch.zeros_like(p)
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

		# Utilizzare la Sigmoid potrebbe causare underflow o divisioni per zero. Per ovviare a questo problema, si limitano i valori del tensore in ingresso al più piccolo intero positivo rappresentabile dal tipo di dato del tensore.
		if self.embedding == "amplitude":
			min_positive_val = torch.finfo(x.dtype).eps
			x = x.clamp_min(min_positive_val)

		return self.torch_layer(x).to(dtype=x.dtype, device=x.device) * self.logit_scale

	# end class

class _QuantumModel(_FeatureClassifier):
	"""Encoder classico -> proiezione -> QNN -> classificatore classico."""

	is_quantum : bool = True
	embedding: str

	def __init__(self, input_height, input_width, input_kind, num_classes,
			q_layers, n_qubits, n_shots,
			random_seed, dev_name, diff_method,
		):

		super().__init__(input_height, input_width, input_kind, num_classes)

		if self.encoder_type not in ("dense", "cnnlstm"):
			raise ValueError(
				f"Encoder sconosciuto: {self.encoder_type}."
				"Opzioni disponibili: 'dense', 'cnnlstm'."
			)

		self.n_qubits = n_qubits
		self.q_layers = q_layers
		self.n_shots = n_shots

		self.random_seed = random_seed
		self.dev_name = dev_name

		self.diff_method = diff_method

		if self.encoder_type == "dense":
			self.feature_extractor = _DenseFeatureExtractor()
			feature_dim = self.input_height * self.input_width

		elif self.encoder_type == "cnnlstm":
			self.feature_extractor = _CNNLSTMFeatureExtractor(
				self.input_height,
				self.input_width,
				self.input_kind
			)

			feature_dim = self.feature_extractor.output_dim

		block = QuantumBlock(
			self.embedding,
			diff_method, q_layers, n_qubits,
			n_shots, random_seed, dev_name
		)

		self.input_adapter = nn.Sequential(
			nn.Linear(feature_dim, block.input_dim),
			self.input_activation()
		)

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
	"""Baseline CNN1D. Supporta solo dataset di tipo Mirage."""

	model_label = "TrafficCNN"
	encoder_type = "cnn"
	supported_inputs = frozenset({"mirage"})

	def __init__(self, input_height, input_width, input_kind, num_classes):

		super().__init__(input_height, input_width, input_kind, num_classes)

		self.feature_extractor = _CNNLSTMFeatureExtractor(
			self.input_height,
			self.input_width,
			self.input_kind,
			n_dim=1
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
	"""Baseline CNN2D. Funziona sia per dataset di tipo Mirage sia per dataset di tipo FlowPic."""

	model_label = "CNNLSTM"
	encoder_type = "cnnlstm"

	def __init__(self, input_height, input_width, input_kind, num_classes):

		super().__init__(input_height, input_width, input_kind, num_classes)

		self.feature_extractor = _CNNLSTMFeatureExtractor(
			self.input_height,
			self.input_width,
			self.input_kind
		)

		self.input_adapter = nn.Identity()
		self.processing_block = nn.Identity()
		self.classifier = nn.Linear(
			self.feature_extractor.output_dim,
			self.num_classes
		)

		# end

	# end class

class ResNetModel(_FeatureClassifier):
	""" Classificatore per traffico di rete basato su backbone ResNet + Linear."""

	model_label = "ResNet18Traffic"
	encoder_type = "resnet"

	def __init__(self, input_height, input_width, input_kind, num_classes):

		super().__init__(input_height, input_width, input_kind, num_classes)

		self.feature_extractor = _ResNetFeatureExtractor()
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

	model_label = "AmpeDense"
	embedding = "amplitude"
	encoder_type = "dense"
	input_activation = nn.Sigmoid

	# end class

class AmpeCNNLSTMModel(_QuantumModel):
	"""CNNLSTM -> Linear -> Sigmoid -> Amplitude/QNN/probs -> Linear."""

	model_label = "AmpeCNNLSTM"
	embedding = "amplitude"
	encoder_type = "cnnlstm"
	input_activation = nn.Sigmoid

	# end class

class AngeDenseModel(_QuantumModel):
	"""Flatten -> Linear -> pi*tanh -> Angle(X)/QNN/probs -> Linear."""

	model_label = "AngeDense"
	embedding = "angle"
	encoder_type = "dense"
	input_activation = AngleActivation

	# end class

class AngeCNNLSTMModel(_QuantumModel):
	"""CNNLSTM -> Linear -> pi*tanh -> Angle(X)/QNN/probs -> Linear."""

	model_label = "AngeCNNLSTM"
	embedding = "angle"
	encoder_type = "cnnlstm"
	input_activation = AngleActivation

	# end class

#	####################################################################	#
#	Selezione dinamica del modello

MODEL_REGISTRY = {
	"traffic": TrafficCNN,
	"baseline": CNNLSTMModel,
	"resnet": ResNetModel,
	"ampe_fixed_probs": AmpeDenseModel,
	"hybrid_ampe_fixed_probs": AmpeCNNLSTMModel,
	"ange_probs": AngeDenseModel,
	"hybrid_ange_probs": AngeCNNLSTMModel,
}

def select_model(
		num_classes: int,
		random_seed: int,
		sc: "SessionConfig",
		rc: "RuntimeConfig",
	) -> nn.Module:
	"""Seleziona e restituisce il modello specificato.

	Args:
		num_classes (int): Numero di classi per la classificazione.
		random_seed (int): Il seme casuale da utilizzare per la riproducibilità.
		sc (SessionConfig): Configurazione dei parametri di sessione per l'esperimento corrente.
		rc (RuntimeConfig): Configurazione dei parametri di runtime per l'esperimento corrente.

	Raises:
		NotImplementedError: Se il modello specificato non è supportato.

	Returns:
		nn.Module: Il modello selezionato.
	"""

	try:
		model_class : _BaseModel = MODEL_REGISTRY[sc.model.name]
	except KeyError as exc:
		raise NotImplementedError(
			f'Modello {sc.model.name} non supportato.\n'
			f'Modelli disponibili: {", ".join(tuple(MODEL_REGISTRY))}.') from exc

	height, width = get_model_input_shape(sc, rc)
	input_kind = "flowpic" if rc.flags.is_flowpic else "mirage"

	if input_kind not in model_class.supported_inputs:
		raise ValueError(
			f"Il modello {sc.model.name} non supporta '{input_kind}'. \n"
			f"Input disponibili: {sorted(model_class.supported_inputs)}"
		)
	
	params = dict(
		input_height= height,
		input_width= width,
		num_classes= num_classes,
		input_kind= input_kind
	)

	if model_class.is_quantum:
		params.update(
			random_seed= random_seed,
			n_qubits= sc.model.quantum.n_qubits,
			q_layers= sc.model.quantum.q_layers,
			n_shots= sc.model.quantum.n_shots,
			dev_name= sc.model.quantum.simulator,
			diff_method= sc.model.quantum.diff_method,
		)

	return model_class(**params)

	# end
