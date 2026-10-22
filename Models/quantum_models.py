"""
	quantum_models.py
	di Mario Gabriele Carofano

	...
"""

#	####################################################################	#
#	Librerie

import torch
import pennylane as qml
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Source.model_selection import compute_model_name

#	####################################################################	#
#	Funzioni di utilità

def get_padding(kernel, padding='same'):
	pad = kernel - 1
	if padding == 'same':
		if kernel % 2:
			return pad // 2, pad // 2
		else:
			return pad // 2, pad // 2 + 1
	return 0, 0

	# end

def get_output_dim(dimension, kernels, strides, dilatation=1, padding='same', return_paddings=False):
	out_dim = dimension
	paddings = []
	if padding == 'same':
		for kernel, stride in zip(kernels, strides):
			paddings.append(get_padding(kernel, padding))
			out_dim = (out_dim + stride - 1) // stride
	else:
		for kernel, stride in zip(kernels, strides):
			paddings.append(get_padding(kernel, padding))
			out_dim = (out_dim - kernel + stride) // stride

	if return_paddings:
		return out_dim, paddings
	return out_dim

	# end

def select_quantum_device(dev_name, n_qubits, n_shots, random_seed):
	if(dev_name == "default.qubit"):
		return qml.device(
			dev_name,
			wires=n_qubits,
			shots=n_shots,
			seed=random_seed
		)

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

def meas_qoutputsize_mapping(n_qubits):
	return {
		"pauli": n_qubits,
		"probs": 2**n_qubits
	}

	# end

#	####################################################################	#
#	QNN with Amplitude Embedding

class AmpeDenseModel(nn.Module):
	def __init__(
			self,
			n_features, n_packets, num_classes,
			n_layers, n_qubits, n_shots,
			random_seed, dev_name
		):

		super(AmpeDenseModel, self).__init__()

		#	------------------------------------------------------------	#
		#	Inizializzazione degli attributi di classe

		self.n_features = n_features
		self.n_packets = n_packets
		self.num_classes = num_classes

		self.n_layers = n_layers
		self.n_qubits = n_qubits
		self.n_shots = n_shots

		self.random_seed = random_seed
		self.dev_name = dev_name

		self.dev = select_quantum_device(dev_name, n_qubits, n_shots, random_seed)

		#	------------------------------------------------------------	#
		#	Inizializzazione di altri parametri

		q_output_size = meas_qoutputsize_mapping(n_qubits)["probs"]
		weight_shapes = {"weights": (n_layers, n_qubits, 3)}
		input_dim = n_packets * n_features

		#	------------------------------------------------------------	#
		#	Definizione del QNode

		@qml.qnode(self.dev, interface="torch")
		def qnode(inputs, weights):
			# Feature map
			qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True, pad_with=0.0)

			# Ansatz
			qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

			# Measurement process
			return qml.probs(wires=range(n_qubits))

		self.qnode = qnode
		if self.dev_name.startswith("fake"): self.qnode = qml.set_shots(self.qnode, self.n_shots)

		#	------------------------------------------------------------	#
		#	Definizione dei layer classici e del layer quantistico

		self.flatten = nn.Flatten()

		self.dense1 = nn.Linear(input_dim, 2**n_qubits)
		self.sigmoid = nn.Sigmoid()

		self.q_layer = qml.qnn.TorchLayer(self.qnode, weight_shapes)

		self.dense2 = nn.Linear(q_output_size, num_classes)

		# end

	def forward(self, x):
		x = self.flatten(x)
		x = self.dense1(x)
		x = self.sigmoid(x)
		x = self.q_layer(x)

		return self.dense2(x)

		# end

	def get_model_name(self):
		return compute_model_name(
			"AmpeDense",
			n_qubits=self.n_qubits,
			n_layers=self.n_layers,
			n_packets=self.n_packets,
			n_features=self.n_features,
			num_classes=self.num_classes
		)

		# end

	def get_model_name_short(self):
		return compute_model_name(
			"AmpeDense",
			n_qubits=self.n_qubits,
			n_layers=self.n_layers
		)

		# end

	# end

#	####################################################################	#
#	QNN with Angle Embedding

class AngeDenseModel(nn.Module):
	def __init__(
			self,
			n_features, n_packets, num_classes,
			n_layers, n_qubits, n_shots,
			random_seed, dev_name
		):

		super(AngeDenseModel, self).__init__()

		#	------------------------------------------------------------	#
		#	Inizializzazione degli attributi di classe

		self.n_features = n_features
		self.n_packets = n_packets
		self.num_classes = num_classes

		self.n_layers = n_layers
		self.n_qubits = n_qubits
		self.n_shots = n_shots

		self.random_seed = random_seed
		self.dev_name = dev_name

		self.dev = select_quantum_device(dev_name, n_qubits, n_shots, random_seed)

		#	------------------------------------------------------------	#
		#	Inizializzazione di altri parametri

		q_output_size = meas_qoutputsize_mapping(n_qubits)["probs"]
		weight_shapes = {"weights": (n_layers, n_qubits, 3)}
		input_dim = n_packets * n_features

		#	------------------------------------------------------------	#
		#	Definizione del QNode

		@qml.qnode(self.dev, interface="torch")
		def qnode(inputs, weights):
			# Feature map
			qml.AngleEmbedding(inputs, wires=range(n_qubits))

			# Ansatz
			qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

			# Measurement process
			return qml.probs(wires=range(n_qubits))

		self.qnode = qnode
		if self.dev_name.startswith("fake"): self.qnode = qml.set_shots(self.qnode, self.n_shots)

		#	------------------------------------------------------------	#
		#	Definizione dei layer classici e del layer quantistico

		self.flatten = nn.Flatten()

		self.dense1 = nn.Linear(input_dim, n_qubits)
		self.relu = nn.ReLU()

		self.q_layer = qml.qnn.TorchLayer(self.qnode, weight_shapes)

		self.dense2 = nn.Linear(q_output_size, num_classes)

		# end

	def forward(self, x):
		x = self.flatten(x)
		x = self.dense1(x)
		x = self.relu(x)
		x = self.q_layer(x)

		return self.dense2(x)

		# end

	def get_model_name(self):
		return compute_model_name(
			"AngeDense",
			n_qubits=self.n_qubits,
			n_layers=self.n_layers,
			n_packets=self.n_packets,
			n_features=self.n_features,
			num_classes=self.num_classes
		)

		# end

	def get_model_name_short(self):
		return compute_model_name(
			"AngeDense",
			n_qubits=self.n_qubits,
			n_layers=self.n_layers
		)

		# end

	# end

#	####################################################################	#
#	Hybrid QNN with Amplitude Embedding and Convolutional LSTM Layers

# class AmpeCNNLSTMModel(nn.Module):
# 	raise NotImplementedError("AmpeCNNLSTMModel is not yet implemented")
