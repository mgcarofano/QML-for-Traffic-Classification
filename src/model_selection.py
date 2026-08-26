"""

    model_selection.py
    di Mario Gabriele Carofano

    Questo modulo contiene funzioni per la selezione dinamica del modello
	e il controllo della compatibilità dei pesi salvati.

"""

#   ########################################################################    #
#   LIBRERIE

import torch.nn as nn

import importlib
import os
import sys

from constants import MODEL_REGISTRY

#   ########################################################################    #

def compute_model_name(
		model_class : str,
		n_qubits : int = None,
		n_layers : int = None,
		n_packets : int = None,
		n_features : int = None,
		num_classes : int = None
	) -> str:
	"""
	Restituisce una stringa con il nome del modello che riassume i parametri principali,
	se disponibili, nel formato "class_Qn_Ln_PxF_Cc"

	Args:
		model_class (str): Nome della classe del modello.
		n_qubits (int, optional): Numero di qubit nel circuito quantistico. Default è None.
		n_layers (int, optional): Numero di layer nel circuito quantistico. Default è None.
		n_packets (int, optional): Numero di pacchetti nell'input. Default è None.
		n_features (int, optional): Numero di feature per pacchetto. Default è None.
		num_classes (int, optional): Numero di classi per la classificazione. Default è None.

	Returns:
		str: Nome del modello.
	"""

	ret = f"{model_class}"

	if n_qubits is not None:
		ret += f"_Q{n_qubits}"

	if n_layers is not None:
		ret += f"_L{n_layers}"

	if n_packets is not None:
		ret += f"_P{n_packets}"

	if n_features is not None:
		ret += f"_F{n_features}"

	if num_classes is not None:
		ret += f"_C{num_classes}"

	return ret

	# end

def get_model_class(model_name):
	"""Restituisce la classe del modello corrispondente al nome specificato.

	Args:
		model_name (str): Nome del modello da recuperare.

	Returns:
		class: Classe del modello corrispondente.

	Raises:
		ValueError: Se il nome del modello non è valido.
	"""

	if model_name not in MODEL_REGISTRY:
		available = ", ".join(MODEL_REGISTRY.keys())
		raise ValueError(f"Modello '{model_name}' non valido.\n\nOpzioni: {available}")

		# end if

	# Import dinamico del modulo e della classe del modello selezionato.
	module_name, class_name = MODEL_REGISTRY[model_name]
	sys.path.append(os.path.abspath('..'))
	clean_module_name = module_name.removeprefix('../').replace('/', '.')
	module = importlib.import_module(clean_module_name)

	# Si ritorna la classe corrispondente al modello selezionato.
	return getattr(module, class_name)
	
	# end

def check_model_compatibility(model, weights):
	"""Controlla la compatibilità tra il modello corrente e i pesi salvati.

	Args:
		model (torch.nn.Module): Modello corrente.
		weights (dict): Pesi salvati del modello.

	Raises:
		RuntimeError: Se le shape dei pesi salvati non corrispondono a quelle del modello corrente.
	"""

	curr_weights_shape = [
		v.shape
		for k, v in model.state_dict().items()
		if "classifier" in k and k.endswith(".weight") and v.dim() == 2
	]

	saved_weights_shape = [
		v.shape
		for k, v in weights.items()
		if "classifier" in k and k.endswith(".weight") and v.dim() == 2
	]

	mismatch_index = next(
		(
			i
			for i, (current_shape, saved_shape) in enumerate(
				zip(curr_weights_shape, saved_weights_shape)
			)
			if current_shape != saved_shape
		),
		None
	)

	if mismatch_index is not None:
		current_shape = curr_weights_shape[mismatch_index]
		saved_shape = saved_weights_shape[mismatch_index]
		model_name = type(model).__name__

		raise RuntimeError(
			f"Errore durante il caricamento del modello: {model_name}\n\n"
			f"Shape corrente diversa da quella salvata, in posizione ({mismatch_index}).\n"
			f"Il checkpoint è stato salvato con un numero di classi ({current_shape}) diverso "
			f"rispetto al modello corrente ({saved_shape}).\n"
			"Verifica che il dataset e il parametro num_classes usati per il "
			"training coincidano con quelli attuali."
		)

		# end if
	# end

def weight_init(m: nn.Module) -> None:
	"""Fills-in weights and biases for convolutional and linear layers. Uses Kaiming uniform initialization for weights (suitable for ReLU/non-linearities) and sets biases to zero. This function is meant to be used with model.apply(), which will call it on every sub-module.

	Args:
		m (nn.Module): a module of the model.
	"""	

	if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):

		# Good idea would be to use kaiming initialization of scaled orthonormal initialization.
		nn.init.kaiming_uniform_(
			m.weight,
			a=0,
			mode='fan_in',
			nonlinearity='leaky_relu'
		)

		# Check if m.bias is enabled.
		if m.bias is not None:
			nn.init.zeros_(m.bias)
	
	return

	# end

def get_lstm_layer():
	"""In precedenza si usava get_output_dim()/get_padding() per calcolare
	analiticamente le dimensioni H e W dopo le due Conv2d, ma la formula
	non teneva conto del padding realmente applicato ai layer (padding=0,
	hardcoded sopra), causando un disallineamento e un valore di
	input_size hardcoded (256) che si rompeva cambiando n_packets / n_features.
	"""

	#   ####################################################################    #
	#   OLD CODE

	# lstm_input_size = features_size1 * filters[1] # 256 (?)

	# # for padding in ['valid', 'same']:
	# features_size0, self.paddings0 = get_output_dim(
	# 	n_packets,
	# 	kernels=[kernel[0], kernel[0]],
	# 	strides=[stride[0], stride[0]],
	# 	padding='valid',
	# 	return_paddings=True
	# )

	# features_size1, self.paddings1 = get_output_dim(
	# 	n_features,
	# 	kernels=[kernel[1], kernel[1]],
	# 	strides=[stride[1], stride[1]],
	# 	padding='valid',
	# 	return_paddings=True
	# )

	pass

	# end
