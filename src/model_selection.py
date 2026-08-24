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
