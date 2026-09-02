"""

    visualization_functions.py \n
    di Mario Gabriele Carofano

    Questo modulo contiene funzioni per la visualizzazione dei risultati
	di training e validazione dei modelli ML, in particolare per la
	generazione di grafici che mostrano l'andamento di loss e accuracy
	in funzione delle epoche di training.

"""

#   ####################################################################    #
#   LIBRERIE

import numpy as np
import matplotlib.pyplot as plt

#   ####################################################################    #
#   FUNZIONI DI VISUALIZZAZIONE

def get_plot_epoch_ticks(
		num_epochs : int,
		best_epoch_num: int
	) -> tuple[np.ndarray, list[str]]:
	"""Genera i tick per per l'asse X di un grafico
	che mostra l'andamento di loss/accuracy in funzione delle epoche.

	I tick includono sempre:
    - la prima epoca;
    - la miglior epoca;
    - L'ultima epoca;
    - ulteriori valori intermedi uniformemente distribuiti.

	Args:
		num_epochs (int): Numero totale di epoche.
		best_epoch_num (int): Numero dell'epoca con la migliore performance.

	Returns:
		tuple: Una tupla contenente gli indici dei tick e le etichette corrispondenti.
	"""

	# Il totale numero di tick da visualizzare è limitato a 10.
	num_ticks = min(10, num_epochs)

	# Si distribuiscono uniformemente i tick,
	# includendo sempre la prima e l'ultima epoca.
	tick_indices = np.linspace(
		start=1, stop=num_epochs,
		num=num_ticks,
		dtype=int
	)

	# Si inserisce manualmente l'epoca con la migliore performance.
	tick_indices = np.unique(
		np.append(tick_indices, best_epoch_num)
	)

	# Se l'inserimento della best epoch porta a più di 10 tick,
	# si rimuove il tick intermedio più vicino alla best epoch,
	# mantenendo sempre la prima, la best e l'ultima epoca.
	while len(tick_indices) > num_ticks:
		removable = [
			tick for tick in tick_indices
			if tick not in {1, best_epoch_num, num_epochs}
		]

		if not removable:
			break

		tick_to_remove = min(
			removable,
			key=lambda tick: abs(tick - best_epoch_num)
		)

		tick_indices = tick_indices[tick_indices != tick_to_remove]

		# end while

	tick_indices = np.sort(tick_indices)
	tick_labels = [str(epoch) for epoch in tick_indices]

	return tick_indices, tick_labels

	# end

def get_plot_metric_ticks(
    min_value : float,
	max_value : float,
	best_value : float,
) -> tuple[np.ndarray, list[str]]:
	"""Genera i tick per per l'asse Y di un grafico
	che mostra l'andamento di loss/accuracy in funzione delle epoche.

	I tick includono sempre:
    - il valore minimo della metrica;
    - il valore della metrica nella best epoch;
    - il valore massimo della metrica;
    - ulteriori valori intermedi uniformemente distribuiti.

    Args:
        metric_history (list[float]): Valori storici della metrica, uno per epoca.
        best_epoch_num (int): Numero dell'epoca migliore, in formato 1-based.

    Returns:
        tuple[np.ndarray, list[str]]: Array dei valori dei tick e relative
        etichette formattate.
    """

	# Caso limite: tutti i valori della metrica sono vicinissimi.
	if np.isclose(min_value, max_value):
		tick_values = np.array([min_value])
		tick_labels = [f"{min_value:.4f}"]

		return tick_values, tick_labels

	# Il totale numero di tick da visualizzare è limitato a 10,
	# uniformemente distribuiti tra minimo e massimo.
	max_num_ticks = 10
	tick_values = np.linspace(
		start=min_value,
		stop=max_value,
		num=max_num_ticks
	)

	# Si inserisce il valore della metrica relativo alla best epoch.
	tick_values = np.unique(
		np.append(tick_values, best_value)
	)

	# Se sono presenti più di 10 tick, rimuove quello intermedio
	# più vicino al valore della best epoch, preservando minimo,
	# valore best epoch e massimo.
	while len(tick_values) > max_num_ticks:
		removable = [
			tick for tick in tick_values
			if not np.isclose(tick, min_value)
			and not np.isclose(tick, best_value)
			and not np.isclose(tick, max_value)
		]

		if not removable:
			break

		tick_to_remove = min(
			removable,
			key=lambda tick: abs(tick - best_value)
		)

		tick_values = tick_values[~np.isclose(tick_values, tick_to_remove)]

		# end while

	# Si ordinano i tick in ordine crescente.
	tick_values = np.sort(tick_values)

	# Per loss e accuracy, 4 decimali costituiscono un buon default.
	tick_labels = [f"{tick:.4f}" for tick in tick_values]

	return tick_values, tick_labels

	# end

def draw_train_val_plot(
		p : plt.Axes,
		metric_name : str,
		train_values: list[float],
		val_values: list[float],
		best_epoch: int
	) -> None:

	num_epochs = len(train_values)
	epochs_range = range(1, num_epochs + 1)
	best_value = float(val_values[best_epoch -1])

	p.plot(epochs_range, train_values, label=f'Train {metric_name}')
	p.plot(epochs_range, val_values, label=f'Val {metric_name}')
	p.set_title(f'{metric_name} over Epochs')
	p.set_xlabel('Epoch')
	p.set_ylabel(metric_name)
	p.legend()

	p.axvline(
		best_epoch,
		color='red', linestyle='--', dashes=(10, 15), linewidth=0.7,
		# label=f'Best epoch ({best_epoch})'
	)

	p.axhline(
		best_value,
		color='red', linestyle='--', dashes=(10, 15), linewidth=0.7,
		# label=f'Best epoch ({best_epoch})'
	)

	epoch_indices, epoch_labels = get_plot_epoch_ticks(num_epochs, best_epoch)

	p.set_xticks(epoch_indices)
	p.set_xticklabels(epoch_labels)
	p.set_xlim(0.5, num_epochs + 0.5)

	metric_indices, metric_labels = get_plot_metric_ticks(
		float(np.min(train_values + val_values)),
		float(np.max(train_values + val_values)),
		best_value
	)

	margin = (metric_indices[-1] - metric_indices[0]) * 0.05
	p.set_yticks(metric_indices)
	p.set_yticklabels(metric_labels)
	p.set_ylim(metric_indices[0] - margin, metric_indices[-1] + margin)

	# end
