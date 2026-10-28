"""
    visualization_functions.py \n
    di Mario Gabriele Carofano

    Questo modulo contiene funzioni per la visualizzazione dei risultati
	di training e validazione dei modelli ML, in particolare per la
	generazione di grafici che mostrano l'andamento di loss e accuracy
	in funzione delle epoche di training.
"""

#   ####################################################################    #
#   Librerie

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

#   ####################################################################    #
#   Funzioni di utilità

def configure_metric_y_ticks(
		p: plt.Axes,
		minimum: float,
		maximum: float,
		selected_value: float,
		max_ticks: int = 8,
		min_gap_fraction: float = 0.10,
	) -> None:
	"""Configura i tick Y preservando minimo, massimo e checkpoint.

	I tick intermedi troppo vicini a quelli già selezionati
	vengono esclusi.

	Se due tick obbligatori sono vicini, le loro etichette
	vengono allineate in direzioni opposte per ridurre
	la sovrapposizione.

	I valori coincidenti sono rappresentati da un solo tick.
	"""

	values = np.asarray(
		[minimum, maximum, selected_value],
		dtype=float,
	)

	if not np.all(np.isfinite(values)):
		raise ValueError("I valori dei tick devono essere finiti.")

	if not minimum <= selected_value <= maximum:
		raise ValueError(
			"selected_value deve essere compreso "
			"tra minimum e maximum."
		)

	if max_ticks < 3:
		raise ValueError("max_ticks deve essere almeno 3.")

	if not 0 < min_gap_fraction < 0.5:
		raise ValueError(
			"min_gap_fraction deve essere compreso tra 0 e 0.5."
		)

	# Conserva i limiti impostati dal grafico.
	y_min, y_max = p.get_ylim()

	# np.unique elimina soltanto i valori esattamente coincidenti.
	# Non usiamo np.isclose(): potrebbe eliminare un estremo
	# distinto ma molto vicino al valore del checkpoint.
	mandatory_ticks = np.unique(values)

	data_span = maximum - minimum

	if data_span == 0:
		ticks = mandatory_ticks

	else:
		min_gap = min_gap_fraction * data_span

		# Candidati intermedi con valori "regolari".
		locator = MaxNLocator(nbins=max_ticks - 1)

		candidates = locator.tick_values(minimum, maximum)
		candidates = candidates[
			(candidates > minimum)
			& (candidates < maximum)
		]

		selected_ticks = mandatory_ticks.tolist()

		for candidate in candidates:
			if len(selected_ticks) >= max_ticks:
				break

			distances = np.abs(
				np.asarray(selected_ticks) - candidate
			)

			if np.all(distances >= min_gap):
				selected_ticks.append(float(candidate))

		ticks = np.sort(np.asarray(selected_ticks))

	# Aumenta la precisione, se necessario, affinché valori
	# distinti non ricevano la stessa etichetta arrotondata.
	labels = None

	for precision in range(6, 18):
		candidate_labels = [
			f"{value:.{precision}g}"
			for value in ticks
		]

		if len(set(candidate_labels)) == len(candidate_labels):
			labels = candidate_labels
			break

	if labels is None:
		labels = [repr(float(value)) for value in ticks]

	# Fissa insieme posizioni ed etichette.
	p.set_yticks(ticks, labels=labels)

	# set_yticks() può espandere i limiti: ripristinali.
	p.set_ylim(y_min, y_max)

	tick_labels = p.get_yticklabels()

	for value, label in zip(ticks, tick_labels):
		label.set_verticalalignment("center")
		label.set_color("black")
		label.set_fontweight("normal")

		if value == selected_value:
			label.set_color("red")
			label.set_fontweight("bold")

	# Gestione delle coppie di tick obbligatori vicini.
	# I tick intermedi sono già stati filtrati.
	if data_span > 0:
		min_gap = min_gap_fraction * data_span

		for index in range(len(ticks) - 1):
			distance = ticks[index + 1] - ticks[index]

			if distance < min_gap:
				# Il testo del valore inferiore si sviluppa
				# sotto il suo tick; quello superiore sopra.
				tick_labels[index].set_verticalalignment("top")
				tick_labels[index + 1].set_verticalalignment("bottom")

	# end

#   ####################################################################    #
#   Funzioni principali

def draw_train_val_plot(
		p : plt.Axes,
		metric_name : str,
		train_values: list[float],
		val_values: list[float],
		best_epoch: int
	) -> None:
	"""Disegna le metriche di loss e accuracy per la fase di addestramento e validazione del modello ed evidenzia la migliore epoca del checkpoint in base al valore della metrica di valutazione selezionata.

	Il checkpoint viene evidenziato con:
	- Una linea verticale all'epoca migliore;
	- Una linea orizzontale al valore corrispondente alla metrica all'epoca migliore;
	- Un punto rosso sul valore della metrica all'epoca migliore;


	Args:
		p (plt.Axes): L'oggetto Axes su cui disegnare il grafico.
		metric_name (str): Nome della metrica da visualizzare (es. 'Loss' o 'Accuracy').
		train_values (list[float]): Valori della metrica sul set di addestramento per ogni epoca.
		val_values (list[float]): Valori della metrica sul set di validazione per ogni epoca.
		best_epoch (int): Numero dell'epoca migliore, in formato 1-based.
	"""

	#   ################################################################    #
	#	Inizializzazione e validazione dei dati di input.

	def check_input(values, name="values"):
		values = np.asarray(values, dtype=float)

		if values.ndim != 1 or values.size == 0:
			raise ValueError(f"{name} deve essere una sequenza 1D non vuota.")

		if not np.all(np.isfinite(values)):
			raise ValueError(f"{name} contiene valori non finiti.")

		return values

		# end

	val = check_input(val_values, name="val_values")
	train = check_input(train_values, name="train_values") if train_values is not None else None

	if train is not None and train.size != val.size:
		raise ValueError("train_values e val_values devono essere sequenze 1D della stessa lunghezza.")
	
	if not 1 <= int(best_epoch) <= val.size:
		raise ValueError(f"Il numero della migliore epoca deve essere compreso tra 1 e {val.size}.")

	epochs = np.arange(1, val.size + 1)
	selected_value = float(val[best_epoch - 1])

	all_values = val if train is None else np.concatenate((train, val))
	lower = float(all_values.min())
	upper = float(all_values.max())

	margin = (
		(upper - lower) * 0.05
		if upper > lower
		else max(abs(lower) * 0.05, 1e-6)
	)

	#   ################################################################    #
	#	Disegno delle curve e del checkpoint selezionato.

	if train is not None:
		p.plot(epochs, train, label=f"Train {metric_name}")
	p.plot(epochs, val, label=f"Val {metric_name}")

	p.axvline(
		best_epoch,
		color="red", linestyle="--", linewidth=0.8,
		label=f"Checkpoint: epoch {best_epoch}",
	)

	p.axhline(
		selected_value,
		color="red", linestyle="--", linewidth=0.8,
	)

	p.scatter(
		[best_epoch], [selected_value],
		color="red", zorder=5,
	)

	#   ################################################################    #
	#   Aggiunta dei metadati ai grafici.

	p.set_title(f"{metric_name} over Epochs")
	p.set_xlabel("Epoch")
	p.set_ylabel(metric_name)
	
	p.set_xlim(0.5, val.size + 0.5)
	p.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
	
	p.set_ylim(lower - margin, upper + margin)
	configure_metric_y_ticks(p, lower, upper, selected_value, 10, 0.10)

	p.grid(alpha=0.2)
	p.legend()

	# end

def draw_confusion_matrix(cm, labels): 

	ax = sns.heatmap(
		cm,
		annot=False,
		fmt='',
		cmap='plasma_r',   # Mappa colori plasma invertita
		linewidths=0.5,    # Spessore linee della griglia
		mask= cm == 0,     # Applica la maschera
		linecolor='black', # Colore linee della griglia
		square=True,       # Celle quadrate
		cbar_kws={ "ticks": [0.1, 1, 10, 100] },
		xticklabels=labels,
		yticklabels=labels
	)

	ax.set_facecolor('white')

	empty_cols = np.where(cm.sum(axis=0) == 0)[0]

	# Se ci sono colonne vuote, aggiunge un simbolo '•' al centro di ciascuna cella vuota.
	for col in empty_cols:
		for row in range(cm.shape[0]):
			
			# Per centrare il testo nella cella, si aggiunge 0.5 a "row" e "col".
			ax.text(
				col + 0.5, row + 0.5, '•',
				ha='center', va='center',
				color='black', fontsize=15
			)
		
			# end for row
		# end for col

	# end
