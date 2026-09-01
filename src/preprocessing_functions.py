"""
    preprocessing_functions.py
    di Mario Gabriele Carofano
    
    Modulo per il preprocessing dei flussi di traffico di rete.
    Fornisce funzioni di normalizzazione e trasformazione logaritmica
    per preparare i dati al machine learning.
"""

#   ####################################################################    #
#   LIBRARIES

import numpy as np

# La classe MinMaxScaler serve per scalare le caratteristiche in un intervallo specifico, tipicamente tra 0 e 1.
# Questo è utile per normalizzare i dati prima di applicare algoritmi di machine learning,
# poiché molti algoritmi funzionano meglio quando le caratteristiche hanno scale simili.
from sklearn.preprocessing import MinMaxScaler

#   ####################################################################    #
#	UTILITY FUNCTIONS

def combine_dir_pl(X : np.ndarray) -> np.ndarray:
	""" Combina la colonna DIR (direzione del pacchetto) con la
	colonna PL (lunghezza del payload), creando una nuova colonna che
	rappresenta la lunghezza del payload con il segno corretto in base
	alla direzione del pacchetto.
	
	- Se DIR è 1 (pacchetto in entrata), il valore di PL viene reso negativo;
	- Se DIR è 0 (pacchetto in uscita), il valore di PL rimane positivo.
	
	Questa funzione è utile per rappresentare in modo più significativo
	la direzione del traffico di rete nei dati preprocessati.

	Args:
		X (numpy.ndarray): Tensor con shape (N, M, 4),
		dove N è il numero di flussi, M è il numero massimo di pacchetti per flusso
		(può essere 36 o 100 a seconda del dataset) e 4 è il numero di features per pacchetto.
		Le features sono: [DIR (0), PL (1), TCPWIN (2), IAT (3)].

	Returns:
		numpy.ndarray: Tensor con shape (N, M, 3), dove la prima colonna è PL con segno corretto,
		la seconda colonna è TCPWIN e la terza colonna è IAT.
	"""

	# Estrae le colonne DIR e PL dall'input.
	control_column = X[:, :, 0] # DIR
	data_values = X[:, :, 1] # PL

	# Calcola il moltiplicatore per il segno:
	# -1 per DIR=1, 1 per DIR=0.
	sign_multiplier = np.where(control_column == 1, -1, 1)

	# Moltiplica i valori PL per il segno.
	signed_data_values = data_values * sign_multiplier

	# Estrae le colonne TCPWIN e IAT.
	other_features = X[:, :, 2:4]

	# Procede alla concatenazione, adattando correttamente le dimensioni dei tensor.
	signed_data_values_reshaped = signed_data_values[:, :, np.newaxis]
	processed_array = np.concatenate((signed_data_values_reshaped, other_features), axis=2)
	return processed_array

	# end

#   ####################################################################    #
#   PREPROCESSING FUNCTIONS

def log1pPreprocessing(
		X : np.ndarray,
		num_packets : int,
		combine_dir_pl_flag : bool = False
	) -> np.ndarray:

	""" Applica il preprocessing ai flussi di traffico,
	con masking del padding e trasformazione logaritmica.

	Trasforma le feature numeriche (Payload Length, TCP Window, IAT)
	applicando il logaritmo naturale (log1p), utile per comprimere
	l'intervallo dinamico dei valori e normalizzare distribuzioni asimmetriche.
	I pacchetti di padding vengono azzerati.
	
	Args:
		X (numpy.ndarray): tensor con shape (N, M, 4),
		dove N è il numero di flussi, M è il numero massimo di pacchetti per flusso
		(può essere 36 o 100 a seconda del dataset) e 4 è il numero di features per pacchetto.
		Le features sono: [DIR (0), PL (1), TCPWIN (2), IAT (3)].
		
		num_packets (int): numero di pacchetti da considerare per flusso.
		combine_dir_pl_flag (bool): se True, combina le feature DIR e PL
		in una singola feature PL con segno, riducendo il numero di feature a 3.

	Returns:
		numpy.ndarray: Tensor preprocessato con shape (N, num_packets, num_features), in
		formato float32. I pacchetti di padding vengono azzerati e le feature
		PL, TCPWIN e IAT sono trasformate con "np.log1p".
	"""

	# Copia l'input per evitare modifiche in-place.
	X_proc = X.copy().astype(np.float32)

	# Identifica i pacchetti di padding (dove la feature PL è -1).
	is_padding = (X_proc[:, :, 1] == -1)

	# Azzera tutte le feature dei pacchetti di padding.
	mask = np.repeat(is_padding[:, :, np.newaxis], 4, axis=2)
	X_proc[mask] = 0

	# Applica la trasformazione logaritmica (Log1p)
	# alle feature PL (1), TCPWIN (2) e IAT (3).
	X_proc[:, :, 1:] = np.log1p(np.maximum(X_proc[:, :, 1:], 0))

	# Combina DIR e PL in PL con segno se il flag è impostato
	if combine_dir_pl_flag:
		X_proc = combine_dir_pl(X_proc)

	# Tronca la sequenza ai primi num_packets e ritorna il risultato.
	return X_proc[:, :num_packets, :]

	# end

def minMaxPreprocessing(
		X : np.ndarray,
		num_packets : int,
		combine_dir_pl_flag : bool = False
    ) -> np.ndarray:

	"""Applica una normalizzazione min-max alle feature dei pacchetti.

	Normalizza tutte le feature nell'intervallo [0, 1],
	garantendo che il modello tratti tutte le feature su scale comparabili.

	Args:
		X (numpy.ndarray): tensor con shape (N, X, 4),
		dove N è il numero di flussi, X è il numero massimo di pacchetti per flusso
		(può essere 36 o 100 a seconda del dataset) e 4 è il numero di features per pacchetto.
		Le features sono: [DIR (0), PL (1), TCPWIN (2), IAT (3)].
		
		num_packets (int): numero di pacchetti da considerare per flusso.
		combine_dir_pl_flag (bool): se True, combina le feature DIR e PL
		in una singola feature PL con segno, riducendo il numero di feature a 3.

	Returns:
		numpy.ndarray: tensor preprocessato con shape (N, num_packets, num_features),
		in formato float32, dove tutte le feature dei pacchetti sono scalate
		nell'intervallo [0, 1] usando la normalizzazione min-max.
	"""

	num_features = 4

	# Copia l'input per evitare modifiche in-place.
	X_proc = X.copy().astype(np.float32)

	# Combina DIR e PL in PL con segno se il flag è impostato
	if combine_dir_pl_flag:
		X_proc = combine_dir_pl(X_proc)
		num_features = 3

	# Inizializza lo scaler MinMaxScaler.
	scaler = MinMaxScaler(feature_range=(0, 1))

	# Adatta i dati in una forma che MinMaxScaler può trattare correttamente.
	# MinMaxScaler.fit() non lavora direttamente su tensori 3D, ma vuole dati
	# organizzati come campioni per righe e feature per colonne.
	# Il valore -1 serve a inferire il numero di righe necessario in base al
	# numero totale di elementi e al numero di feature fissato.
	scaler.fit(np.reshape(X_proc, [-1, num_features]))

	# Applica la trasformazione min-max a tutte le feature dei pacchetti.
	X_proc = scaler.transform(np.reshape(X_proc, [-1, num_features]))

	# Ripristina la forma originale del tensor.
	X_proc = np.reshape(X_proc, [-1, X.shape[1], num_features])

	# Tronca la sequenza ai primi num_packets e ritorna il risultato.
	return X_proc[:, :num_packets, :]

	# end
