"""
	## traffic_dataset.py \n
	*di Mario Gabriele Carofano*

	---

	Modulo per l'implementazione e gestione del dataset di traffico mobile.
"""

import torch
from torch.utils.data import Dataset
import numpy as np

class TrafficDataset(Dataset):
	"""Dataset di traffico mobile con feature ed etichette memorizzate in CPU. Verifica gli array di feature e etichette, li converte in tensori mantenuti sulla CPU e applica, se indicata, una trasformazione a ciascun campione.
	
	## Note \n
	- Il trasferimento sul device viene gestito nel loop di esecuzione. Inoltre, si lascia al `DataLoader` la costruzione dei batch e la gestione del memory pinning, ottimizzando così le prestazioni complessive.

	Args:
		X_raw (np.ndarray): Array delle feature dei campioni.
		y_raw (np.ndarray): Array delle etichette corrispondenti ai campioni.
		lengths (np.ndarray | None): Array delle lunghezze dei campioni, se applicabile.
		num_classes (int): Numero di classi presenti nel dataset.
		dtype (torch.dtype): Tipo di dato dei tensori delle feature.
		transform (callable, optional): Funzione di trasformazione da applicare ai campioni.

	Raises:
		TypeError: Se i parametri di input non sono del tipo corretto.
		ValueError: Se i vettori in input non sono compatibili tra loro.
	"""

	def __init__(
			self,
			X_raw: np.ndarray, y_raw: np.ndarray,
			lengths: np.ndarray | None,
			num_classes: int,
			dtype: torch.dtype,
			transform=None,
		):

		#   ############################################################    #
		#	Verifica degli input.
		#	I dati devono essere già validati e preprocessati dal chiamante.

		if not isinstance(X_raw, np.ndarray):
			raise TypeError("Il tensore di campioni in input deve essere un array NumPy.")
		if not isinstance(y_raw, np.ndarray):
			raise TypeError("Il vettore di etichette deve essere un array NumPy.")

		if X_raw.ndim != 2 or not np.issubdtype(X_raw.dtype, np.floating):
			raise TypeError(
				"Errore nel caricamento dell'array delle feature: \n"
				"Atteso: array 2D di float. "
				f"Ricevuto: shape {X_raw.shape}, dtype {X_raw.dtype}"
			)

		if y_raw.ndim != 1 or not np.issubdtype(y_raw.dtype, np.integer):
			raise TypeError(
				"Errore nel caricamento del vettore delle etichette: \n"
				"Atteso: vettore 1D di interi. "
				f"Ricevuto: shape {y_raw.shape}, dtype {y_raw.dtype}"
			)

		if len(X_raw) != len(y_raw):
			raise ValueError(
				"Il numero di campioni e il numero di etichette devono corrispondere. "
				f"Ricevuto: len(X_raw)={len(X_raw)}, len(y_raw)={len(y_raw)}"
			)

		if dtype not in (torch.float32, torch.float64):
			raise TypeError(
				"Il tipo di dato deve essere torch.float32 o torch.float64. \n"
				f"Ricevuto: {dtype}"
			)

		if transform is not None and not callable(transform):
			raise TypeError(
				"Il parametro 'transform' deve essere una funzione. \n"
				f"Ricevuto: {type(transform)}"
			)

		if lengths is not None:
			if not isinstance(lengths, np.ndarray):
				raise TypeError("Il vettore delle lunghezze deve essere un array NumPy.")
			if lengths.ndim != 1 or not np.issubdtype(lengths.dtype, np.integer):
				raise TypeError(
					"Il vettore delle lunghezze deve essere un array 1D. "
					f"Ricevuto: shape {lengths.shape}, dtype {lengths.dtype}"
				)
			if len(lengths) != len(y_raw):
				raise ValueError(
					"Il vettore delle lunghezze non è compatibile con il vettore delle etichette. "
					f"Ricevuto: len(lengths)={len(lengths)}, len(y_raw)={len(y_raw)}"
				)

		self.num_classes = num_classes
		self.transform = transform

		#   ############################################################    #
		#	Normalizza la disposizione in memoria e converte in tensori CPU.

		self.X = torch.as_tensor(
			np.ascontiguousarray(X_raw),
			dtype=dtype,
			device="cpu",
		)
		
		self.y = torch.as_tensor(
			np.ascontiguousarray(y_raw),
			dtype=torch.long,
			device="cpu",
		)

		self.lengths = torch.as_tensor(
			np.ascontiguousarray(lengths),
			dtype=torch.long,
			device="cpu",
		) if lengths is not None else None

		# end
	
	def __len__(self):
		return len(self.y)

		# end

	def __num_classes__(self):
		return self.num_classes

		# end

	def __getitem__(self, idx):
		x = self.X[idx]

		if self.transform is not None:
			x = self.transform(x)

		if self.lengths is None:
			return x, self.y[idx]

		return x, self.y[idx], self.lengths[idx]

		# end

	# end class
