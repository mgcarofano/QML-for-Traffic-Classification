"""
	classes.py \n
	by Mario Gabriele Carofano

	Il modulo fornisce la classe `TrafficDataset`, che verifica gli array di feature e etichette, li converte in tensori mantenuti sulla CPU e applica, se indicata, una trasformazione a ciascun campione. La creazione dei batch e il trasferimento dei dati sul dispositivo sono demandati rispettivamente al `DataLoader` e al loop di esecuzione.
"""

import torch
from torch.utils.data import Dataset
import numpy as np

class TrafficDataset(Dataset):
	"""Dataset di traffico con feature ed etichette memorizzate in CPU.
	
	**Nota:** \n
	Il trasferimento sul device viene gestito nel loop di esecuzione. Inoltre, si lascia al DataLoader la costruzione dei batch e la gestione del memory pinning, ottimizzando così le prestazioni complessive.
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

		if y_raw.ndim != 1 or not np.issubdtype(y_raw.dtype, np.integer):
			raise TypeError(
				"Errore nel caricamento del vettore delle etichette: \n"
				"Atteso: vettore 1D di interi. "
				f"Ricevuto: shape {y_raw.shape}, dtype {y_raw.dtype}"
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
					"Il vettore delle lunghezze deve avere la stessa lunghezza del vettore delle etichette. "
					f"Ricevuto: len(lengths)={len(lengths)}, len(y_raw)={len(y_raw)}"
				)

		#   ############################################################    #

		# Normalizza la disposizione in memoria e converte in tensori CPU.
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

		self.num_classes = num_classes
		self.transform = transform

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
