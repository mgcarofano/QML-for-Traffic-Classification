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
			lengths: np.ndarray,
			num_classes: int,
			dtype: torch.dtype,
			transform=None,
		):

		#   ############################################################    #
		#	Verifica degli input

		assert isinstance(X_raw, np.ndarray), "X_raw deve essere un array NumPy"
		assert isinstance(y_raw, np.ndarray), "y_raw deve essere un array NumPy"
		assert isinstance(lengths, np.ndarray), "lengths deve essere un array NumPy"

		assert X_raw.ndim >= 2, "X_raw deve includere campioni e feature."
		assert y_raw.ndim == 1, "y_raw deve avere shape (N,)."
		assert lengths.ndim == 1, "lengths deve avere shape (N,)."

		assert X_raw.shape[0] == y_raw.shape[0], "X_raw e y_raw devono avere lo stesso numero di campioni."
		assert X_raw.shape[0] == lengths.shape[0], "X_raw e lengths devono avere lo stesso numero di campioni."

		assert isinstance(num_classes, int) and num_classes > 0, "num_classes deve essere un intero positivo."

		assert y_raw.min() >= 0 and y_raw.max() < num_classes, "y_raw deve contenere etichette valide tra 0 e il numero totale di classi."

		assert dtype in [torch.float32, torch.float64], "dtype deve essere torch.float32 o torch.float64."

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
		)

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
