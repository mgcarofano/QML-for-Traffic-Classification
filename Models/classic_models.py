"""
    classic_models.py
    di Mario Gabriele Carofano

    Questo modulo raccoglie i modelli classici non-quantistici usati come
    baseline di confronto rispetto ai modelli ibridi quantistico-classici.

    Args:
        n_packets (int): Numero di pacchetti nell'input.
        n_features (int): Numero di feature per pacchetto.
        num_classes (int): Numero di classi per la classificazione.
"""

#   ####################################################################    #
#   Librerie

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from Source.model_selection import \
    compute_model_name, \
    get_output_dim

#   ####################################################################    #
#   Classic Conv1D model

# Author : MatteoRichardGaudino
class TrafficCNN(nn.Module):
    """Implementazione di un modello classico CNN1D per la classificazione del traffico."""

    def __init__(self, n_features, n_packets, num_classes):

        super(TrafficCNN, self).__init__()

        self.n_packets = n_packets
        self.n_features = n_features
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Global Average Pooling -> (B, 256, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x

        # end

    def get_model_name(self):
        return compute_model_name("TrafficCNN", num_classes=self.num_classes)

        # end

    def get_model_name_short(self):
        return compute_model_name("TrafficCNN", num_classes=self.num_classes)

        # end
    
    # end class

#   ####################################################################    #
#   Conv2D + LSTM

# Author : Vincenzo Spadari
# class CNNLSTMModel(nn.Module):
#     """Implementazione di un modello CNN+LSTM per la classificazione del traffico."""

#     def __init__(self, n_features, n_packets, num_classes):

#         super(CNNLSTMModel, self).__init__()

#         self.n_packets = n_packets
#         self.n_features = n_features
#         self.num_classes = num_classes

#         in_channels = 1
#         out_features_size = 100
#         filters = [32, 64]
#         kernel = (4, 2)
#         stride = (1, 1)

#         hidden_size = max([100, 50])

#         self.conv1 = nn.Conv2d(in_channels, filters[0], kernel, stride=stride, padding=0)
#         self.bn1 = nn.BatchNorm2d(filters[0])
#         self.conv2 = nn.Conv2d(filters[0], filters[1], kernel, stride=stride, padding=0)
#         self.bn2 = nn.BatchNorm2d(filters[1])

#         # # Calcolo dinamico della dimensione di input per l'LSTM.
#         # # Si esegue un forward "a vuoto" sui soli layer convoluzionali,
#         # # con le dimensioni reali di input, per ricavare la shape effettiva.
#         # with torch.no_grad():
#         #     dummy = torch.zeros(1, in_channels, n_packets, n_features)
#         #     dummy_out = self.conv2(self.conv1(dummy))
#         #     _, c, h, w = dummy_out.shape
#         #     # lstm_input_size = c * w
#         #     lstm_input_size = c * h

#         # self.lstm = nn.LSTM(
#         #     input_size=lstm_input_size,
#         #     hidden_size=hidden_size,
#         #     batch_first=True
#         # )

#         features_size0, self.paddings0 = get_output_dim(
#             n_packets,
#             kernels=[kernel[0], kernel[0]],
#             strides=[stride[0], stride[0]],
#             padding='valid',
#             return_paddings=True
#         )
#         features_size1, self.paddings1 = get_output_dim(
#             n_features,
#             kernels=[kernel[1], kernel[1]],
#             strides=[stride[1], stride[1]],
#             padding='valid',
#             return_paddings=True
#         )

#         lstm_input_size = features_size1 * filters[1]
#         self.lstm = nn.LSTM(lstm_input_size, hidden_size, batch_first=True)

#         self.fc1 = nn.Linear(hidden_size, out_features_size)
#         self.fc = nn.Linear(out_features_size, num_classes)

#         self.relu = nn.ReLU()
#         self.dropout1 = nn.Dropout(0.2)
#         self.dropout2 = nn.Dropout(0.4)

#         # end

#     def forward(self, x):
#         x = x.unsqueeze(1)

#         out = self.relu(self.extract_features(x))

#         out = self.dropout2(out)
#         out = self.fc(out)

#         return out

#         # end

#     def extract_features(self, x):
#         out = self.relu(self.conv1(x))
#         out = self.bn1(out)

#         out = self.relu(self.conv2(out))
#         out = self.bn2(out)

#         # size_interm = out.size()
#         # out = out.transpose(1, 2)
#         # out = out.reshape(
#         #     size_interm[0], size_interm[3],
#         #     size_interm[1] * size_interm[2]
#         # )

#         out = out.permute(0, 2, 1, 3)
#         out = out.flatten(start_dim=2)
#         if out.size(-1) != self.lstm.input_size:
#             raise ValueError(
#                 f"Input LSTM incompatibile: shape={tuple(out.shape)}, "
#                 f"input_size atteso={self.lstm.input_size}"
#             )

#         out, _ = self.lstm(out)
#         out = out[:, -1, :]

#         out = self.dropout1(out)
#         out = self.fc1(out)

#         return out

#         # end

#     def get_model_name(self):
#         return compute_model_name(
#             "CNNLSTM",
#             n_packets=self.n_packets,
#             n_features=self.n_features,
#             num_classes=self.num_classes
#         )

#         # end

#     def get_model_name_short(self):
#         return compute_model_name("CNNLSTM")

#         # end
    
#     # end class

class CNNLSTMModel(nn.Module):

    def __init__(
            self,
            n_features, n_packets, num_classes,
            n_layers
        ):

        super().__init__()

        if min(n_features, n_packets, num_classes, n_layers) < 1:
            raise ValueError("I parametri devono essere interi positivi.")

        self.n_features = n_features
        self.n_packets = n_packets
        self.num_classes = num_classes
        self.n_layers = n_layers

        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=(3, 1),
                padding=(1, 0),
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(3, n_features),
                padding=(1, 0),
            ),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.2 if n_layers > 1 else 0.0,
        )

        self.classifier = nn.Linear(64, num_classes)
    
        # end

    def forward(self, x):
        # x: (batch, n_packets, n_features), floating point.
        if x.ndim != 3 or x.shape[1:] != (self.n_packets, self.n_features):
            raise ValueError(
                f"Atteso (batch, {self.n_packets}, {self.n_features}), "
                f"ricevuto {tuple(x.shape)}."
            )

        # Calcolo prima della CNN: i bias possono rendere non nullo il padding.
        lengths = x.ne(0).any(dim=-1).sum(dim=1)
        if (lengths == 0).any():
            raise ValueError("Ogni flusso deve contenere almeno un pacchetto reale.")

        x = self.cnn(x.unsqueeze(1))       # (batch, 32, n_packets, 1)
        x = x.squeeze(-1).transpose(1, 2)  # (batch, n_packets, 32)

        x = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (h_n, _) = self.lstm(x)
        return self.classifier(h_n[-1])    # (batch, num_classes), logits

        # end

    def get_model_name(self):
        return compute_model_name(
            "CNNLSTM",
            n_packets=self.n_packets,
            n_features=self.n_features,
            num_classes=self.num_classes
        )

        # end

    def get_model_name_short(self):
        return compute_model_name("CNNLSTM")

        # end

    # end class
