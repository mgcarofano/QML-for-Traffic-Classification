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
from Source.model_selection import compute_model_name

#   ####################################################################    #
#   Implementazione dei modelli classici
#   ...

#   ---------------------------------------------------------------------   #
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

#   ---------------------------------------------------------------------   #
#   Conv2D + LSTM

# Author : Vincenzo Spadari
class CNNLSTMModel(nn.Module):
    """Implementazione di un modello CNN+LSTM per la classificazione del traffico."""

    def __init__(self, n_features, n_packets, num_classes):

        super(CNNLSTMModel, self).__init__()

        self.n_packets = n_packets
        self.n_features = n_features
        self.num_classes = num_classes

        in_channels = 1
        out_features_size = 100
        filters = [32, 64]
        kernel = (4, 2)
        stride = (1, 1)

        hidden_size = max([100, 50])

        self.conv1 = nn.Conv2d(in_channels, filters[0], kernel, stride=stride, padding=0)
        self.bn1 = nn.BatchNorm2d(filters[0])
        self.conv2 = nn.Conv2d(filters[0], filters[1], kernel, stride=stride, padding=0)
        self.bn2 = nn.BatchNorm2d(filters[1])

        # Calcolo dinamico della dimensione di input per l'LSTM.
        # Si esegue un forward "a vuoto" sui soli layer convoluzionali,
        # con le dimensioni reali di input, per ricavare la shape effettiva.
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, n_packets, n_features)
            dummy_out = self.bn2(self.conv2(self.bn1(self.conv1(dummy))))
            _, c, h, w = dummy_out.shape
            lstm_input_size = c * h

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc1 = nn.Linear(hidden_size, out_features_size)
        self.fc = nn.Linear(out_features_size, num_classes)

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.4)

        # end

    def forward(self, x):
        x = x.unsqueeze(1)

        out = F.relu(self.extract_features(x))
        out = self.dropout2(out)
        out = self.fc(out)

        return out

        # end

    def extract_features(self, x):
        out = F.relu(self.conv1(x))
        out = self.bn1(out)

        out = F.relu(self.conv2(out))
        out = self.bn2(out)

        size_interm = out.size()
        out = out.transpose(1, 2)
        out = out.reshape(
            size_interm[0], size_interm[3],
            size_interm[1] * size_interm[2]
        )

        out, _ = self.lstm(out)
        out = out[:, -1, :]

        out = self.dropout1(out)
        out = self.fc1(out)

        return out

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
