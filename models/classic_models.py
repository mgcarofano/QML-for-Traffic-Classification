"""

    classic_models.py
    di Mario Gabriele Carofano

    Questo modulo raccoglie i modelli classici non-quantistici usati come
    baseline di confronto rispetto ai modelli ibridi quantistico-classici.

    Le classi mantengono la stessa interfaccia delle classi dei modelli quantistici,
    in modo da poter essere utilizzate in modo intercambiabile nel notebook principale.

    Args:
        n_qubits (int): Numero di qubit per il circuit  o quantistico (non utilizzato).
        n_layers (int): Numero di layer per StronglyEntanglingLayers (non utilizzato).
        n_packets (int): Numero di pacchetti nell'input.
        n_features (int): Numero di feature per pacchetto.
        num_classes (int): Numero di classi per la classificazione.
        n_shots (int, optional): Numero di misurazioni da eseguire (default: None) (non utilizzato).
        random_seed (int): Seed per il dispositivo quantistico (default: 42) (non utilizzato).

"""

#   ####################################################################    #
#   LIBRERIE e IMPORT

import torch
import torch.nn as nn
import torch.nn.functional as F
from model_selection import compute_model_name

#   ####################################################################    #
#   Dense model

# Author : MatteoRichardGaudino
class ClassicalDenseBaseline(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self,
            n_qubits, n_layers,
            n_packets, n_features,
            num_classes, n_shots=None,
            random_seed=42,
        ):

        super(ClassicalDenseBaseline, self).__init__()

        # Salva i parametri come attributi
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_packets = n_packets
        self.n_features = n_features
        self.num_classes = num_classes
        self.n_shots = n_shots
        self.random_seed = random_seed

        # 1. Flatten
        # Input shape originale: (n_packets, n_features)
        # Dimensione appiattita: n_packets * n_features
        self.flatten = nn.Flatten()

        # 2. Dense Layer (Pre-processing per il quantum layer)
        # Output dim deve essere 2^n_qubits per l'AmplitudeEmbedding
        input_dim = n_packets * n_features
        self.dense1 = nn.Linear(input_dim, 2**n_qubits)
        # self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        # 4. Output Dense Layer
        # Input dim è 2^n_qubits (output di qml.probs)
        self.dense2 = nn.Linear(2**n_qubits, num_classes)
        # self.softmax = nn.Softmax(dim=1)

        # end

    def forward(self, x):

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu(x)

        return self.dense2(x)
        # return self.softmax(x)

        # end

    def get_model_name(self):
        return compute_model_name("ClassicalDenseBaseline", num_classes=self.num_classes)

        # end

    def get_model_name_short(self):
        return compute_model_name("ClassicalDenseBaseline", num_classes=self.num_classes)

        # end
    
    # end class

#   ####################################################################    #
#   Classic Conv1D model

# Author : MatteoRichardGaudino
class TrafficCNN(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self,
            n_qubits, n_layers,
            n_packets, n_features,
            num_classes, n_shots=None,
            random_seed=42,
        ):

        super(TrafficCNN, self).__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_packets = n_packets
        self.n_features = n_features
        self.num_classes = num_classes
        self.n_shots = n_shots
        self.random_seed = random_seed

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
#   Dense pre-processing + Conv1D + Dense

class _DenseCnn1DBackbone(nn.Module):
    """Backbone condiviso da AmpCnn, ClassicalTwinModel e ClassicalLight,
    che implementa la parte comune dell'architettura. Il blocco intermedio
    è l'unica parte che varia tra le tre classi, e viene iniettato dal
    chiamante, invece di duplicare la stessa logica in ogni classe.

    Args:
        nn.Module: Base class for all neural network modules in PyTorch.

    """

    def __init__(self, n_qubits, n_packets, n_features, num_classes, middle_block=None):

        super().__init__()

        self.n_qubits = n_qubits

        # Dimensione output quantistico (es. 2^5 = 32)
        self.q_output_dim = 2 ** n_qubits

        # 1. Pre-processing denso.
        self.flatten = nn.Flatten()
        self.dense_pre = nn.Linear(n_packets * n_features, self.q_output_dim)
        self.activation_pre = nn.Sigmoid()

        # 2. Blocco intermedio iniettato
        self.middle_block = middle_block if middle_block is not None else nn.Identity()

        # 3. Strato convoluzionale 1D
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.relu = nn.ReLU()

        # 4. Strato denso finale
        flattened_size = 32 * (self.q_output_dim // 2)
        self.fc1 = nn.Linear(flattened_size, 64)
        self.fc2 = nn.Linear(64, num_classes)

        # end
    
    def forward(self, x):

        x = self.flatten(x)
        x = self.dense_pre(x)
        x = self.activation_pre(x)

        x = self.middle_block(x)

        # Reshape per Conv1d: (Batch, 1, 2^n_qubits)
        x = x.view(-1, 1, self.q_output_dim)

        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))

        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

        # end

    # end class

# Author : MatteoRichardGaudino
class ClassicalTwinModel(nn.Module):
    """Variante classica implementata come wrapper di _DenseCnn1DBackbone:
    il blocco intermedio è un Linear+ReLU che sostituisce
    il lavoro del quantum layer.

    Args:
        nn.Module: Base class for all neural network modules in PyTorch.
    """

    def __init__(self,
            n_qubits, n_layers,
            n_packets, n_features,
            num_classes, n_shots=None,
            random_seed=42
        ):

        super(ClassicalTwinModel, self).__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_packets = n_packets
        self.n_features = n_features
        self.num_classes = num_classes
        self.n_shots = n_shots
        self.random_seed = random_seed

        q_output_dim = 2 ** n_qubits

        self.backbone = _DenseCnn1DBackbone(
            n_qubits, n_packets, n_features, num_classes,
            middle_block = nn.Sequential(
                nn.Linear(q_output_dim, q_output_dim),
                nn.Sigmoid(),
            )
        )

        # end

    def forward(self, x):
        return self.backbone(x)

        # end

    def get_model_name(self):
        return compute_model_name(
            "AmpHybrid_CUSTOM_TWIN",
            self.n_qubits, self.n_layers,
            self.n_packets, self.n_features, self.num_classes
        )

        # end

    def get_model_name_short(self):
        return compute_model_name(
            "AmpHybrid_CUSTOM_TWIN",
            self.n_qubits, self.n_layers
        )

        # end
    
    # end class

# Author : MatteoRichardGaudino
class ClassicalLight(nn.Module):
    """Variante classica implementata come wrapper di _DenseCnn1DBackbone:
    nessun blocco intermedio, il tensore passa direttamente
    al blocco convoluzionale.

    Args:
        nn.Module: Base class for all neural network modules in PyTorch.
    """

    def __init__(self,
            n_qubits, n_layers,
            n_packets, n_features,
            num_classes, n_shots=None,
            random_seed=42,
        ):

        super(ClassicalLight, self).__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_packets = n_packets
        self.n_features = n_features
        self.num_classes = num_classes
        self.n_shots = n_shots
        self.random_seed = random_seed

        self.backbone = _DenseCnn1DBackbone(
            n_qubits, n_packets, n_features, num_classes,
            middle_block = nn.Identity()
        )

        # end

    def forward(self, x):
        return self.backbone(x)

        # end

    def get_model_name(self):
        return compute_model_name(
            "AmpHybrid_CUSTOM_LIGHT",
            self.n_qubits, self.n_layers,
            self.n_packets, self.n_features, self.num_classes
        )

        # end

    def get_model_name_short(self):
        return compute_model_name(
            "AmpHybrid_CUSTOM_LIGHT",
            self.n_qubits, self.n_layers
        )

        # end
    
    # end class

#   ####################################################################    #
#   Conv2D + LSTM

# Author : Vincenzo Spadari
class CNNLSTMModel(nn.Module):
    """CNN+LSTM model.

    Args:
        nn (_type_): _description_
    """

    def __init__(self,
            n_qubits, n_layers,
            n_packets, n_features,
            num_classes, n_shots=None,
            random_seed=42,
        ):

        super(CNNLSTMModel, self).__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_packets = n_packets
        self.n_features = n_features
        self.num_classes = num_classes
        self.n_shots = n_shots
        self.random_seed = random_seed

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
            self.n_packets, self.n_features, self.num_classes
        )

        # end

    def get_model_name_short(self):
        return compute_model_name("CNNLSTM")

        # end
    
    # end class
