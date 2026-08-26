"""

    quantum_models.py
    di Mario Gabriele Carofano

    Questo modulo raccoglie i modelli ibridi quantistico-classici basati su
    PennyLane, che combinano diverse strategie di embedding, con ansatz variazionali
    e strati classici di pre/post-processing.

    Le classi mantengono la stessa interfaccia delle classi dei modelli quantistici,
    in modo da poter essere utilizzate in modo intercambiabile nel notebook principale.

    Args:
        n_qubits (int): Numero di qubit per il circuito quantistico.
        n_layers (int): Numero di layer per StronglyEntanglingLayers.
        n_packets (int): Numero di pacchetti nell'input.
        n_features (int): Numero di feature per pacchetto.
        num_classes (int): Numero di classi per la classificazione.
        n_shots (int, optional): Numero di misurazioni da eseguire (default: None).
        random_seed (int): Seed per il dispositivo quantistico (default: 42).

"""

#   ####################################################################    #
#   LIBRERIE e IMPORT

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
from constants import SIMULATOR
from model_selection import compute_model_name
from classic_models import _DenseCnn1DBackbone

#   ####################################################################    #
#   FUNZIONI di utilità

# Author : Vincenzo Spadari
def meas_qoutputsize_mapping(n_qubits : int) -> dict:
    """Restituisce la dimensione dell'output di misurazione in base al numero di qubit.

    Args:
        n_qubits (int): Numero di qubit per il circuito quantistico.

    Returns:
        dict: Mappa tra tipo di misurazione e dimensione dell'output.
    """

    return {
        "pauli": n_qubits,
        "probs": 2**n_qubits
    }

    # end

#   ####################################################################    #
#   Amplitude embedding quantum model

# Author : Vincenzo Spadari
class AmpeDenseModel(nn.Module):
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

        #   ############################################################    #
        #   Inizializzazione degli attributi del modello

        super(AmpeDenseModel, self).__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers

        self.n_packets = n_packets
        self.n_features = n_features
        self.num_classes = num_classes

        self.random_seed = random_seed
        self.n_shots = n_shots

        # Si inizializza il dispositivo quantistico
        # con il numero di qubit e il seed specificato.
        self.dev = qml.device(
            SIMULATOR,
            wires=n_qubits,

            # Siccome le misurazioni in un circuito quantistico
            # sono non-deterministiche, si può specificare il numero
            # di "shots" (misurazioni) da eseguire per ottenere
            # una stima più accurata delle probabilità di output.
            shots=n_shots,

            seed=random_seed
        )

        #   ############################################################    #
        #   Calcolo delle variabili

        q_output_size = meas_qoutputsize_mapping(n_qubits)["probs"]
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        input_dim = n_packets * n_features

        #   ############################################################    #
        #   Definizione del circuito quantistico

        @qml.qnode(self.dev, interface="torch")
        def qnode(inputs, weights):
            # Feature map
            qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True, pad_with=0.0)

            # Ansatz
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

            # Processo di misurazione
            return qml.probs(wires=range(n_qubits))

            # end

        self.qnode = qnode
        # if self.dev_name.startswith("fake"):
        #     self.qnode = qml.set_shots(self.qnode, self.n_shots)

        #   ############################################################    #
        #   Architettura del modello

        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(input_dim, 2**n_qubits)
        self.sigmoid = nn.Sigmoid()
        self.q_layer = qml.qnn.TorchLayer(self.qnode, weight_shapes)
        self.dense2 = nn.Linear(q_output_size, num_classes)

        # end

    def forward(self, x):

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.sigmoid(x)
        x = self.q_layer(x)

        return self.dense2(x)

        # end

    def quantum_forward(self, x):
        """
        Method to execute only the forward pass through the quantum layer.
        Useful for extracting quantum statevector (simulation).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Quantum layer output (statevector)
        """

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.sigmoid(x)
        x = self.q_layer(x)

        return x

        # end

    def get_model_name(self):
        return compute_model_name(
            "AmpeDense",
            self.n_qubits, self.n_layers,
            self.n_packets, self.n_features, self.num_classes
        )

        # end

    def get_model_name_short(self):
        return compute_model_name(
            "AmpeDense",
            self.n_qubits, self.n_layers
        )

        # end
    
    # end class

#   ####################################################################    #
#   Angle embedding quantum model 

# Author : Vincenzo Spadari
class AngeDenseModel(nn.Module):
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

        super(AngeDenseModel, self).__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers

        self.n_packets = n_packets
        self.n_features = n_features
        self.num_classes = num_classes

        self.random_seed = random_seed
        self.n_shots = n_shots

        self.dev = qml.device(
            SIMULATOR,
            wires=n_qubits,
            shots=n_shots,
            seed=random_seed
        )

        q_output_size = meas_qoutputsize_mapping(n_qubits)["probs"]
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        input_dim = n_packets * n_features

        # @qml.qnode(self.dev, interface="torch", diff_method="finite-diff")
        @qml.qnode(self.dev, interface="torch")
        def qnode(inputs, weights):
            # Feature map
            qml.AngleEmbedding(inputs, wires=range(n_qubits))

            # Ansatz
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

            # Processo di misurazione
            # return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
            return qml.probs(wires=range(n_qubits))

            # end

        self.qnode = qnode
        # if self.dev_name.startswith("fake"):
        #     self.qnode = qml.set_shots(self.qnode, self.n_shots)

        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(input_dim, n_qubits)
        self.relu = nn.ReLU()
        self.q_layer = qml.qnn.TorchLayer(self.qnode, weight_shapes)
        self.dense2 = nn.Linear(q_output_size, num_classes)

        # end

    def forward(self, x):

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.q_layer(x)

        return self.dense2(x)

        # end

    def quantum_forward(self, x):

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.q_layer(x)

        return x

        # end

    def get_model_name(self):
        return compute_model_name(
            "AngeDense",
            self.n_qubits, self.n_layers,
            self.n_packets, self.n_features, self.num_classes
        )

        # end

    def get_model_name_short(self):
        return compute_model_name(
            "AngeDense",
            self.n_qubits, self.n_layers
        )

        # end
    
    # end class

#   ####################################################################    #
#   Ring quantum model

# Author : MatteoRichardGaudino
class RingHybridModel(nn.Module):
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

        super(RingHybridModel, self).__init__()

        # Salva i parametri come attributi
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_packets = n_packets
        self.n_features = n_features
        self.num_classes = num_classes
        self.n_shots = n_shots
        self.random_seed = random_seed

        # Definizione del dispositivo quantistico
        self.dev = qml.device(
            SIMULATOR,
            wires=n_qubits,
            shots=n_shots,
            seed=random_seed
        )

        # Definizione del QNode
        @qml.qnode(self.dev, interface="torch")
        def qnode(inputs, weights):
            # Feature map - split inputs in two parts
            inputs1 = inputs[..., :n_qubits]
            inputs2 = inputs[..., n_qubits:]

            # First AngleEmbedding with forward ring CNOT
            qml.AngleEmbedding(inputs1, rotation='Y', wires=range(n_qubits))
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[n_qubits - 1, 0])

            # Second AngleEmbedding with backward ring CNOT
            qml.AngleEmbedding(inputs2, rotation='Y', wires=range(n_qubits))
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i + 1, i])
            qml.CNOT(wires=[0, n_qubits - 1])

            # Ansatz
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

            # Processo di misurazione
            return qml.probs(wires=range(n_qubits))

        self.qnode = qnode

        # Definizione delle forme dei pesi per il layer quantistico
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}

        # 1. Flatten
        self.flatten = nn.Flatten()

        # 2. Dense Layer (Pre-processing per il quantum layer)
        # Output dim deve essere 2*n_qubits per il Ring embedding
        input_dim = n_packets * n_features
        self.dense1 = nn.Linear(input_dim, 2 * n_qubits)
        self.relu = nn.ReLU()

        # 3. Quantum Layer
        self.q_layer = qml.qnn.TorchLayer(self.qnode, weight_shapes)

        # 4. Output Dense Layer
        # Input dim è 2^n_qubits (output di qml.probs)
        self.dense2 = nn.Linear(2**n_qubits, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.q_layer(x)
        return self.dense2(x)

    def get_model_name(self):
        return compute_model_name(
            "RingHybrid",
            self.n_qubits, self.n_layers,
            self.n_packets, self.n_features, self.num_classes
        )

        # end

    def get_model_name_short(self):
        return compute_model_name(
            "RingHybrid",
            self.n_qubits, self.n_layers
        )

        # end
    
    # end class

#   ####################################################################    #
#   Waterfall quantum model

# Author : MatteoRichardGaudino
class WaterfallHybridModel(nn.Module):
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

        super(WaterfallHybridModel, self).__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_packets = n_packets
        self.n_features = n_features
        self.num_classes = num_classes
        self.n_shots = n_shots
        self.random_seed = random_seed

        self.dev = qml.device(
            SIMULATOR,
            wires=n_qubits,
            shots=n_shots,
            seed=random_seed
        )

        input_dim = n_packets * n_features
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}

        @qml.qnode(self.dev, interface="torch")
        def qnode(inputs, weights):
            # Feature map - split inputs in two parts.
            inputs1 = inputs[..., :n_qubits]
            inputs2 = inputs[..., n_qubits:]

            # First AngleEmbedding with waterfall CNOT
            qml.AngleEmbedding(inputs1, rotation='Y', wires=range(n_qubits))
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    qml.CNOT(wires=[i, j])

            # Second AngleEmbedding
            qml.AngleEmbedding(inputs2, rotation='Z', wires=range(n_qubits))

            # Ansatz
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

            # Processo di misurazione
            return qml.probs(wires=range(n_qubits))

            # end

        self.qnode = qnode

        # 1. Flatten
        self.flatten = nn.Flatten()

        # 2. Dense Layer (Pre-processing per il quantum layer)
        # Output dim deve essere 2*n_qubits per il Waterfall embedding.
        self.dense1 = nn.Linear(input_dim, 2 * n_qubits)
        self.relu = nn.ReLU()

        # 3. Quantum Layer
        self.q_layer = qml.qnn.TorchLayer(self.qnode, weight_shapes)

        # 4. Output Dense Layer
        # Input dim è 2^n_qubits (output di qml.probs)
        self.dense2 = nn.Linear(2**n_qubits, num_classes)

        # end

    def forward(self, x):

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.q_layer(x)
        x = self.dense2(x)

        return x

        # end

    def quantum_forward(self, x):

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.q_layer(x)

        return x

        # end

    def get_model_name(self):
        return compute_model_name(
            "WaterfallHybrid",
            self.n_qubits, self.n_layers,
            self.n_packets, self.n_features, self.num_classes
        )

        # end

    def get_model_name_short(self):
        return compute_model_name(
            "WaterfallHybrid",
            self.n_qubits, self.n_layers
        )

        # end
    
    # end class

#   ####################################################################    #
#   AmplitudeEmbedding + CNN1D + Dense

# Author : MatteoRichardGaudino
class AmpCnn(nn.Module):
    """Variante quantistica implementata come wrapper di _DenseCnn1DBackbone:
    il blocco intermedio è un quantum layer con AmplitudeEmbedding
    + StronglyEntanglingLayers.

    Args:
        nn.Module: Base class for all neural network modules in PyTorch.
    """

    def __init__(self,
            n_qubits, n_layers,
            n_packets, n_features,
            num_classes, n_shots=None,
            random_seed=42,
        ):

        #   ############################################################    #
        #   Inizializzazione degli attributi del modello

        super(AmpCnn, self).__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_packets = n_packets
        self.n_features = n_features
        self.num_classes = num_classes
        self.n_shots = n_shots
        self.random_seed = random_seed

        self.dev = qml.device(
            SIMULATOR,
            wires=n_qubits,
            shots=n_shots,
            seed=random_seed
        )

        #   ############################################################    #
        #   Definizione del circuito quantistico

        @qml.qnode(self.dev, interface="torch")
        def qnode(inputs, weights):
            qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True, pad_with=0.0)
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return qml.probs(wires=range(n_qubits))

            # end
        
        self.q_layer = qml.qnn.TorchLayer(
            qnode,
            {"weights": (n_layers, n_qubits, 3)}
        )

        #   ############################################################    #
        #   Architettura del modello

        self.backbone = _DenseCnn1DBackbone(
            n_qubits, n_packets, n_features, num_classes,
            middle_block=self.q_layer
        )

        # end

    def forward(self, x):
        return self.backbone(x)

        # end

    def get_model_name(self):
        return compute_model_name(
            "AmpHybrid_CUSTOM",
            self.n_qubits, self.n_layers,
            self.n_packets, self.n_features, self.num_classes
        )

        # end

    def get_model_name_short(self):
        return compute_model_name(
            "AmpHybrid_CUSTOM",
            self.n_qubits, self.n_layers
        )

        # end
    
    # end class

#   ####################################################################    #
#   CNN1D + Quantum + CNN1D + Dense

# Author : MatteoRichardGaudino
class CnnAmpCnn(nn.Module):
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

        super(CnnAmpCnn, self).__init__()

        # Parametri
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_packets = n_packets
        self.n_features = n_features
        self.num_classes = num_classes
        self.n_shots = n_shots
        self.random_seed = random_seed

        self.q_input_dim = 2**n_qubits

        # --- 1. PRIMA CNN (Input -> CNN1D) ---
        # Input shape: (Batch, n_features, n_packets)
        self.cnn1_pre = nn.Sequential(
            nn.Conv1d(in_channels=n_features, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten()
        )

        # Calcolo dimensione dopo cnn1 per collegarla al Quantum Layer
        # Assumendo MaxPool riduca a metà la lunghezza (n_packets // 2)
        cnn1_out_features = 16 * (n_packets // 2)

        # Layer di adattamento per AmplitudeEmbedding (deve essere di dimensione 2^n_qubits)
        self.pre_quantum_dense = nn.Linear(cnn1_out_features, self.q_input_dim)
        self.pre_quantum_act = nn.Sigmoid()

        # --- 2. QUANTUM LAYER ---
        self.dev = qml.device(
            SIMULATOR,
            wires=n_qubits,
            shots=n_shots,
            seed=random_seed
        )

        weight_shapes = {"weights": (n_layers, n_qubits, 3)}

        @qml.qnode(self.dev, interface="torch")
        def qnode(inputs, weights):
            # L'input qui deve avere dimensione 2^n_qubits
            qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True, pad_with=0.0)
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return qml.probs(wires=range(n_qubits))

            # end

        self.quantum_layer = qml.qnn.TorchLayer(qnode, weight_shapes)

        # --- 3. SECONDA CNN (Quantum -> CNN1D) ---
        # L'output quantistico è (Batch, 2^n_qubits), lo trasformiamo in (Batch, 1, 2^n_qubits)
        self.cnn2_post = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten()
        )

        # --- 4. DENSE LAYERS (CNN1D -> Dense -> Dense) ---
        # Dimensione dopo la seconda CNN e il pooling: 32 canali * (2^n_qubits / 2)
        cnn2_out_features = 32 * (self.q_input_dim // 2)

        self.fc1 = nn.Linear(cnn2_out_features, 64)
        self.relu_fc = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

        # end

    def forward(self, x):
        # 1. CNN1D iniziale
        # x shape: (Batch, n_packets, n_features) -> serve (Batch, n_features, n_packets)
        # x = x.transpose(1, 2)
        x = self.cnn1_pre(x)

        # Adattamento per il Quantum Layer
        x = self.pre_quantum_dense(x)
        x = self.pre_quantum_act(x)

        # 2. Quantum Layer
        x = self.quantum_layer(x) # Output shape: (Batch, 2^n_qubits)

        # 3. CNN1D secondaria
        # Reshape per Conv1d: (Batch, 1, Seq_Length)
        x = x.view(-1, 1, self.q_input_dim)
        x = self.cnn2_post(x)

        # 4. Dense Layers finali
        x = self.relu_fc(self.fc1(x))
        x = self.fc2(x)

        return x

        # end

    def get_model_name(self):
        return compute_model_name(
            "CnnAmpCnn",
            self.n_qubits, self.n_layers,
            self.n_packets, self.n_features, self.num_classes
        )

        # end

    def get_model_name_short(self):
        return compute_model_name(
            "CnnAmpCnn",
            self.n_qubits, self.n_layers
        )

        # end
    
    # end class

#   ####################################################################    #
#   AmplitudeEmbedding + CNN1D + LSTM + Dense

# Author : Vincenzo Spadari
class AmpeCNNLSTMModel(nn.Module):
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

        super(AmpeCNNLSTMModel, self).__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_packets = n_packets
        self.n_features = n_features
        self.num_classes = num_classes
        self.random_seed = random_seed
        self.n_shots = n_shots

        in_channels = 1
        out_features_size = 2**n_qubits
        filters = [32, 64]
        kernel = (4, 2)
        stride = (1, 1)

        self.dev = qml.device(
            SIMULATOR,
            wires=n_qubits,
            shots=n_shots,
            seed=random_seed
        )

        hidden_size = max([100, 50])
        q_output_size = meas_qoutputsize_mapping(n_qubits)["probs"]
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}

        @qml.qnode(self.dev, interface="torch")
        def qnode(inputs, weights):
            # Feature map
            qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True, pad_with=0.0)

            # Ansatz
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

            # Processo di misurazione
            return qml.probs(wires=range(n_qubits))

            # end

        self.qnode = qnode
        # if self.dev_name.startswith("fake"):
        #     self.qnode = qml.set_shots(self.qnode, self.n_shots)

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

        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_features_size)

        self.q_layer = qml.qnn.TorchLayer(self.qnode, weight_shapes)
        self.fc = nn.Linear(q_output_size, num_classes)

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.4)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # end

    def forward(self, x):

        x = x.unsqueeze(1)

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
        out = self.relu(out)

        out = self.dropout2(out)
        out = self.fc2(out)
        out = self.sigmoid(out)

        out = self.q_layer(out)

        out = self.fc(out)

        return out

        # end

    def quantum_forward(self, x):

        x = x.unsqueeze(1)

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
        out = self.relu(out)

        out = self.dropout2(out)
        out = self.fc2(out)
        out = self.sigmoid(out)

        return self.q_layer(out)

        # end

    def get_model_name(self):
        return compute_model_name(
            "AmpeCNNLSTM",
            self.n_qubits, self.n_layers,
            self.n_packets, self.n_features, self.num_classes
        )

        # end

    def get_model_name_short(self):
        return compute_model_name(
            "AmpeCNNLSTM",
            self.n_qubits, self.n_layers
        )

        # end
    
    # end class

#   ####################################################################    #
#   REFERENCES

#   https://www.quantum-inspire.com/kbase/number-of-shots
