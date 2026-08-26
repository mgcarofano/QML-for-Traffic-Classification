"""

    quantum_models.py
    di MatteoRichardGaudino

    ...

"""

#   ####################################################################    #
#   LIBRERIE e IMPORT

import pennylane as qml
import torch.nn as nn
from constants import SIMULATOR
from model_selection import compute_model_name

#   ####################################################################    #
#   Amplitude embedding quantum model

class AmpHybridModel(nn.Module):
    def __init__(self, n_qubits, n_layers, n_packets, n_features, num_classes, random_seed=42):
        """
        Modello ibrido quantistico-classico con Amplitude Embedding.

        Args:
            n_qubits (int): Numero di qubit per il circuito quantistico.
            n_layers (int): Numero di layer per StronglyEntanglingLayers.
            n_packets (int): Numero di pacchetti nell'input.
            n_features (int): Numero di feature per pacchetto.
            num_classes (int): Numero di classi per la classificazione.
            random_seed (int): Seed per il dispositivo quantistico. Default è 42.
        """

        super(AmpHybridModel, self).__init__()

        # Salva i parametri come attributi
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_packets = n_packets
        self.n_features = n_features
        self.num_classes = num_classes
        self.random_seed = random_seed

        # Definizione del dispositivo quantistico
        self.dev = qml.device(SIMULATOR, wires=n_qubits, seed=random_seed)

        # Definizione del QNode
        @qml.qnode(self.dev, interface="torch")
        def qnode(inputs, weights):
            # Feature map
            qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True, pad_with=0.0)

            # Ansatz
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

            # Processo di misurazione
            return qml.probs(wires=range(n_qubits))

        self.qnode = qnode

        # Definizione delle forme dei pesi per il layer quantistico
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}

        # 1. Flatten
        # Input shape originale: (n_packets, n_features)
        # Dimensione appiattita: n_packets * n_features
        self.flatten = nn.Flatten()

        # 2. Dense Layer (Pre-processing per il quantum layer)
        # Output dim deve essere 2^n_qubits per l'AmplitudeEmbedding
        input_dim = n_packets * n_features
        self.dense1 = nn.Linear(input_dim, 2**n_qubits)
        self.sigmoid = nn.Sigmoid()

        # 3. Quantum Layer
        # Sostituisce qml.qnn.KerasLayer
        self.q_layer = qml.qnn.TorchLayer(self.qnode, weight_shapes)

        # 4. Output Dense Layer
        # Input dim è 2^n_qubits (output di qml.probs)
        self.dense2 = nn.Linear(2**n_qubits, num_classes)
        # self.softmax = nn.Softmax(dim=1)

        # end

    def forward(self, x):

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.sigmoid(x)
        x = self.q_layer(x)

        return self.dense2(x)
    
        # return self.softmax(x)

        # end

    def quantum_forward(self, x):
        """
        Metodo per eseguire solo il passaggio attraverso il layer quantistico.
        Utile per l'estrazione delle caratteristiche quantistiche.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output del layer quantistico
        """

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.sigmoid(x)
        x = self.q_layer(x)

        return x

        # end

    def get_model_name(self):
        return compute_model_name(
            "AmpHybrid",
            self.n_qubits, self.n_layers,
            self.n_packets, self.n_features, self.num_classes
        )

        # end

    def get_model_name_short(self):
        return compute_model_name(
            "AmpHybrid",
            self.n_qubits, self.n_layers
        )

        # end
    
    # end class

#   ####################################################################    #
#   Angle embedding quantum model 

class AngleHybridModel(nn.Module):
    def __init__(self, n_qubits, n_layers, n_packets, n_features, num_classes, random_seed=42):
        """
        Modello ibrido quantistico-classico con Angle Embedding.

        Args:
            n_qubits (int): Numero di qubit per il circuito quantistico
            n_layers (int): Numero di layer per StronglyEntanglingLayers
            n_packets (int): Numero di pacchetti nell'input
            n_features (int): Numero di feature per pacchetto
            num_classes (int): Numero di classi per la classificazione
            random_seed (int): Seed per il dispositivo quantistico (default: 42)
        """
        super(AngleHybridModel, self).__init__()

        # Salva i parametri come attributi
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_packets = n_packets
        self.n_features = n_features
        self.num_classes = num_classes
        self.random_seed = random_seed

        # Definizione del dispositivo quantistico
        self.dev = qml.device(SIMULATOR, wires=n_qubits, seed=random_seed)

        # Definizione del QNode
        @qml.qnode(self.dev, interface="torch")
        def qnode(inputs, weights):
            # Feature map
            qml.AngleEmbedding(inputs, wires=range(n_qubits))

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
        # Output dim deve essere n_qubits per l'AngleEmbedding
        input_dim = n_packets * n_features
        self.dense1 = nn.Linear(input_dim, n_qubits)
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
            "AngleHybrid",
            self.n_qubits, self.n_layers,
            self.n_packets, self.n_features, self.num_classes
        )

        # end

    def get_model_name_short(self):
        return compute_model_name(
            "AngleHybrid",
            self.n_qubits, self.n_layers
        )

        # end
    
    # end class

#   ####################################################################    #
#   Ring quantum model

class RingHybridModel(nn.Module):
    def __init__(self, n_qubits, n_layers, n_packets, n_features, num_classes, random_seed=42):
        """
        Modello ibrido quantistico-classico con Ring Embedding.

        Args:
            n_qubits (int): Numero di qubit per il circuito quantistico
            n_layers (int): Numero di layer per StronglyEntanglingLayers
            n_packets (int): Numero di pacchetti nell'input
            n_features (int): Numero di feature per pacchetto
            num_classes (int): Numero di classi per la classificazione
            random_seed (int): Seed per il dispositivo quantistico (default: 42)
        """
        super(RingHybridModel, self).__init__()

        # Salva i parametri come attributi
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_packets = n_packets
        self.n_features = n_features
        self.num_classes = num_classes
        self.random_seed = random_seed

        # Definizione del dispositivo quantistico
        self.dev = qml.device(SIMULATOR, wires=n_qubits, seed=random_seed)

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

class WaterfallHybridModel(nn.Module):
    def __init__(self, n_qubits, n_layers, n_packets, n_features, num_classes, random_seed=42):
        """
        Modello ibrido quantistico-classico con Waterfall Embedding.

        Args:
            n_qubits (int): Numero di qubit per il circuito quantistico
            n_layers (int): Numero di layer per StronglyEntanglingLayers
            n_packets (int): Numero di pacchetti nell'input
            n_features (int): Numero di feature per pacchetto
            num_classes (int): Numero di classi per la classificazione
            random_seed (int): Seed per il dispositivo quantistico (default: 42)
        """
        super(WaterfallHybridModel, self).__init__()

        # Salva i parametri come attributi
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_packets = n_packets
        self.n_features = n_features
        self.num_classes = num_classes
        self.random_seed = random_seed

        # Definizione del dispositivo quantistico
        self.dev = qml.device(SIMULATOR, wires=n_qubits, seed=random_seed)

        # Definizione del QNode
        @qml.qnode(self.dev, interface="torch")
        def qnode(inputs, weights):
            # Feature map - split inputs in two parts
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

        self.qnode = qnode

        # Definizione delle forme dei pesi per il layer quantistico
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}

        # 1. Flatten
        self.flatten = nn.Flatten()

        # 2. Dense Layer (Pre-processing per il quantum layer)
        # Output dim deve essere 2*n_qubits per il Waterfall embedding
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

class AmpCnn(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, n_qubits, n_layers, n_packets, n_features, num_classes, random_seed=42):
        super(AmpCnn, self).__init__()

        # Parametri
        self.n_qubits = n_qubits

        # Dimensione output quantistico (es. 2^5 = 32)
        self.q_output_dim = 2**n_qubits

        self.n_layers = n_layers
        self.n_packets = n_packets
        self.n_features = n_features
        self.num_classes = num_classes

        # Configurazione Pennylane
        self.dev = qml.device(SIMULATOR, wires=n_qubits, seed=random_seed)

        @qml.qnode(self.dev, interface="torch")
        def qnode(inputs, weights):
            qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True, pad_with=0.0)
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return qml.probs(wires=range(n_qubits))

        weight_shapes = {"weights": (n_layers, n_qubits, 3)}

        # --- ARCHITETTURA ---

        # 1. Pre-processing Classico
        self.flatten_in = nn.Flatten()
        self.dense_pre = nn.Linear(n_packets * n_features, self.q_output_dim)
        self.activation_pre = nn.Sigmoid()

        # 2. Quantum Layer
        self.q_layer = qml.qnn.TorchLayer(qnode, weight_shapes)

        # 3. Strati Convoluzionali 1D (Aggiunti)
        # Input shape per Conv1d: (Batch, Channels, Length) -> (Batch, 1, 2^n_qubits)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2) # Riduce la lunghezza della sequenza a metà

        # 4. Strati Densi Finali (Aggiunti)
        # Calcolo della dimensione dopo convoluzioni e pooling:
        # La lunghezza cala da 2^n_qubits a (2^n_qubits / 2) a causa del MaxPool1d
        self.flatten_conv = nn.Flatten()
        flattened_size = 32 * (self.q_output_dim // 2)

        self.fc1 = nn.Linear(flattened_size, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

        # end

    def forward(self, x):
        # Input -> Dense -> Quantum
        x = self.flatten_in(x)
        x = self.dense_pre(x)
        x = self.activation_pre(x)
        x = self.q_layer(x)

        # Reshape per Conv1d: (Batch, 1, 2^n_qubits)
        x = x.view(-1, 1, self.q_output_dim)

        # Fase Convoluzionale 1D
        x = self.relu1(self.conv1(x))
        x = self.pool(self.relu2(self.conv2(x)))

        # Fase Densa
        x = self.flatten_conv(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)

        return x

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

class CnnAmpCnn(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, n_qubits, n_layers, n_packets, n_features, num_classes, random_seed=42):
        super(CnnAmpCnn, self).__init__()

        # Parametri
        self.n_qubits = n_qubits
        self.q_input_dim = 2**n_qubits  # Dimensione necessaria per AmplitudeEmbedding
        self.n_layers = n_layers
        self.n_packets = n_packets
        self.n_features = n_features
        self.num_classes = num_classes

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
        self.dev = qml.device(SIMULATOR, wires=n_qubits)

        @qml.qnode(self.dev, interface="torch")
        def qnode(inputs, weights):
            # L'input qui deve avere dimensione 2^n_qubits
            qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True, pad_with=0.0)
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return qml.probs(wires=range(n_qubits))

        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
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
