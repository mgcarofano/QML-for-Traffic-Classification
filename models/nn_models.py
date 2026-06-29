import pennylane as qml
import torch.nn as nn


SIMULATOR = "default.qubit"

# -------------------- Amplitude embedding quantum model --------------------

class AmpHybridModel(nn.Module):
    def __init__(self, n_qubits, n_layers, n_packets, n_features, num_classes, random_seed=42):
        """
        Modello ibrido quantistico-classico con Amplitude Embedding.

        Args:
            n_qubits (int): Numero di qubit per il circuito quantistico
            n_layers (int): Numero di layer per StronglyEntanglingLayers
            n_packets (int): Numero di pacchetti nell'input
            n_features (int): Numero di feature per pacchetto
            num_classes (int): Numero di classi per la classificazione
            random_seed (int): Seed per il dispositivo quantistico (default: 42)
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

    def forward(self, x):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.sigmoid(x)
        x = self.q_layer(x)
        return self.dense2(x)
        # return self.softmax(x)

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

    def get_model_name(self):
        """
        Restituisce una stringa con il nome del modello che riassume i parametri principali.

        Returns:
            str: Nome del modello nel formato "AmpHybrid_Qn_Ln_PxF_Cc"
                 dove n=n_qubits, n=n_layers, P=n_packets, F=n_features, C=num_classes
        """
        return f"AmpHybrid_Q{self.n_qubits}_L{self.n_layers}_{self.n_packets}x{self.n_features}_C{self.num_classes}"

    def get_model_name_short(self):
        """
        Restituisce una stringa con il nome breve del modello.

        Returns:
            str: Nome breve del modello nel formato "AmpHybrid_Qn_Ln"
                 dove n=n_qubits, n=n_layers
        """
        return f"AmpHybrid_Q{self.n_qubits}_L{self.n_layers}"


# -------------------- Angle embedding quantum model --------------------

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
        """
        Restituisce una stringa con il nome del modello che riassume i parametri principali.

        Returns:
            str: Nome del modello nel formato "AngleHybrid_Qn_Ln_PxF_Cc"
                 dove n=n_qubits, n=n_layers, P=n_packets, F=n_features, C=num_classes
        """
        return f"AngleHybrid_Q{self.n_qubits}_L{self.n_layers}_{self.n_packets}x{self.n_features}_C{self.num_classes}"

    def get_model_name_short(self):
        """
        Restituisce una stringa con il nome breve del modello.

        Returns:
            str: Nome breve del modello nel formato "AngleHybrid_Qn_Ln"
                 dove n=n_qubits, n=n_layers
        """
        return f"AngleHybrid_Q{self.n_qubits}_L{self.n_layers}"


# -------------------- Ring quantum model --------------------

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
        """
        Restituisce una stringa con il nome del modello che riassume i parametri principali.

        Returns:
            str: Nome del modello nel formato "RingHybrid_Qn_Ln_PxF_Cc"
                 dove n=n_qubits, n=n_layers, P=n_packets, F=n_features, C=num_classes
        """
        return f"RingHybrid_Q{self.n_qubits}_L{self.n_layers}_{self.n_packets}x{self.n_features}_C{self.num_classes}"

    def get_model_name_short(self):
        """
        Restituisce una stringa con il nome breve del modello.

        Returns:
            str: Nome breve del modello nel formato "RingHybrid_Qn_Ln"
                 dove n=n_qubits, n=n_layers
        """
        return f"RingHybrid_Q{self.n_qubits}_L{self.n_layers}"


# -------------------- Waterfall quantum model --------------------

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
        """
        Restituisce una stringa con il nome del modello che riassume i parametri principali.

        Returns:
            str: Nome del modello nel formato "WaterfallHybrid_Qn_Ln_PxF_Cc"
                 dove n=n_qubits, n=n_layers, P=n_packets, F=n_features, C=num_classes
        """
        return f"WaterfallHybrid_Q{self.n_qubits}_L{self.n_layers}_{self.n_packets}x{self.n_features}_C{self.num_classes}"

    def get_model_name_short(self):
        """
        Restituisce una stringa con il nome breve del modello.

        Returns:
            str: Nome breve del modello nel formato "WaterfallHybrid_Qn_Ln"
                 dove n=n_qubits, n=n_layers
        """
        return f"WaterfallHybrid_Q{self.n_qubits}_L{self.n_layers}"


# -------------------- Classic conv1d --------------------
class TrafficCNN(nn.Module):
    def __init__(self, n_qubits, n_layers, n_packets, n_features, num_classes, random_seed=42):
        super(TrafficCNN, self).__init__()

        self.num_classes = num_classes

        self.features = nn.Sequential(
            # Input: (B, 4, 36)
            nn.Conv1d(4, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.Dropout(0.2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.Dropout(0.2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Global Average Pooling -> (B, 256, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_model_name(self):
        return f"TrafficCNN_C{self.num_classes}"

    def get_model_name_short(self):
        return f"TrafficCNN_C{self.num_classes}"