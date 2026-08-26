"""

    classic_models.py
    di MatteoRichardGaudino

    ...

"""

#   ####################################################################    #
#   LIBRERIE e IMPORT

import torch.nn as nn
from model_selection import compute_model_name

#   ####################################################################    #
#   Dense Model

class Dense(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, n_qubits, n_layers, n_packets, n_features, num_classes, random_seed=42):
        super(Dense, self).__init__()

        # Salva i parametri come attributi
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_packets = n_packets
        self.n_features = n_features
        self.num_classes = num_classes
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

        # 3. Quantum Layer
        # Sostituisce qml.qnn.KerasLayer
        # self.q_layer = nn.Linear(2**n_qubits, 2**n_qubits)  # Placeholder per il quantum layer

        # 4. Output Dense Layer
        # Input dim è 2^n_qubits (output di qml.probs)
        self.dense2 = nn.Linear(2**n_qubits, num_classes)
        # self.softmax = nn.Softmax(dim=1)

        # end

    def forward(self, x):

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu(x)
        # x = self.q_layer(x)

        return self.dense2(x)
        # return self.softmax(x)

        # end

    def get_model_name(self):
        return compute_model_name("Dense", num_classes=self.num_classes)

        # end

    def get_model_name_short(self):
        return compute_model_name("Dense", num_classes=self.num_classes)

        # end
    
    # end class

#   ####################################################################    #
#   Classic CONV1D

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
        return compute_model_name("TrafficCNN", num_classes=self.num_classes)

        # end

    def get_model_name_short(self):
        return compute_model_name("TrafficCNN", num_classes=self.num_classes)

        # end
    
    # end class

#   ####################################################################    #
#   Classical Twin Model

class ClassicalTwinModel(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, n_qubits, n_layers, n_packets, n_features, num_classes, random_seed=42):
        super(ClassicalTwinModel, self).__init__()

        self.n_qubits = n_qubits
        self.q_output_dim = 2**n_qubits  # Dimensione output quantistico (es. 2^5 = 32)
        self.n_layers = n_layers
        self.n_packets = n_packets
        self.n_features = n_features
        self.num_classes = num_classes

        # --- ARCHITETTURA ---

        # 1. Pre-processing (Uguale all'originale)
        self.flatten_in = nn.Flatten()
        self.dense_pre = nn.Linear(n_packets * n_features, self.q_output_dim)
        self.activation_pre = nn.Sigmoid()

        # 2. Strato che sostituisce il Quantum Layer
        # Usiamo un Linear che mappa 2^n_qubits in 2^n_qubits
        # Questo sostituisce il lavoro dei gate quantistici
        self.classical_replacement = nn.Linear(self.q_output_dim, self.q_output_dim)
        self.activation_replacement = nn.ReLU() # o Sigmoid per simulare meglio i limiti del quantum

        # 3. Strati Convoluzionali 1D (Uguali all'originale)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)

        # 4. Strati Densi Finali (Uguali all'originale)
        self.flatten_conv = nn.Flatten()
        flattened_size = 32 * (self.q_output_dim // 2)

        self.fc1 = nn.Linear(flattened_size, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

        # end

    def forward(self, x):
        # Input -> Dense -> Sostituto Classico
        x = self.flatten_in(x)
        x = self.dense_pre(x)
        x = self.activation_pre(x)

        # Qui bypassiamo il quantum layer e usiamo il sostituto classico
        x = self.classical_replacement(x)
        x = self.activation_replacement(x)

        # Reshape per Conv1d
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

#   ####################################################################    #
#   Classical Light Model

class ClassicalLight(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, n_qubits, n_layers, n_packets, n_features, num_classes, random_seed=42):
        super(ClassicalLight, self).__init__()

        # Parametri
        self.n_qubits = n_qubits

        # Dimensione output quantistico (es. 2^5 = 32)
        self.q_output_dim = 2**n_qubits

        self.n_layers = n_layers
        self.n_packets = n_packets
        self.n_features = n_features
        self.num_classes = num_classes
        self.latent_dim = self.q_output_dim

        # --- ARCHITETTURA ---

        # 1. Pre-processing Classico
        self.flatten_in = nn.Flatten()
        self.dense_pre = nn.Linear(n_packets * n_features, self.latent_dim)
        self.activation_pre = nn.Sigmoid()
        # self.activation_pre = nn.ReLU()

        # 2. Strati Convoluzionali 1D
        # Input shape per Conv1d: (Batch, Channels, Length) -> (Batch, 1, latent_dim)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)

        # 3. Strati Densi Finali
        # Calcolo della dimensione dopo il MaxPool1d (lunghezza dimezzata)
        self.flatten_conv = nn.Flatten()
        flattened_size = 32 * (self.latent_dim // 2)

        self.fc1 = nn.Linear(flattened_size, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

        # end

    def forward(self, x):
        # Input -> Dense (sostituisce il blocco Quantum)
        x = self.flatten_in(x)
        x = self.dense_pre(x)
        x = self.activation_pre(x)

        # Reshape per Conv1d: (Batch, Channels, Length)
        x = x.view(-1, 1, self.latent_dim)

        # Fase Convoluzionale 1D
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool(x)

        # Fase Densa
        x = self.flatten_conv(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)

        return x

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
