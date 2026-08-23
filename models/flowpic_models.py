"""

    flowpic_models.py
    di Mario Gabriele Carofano

    ...

"""

#   ####################################################################    #
#   LIBRERIE

import pennylane as qml
import torch.nn as nn

#   ####################################################################    #
#   CLASSICAL CONV2D MODEL

class FlowPicCNN(nn.Module):
    """ CNN2D per la classificazione di istogrammi FlowPic (1, 150, 150). """

    def __init__(self, n_qubits, n_layers, n_packets, n_features, num_classes, random_seed=42):
        super(FlowPicCNN, self).__init__()

        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 150 -> 75

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 75 -> 37

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # -> (B, 128, 1, 1), indipendente da eventuali variazioni di H,W
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_model_name(self):
        return f"FlowPicCNN_C{self.num_classes}"

    def get_model_name_short(self):
        return f"FlowPicCNN_C{self.num_classes}"

    # end

#   ####################################################################    #
#   QUANTUM HYBRID MODEL

class AmpHybridFlowPicCNN(nn.Module):
    """ CNN2D + layer quantistico (Amplitude Embedding) per FlowPic. """

    def __init__(self, n_qubits, n_layers, n_packets, n_features, num_classes, random_seed=42):
        super(AmpHybridFlowPicCNN, self).__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.num_classes = num_classes
        self.q_output_dim = 2 ** n_qubits

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )

        self.pre_quantum = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, self.q_output_dim),
            nn.Sigmoid(),
        )

        self.dev = qml.device("default.qubit", wires=n_qubits, seed=random_seed)

        @qml.qnode(self.dev, interface="torch")
        def qnode(inputs, weights):
            qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True, pad_with=0.0)
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return qml.probs(wires=range(n_qubits))

        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(qnode, weight_shapes)

        self.fc_out = nn.Linear(self.q_output_dim, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pre_quantum(x)
        x = self.q_layer(x)
        return self.fc_out(x)

    def get_model_name(self):
        return f"AmpHybridFlowPicCNN_Q{self.n_qubits}_L{self.n_layers}_C{self.num_classes}"

    def get_model_name_short(self):
        return f"AmpHybridFlowPicCNN_Q{self.n_qubits}_L{self.n_layers}"

    # end