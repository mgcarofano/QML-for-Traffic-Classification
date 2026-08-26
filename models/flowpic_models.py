"""

    flowpic_models.py
    di Mario Gabriele Carofano

    Questo modulo contiene la definizione di modelli classici e ibridi
    quantistici per la classificazione di istogrammi FlowPic, basati su
    strati convoluzionali bidimensionali (nn.Conv2d) per l'estrazione di
    feature spaziali dagli istogrammi, seguiti da strati fully-connected
    per la classificazione.

    Le classi mantengono la stessa interfaccia delle classi dei modelli quantistici,
    in modo da poter essere utilizzate in modo intercambiabile nel notebook principale.

    Args:
        n_qubits (int): Numero di qubit per il circuito quantistico (non utilizzato).
        n_layers (int): Numero di layer per StronglyEntanglingLayers (non utilizzato).
        n_packets (int): Numero di pacchetti nell'input.
        n_features (int): Numero di feature per pacchetto.
        num_classes (int): Numero di classi per la classificazione.
        n_shots (int, optional): Numero di misurazioni da eseguire (default: None) (non utilizzato).
        random_seed (int): Seed per il dispositivo quantistico (default: 42) (non utilizzato).

"""

#   ####################################################################    #
#   LIBRERIE e IMPORT

# import pennylane as qml
import torch
import torch.nn as nn
from model_selection import compute_model_name

#   ####################################################################    #
#   Classic CONV2D

class FlowPicCNN(nn.Module):
    """CNN2D per la classificazione di istogrammi FlowPic.

    Args:
        nn.Module: Base class for all neural network modules in PyTorch.
    """

    def __init__(self,
            n_qubits, n_layers,
            n_packets, n_features,
            num_classes, n_shots=None,
            random_seed=42,
        ):

        super(FlowPicCNN, self).__init__()

        # Salva i parametri come attributi.
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_packets = n_packets
        self.n_features = n_features
        self.num_classes = num_classes
        self.n_shots = n_shots
        self.random_seed = random_seed

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
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
        return compute_model_name(
            "FlowPicCNN",
            num_classes=self.num_classes
        )

        # end

    def get_model_name_short(self):
        return compute_model_name(
            "FlowPicCNN",
            num_classes=self.num_classes
        )

        # end

    # end class

#   ####################################################################    #
#   RESNET MODEL

class _ResidualBlock(nn.Module):
    """A class which implements a residual block, a building block
    used in convolutional neural networks (e.g. ResNet).
    They allow gradients to flow more easily through the network during
    training, which helps to mitigate the vanishing gradient problem.

    Args:
        nn.Module: Base class for all neural network modules in PyTorch.
    """	

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            skip_module: nn.Module = None
    ) -> None:
        """Initialize a new instance of the ResidualBlock.

        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            stride (int, optional): stride for the convolutional layers. Defaults to 1.
            skip_module (nn.Module, optional): a skip connection that is an optional parameter. Defaults to None.
        """	

        # Call the constructor of the parent class to ensure proper initialization.
        super(_ResidualBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(out_channels)
        )

        self.skip_module = skip_module
        self.relu = nn.ReLU(inplace=True)
        self.out_channels = out_channels

        return
    
        # end

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass of the residual block. The input tensor is passed through two convolutional layers, followed by a skip connection (if provided) and a ReLU activation function.

        Args:
            x (torch.Tensor): input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: output tensor of shape (B, C, H, W).
        """	

        # Apply the first convolutional layer.
        out = self.conv1(x)

        # Apply the second convolutional layer.
        out = self.conv2(out)

        # Apply downsampling if provided.
        residual = self.skip_module(x) if self.skip_module is not None else x

        # Add the residual connection.
        out += residual

        # Apply the ReLU activation function.
        out = self.relu(out)

        return out
    
        # end
    
    # end class

class _ResNet(nn.Module):
    """A class which implements a ResNet model,
    a type of CNN that uses residual connections.

    Args:
        nn.Module: Base class for all neural network modules in PyTorch.
    """	

    def __init__(self, block: nn.Module, layers: list[int], num_classes: int = 10):
        """Initialize a new instance of the ResNet.

        Args:
            block (nn.Module): the block to add.
            layers (list[int]): number of blocks in each layer.
            num_classes (int, optional): number of output classes. Defaults to 10.
        """	
        
        # Call the constructor of the parent class to ensure proper initialization.
        super(_ResNet, self).__init__()

        assert len(layers) == 4, "Layers must be a list of 4 integers."

        self.in_channels = 64
        self.num_classes = num_classes
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size = 7, stride = 2, padding = 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        )

        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)

        self.clf = nn.Sequential(
            # nn.AvgPool2d(7, stride=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

        return
    
        # end

    def _make_layer(
            self,
            block: nn.Module,
            out_channels: int,
            blocks: int,
            stride: int = 1
        ) -> nn.Module:
        """An helper function which adds the layers one by one along with the ResidualBlock.

        Args:
            block (nn.Module): the block to add.
            out_channels (int): number of output channels.
            blocks (int): number of blocks to add.
            stride (int, optional): stride for the convolutional layers. Defaults to 1.

        Returns:
            nn.Module: a sequential module containing the layers.
        """	

        skip_module = None

        if stride != 1 or self.in_channels != out_channels:
            skip_module = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, skip_module))

        self.in_channels = out_channels

        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)
    
        # end

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass of the ResNet model. The input tensor is passed through several convolutional layers, followed by residual blocks and a fully connected layer.

        Args:
            x (torch.Tensor): input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: output tensor of shape (B, NC), where NC is the number of classes.
        """	
        
        x = self.conv1(x)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.clf(x)

        return x
    
        # end

    # end class

class ResNetModel(_ResNet):
    """Questa classe implementa un wrapper per il modello ResNet, esponendo la stessa firma
    degli altri modelli del progetto. Permette di istanziare varianti standard di ResNet
    (es. ResNet18, ResNet34) tramite una factory comune.

    I parametri n_qubits, n_packets e n_features non sono utilizzati dall'architettura,
    ma vengono mantenuti solo per compatibilità. Il numero di layer (n_layers) seleziona
    la configurazione dei ResidualBlock secondo lo schema standard:
    - ResNet18: [2, 2, 2, 2]
    - ResNet34: [3, 4, 6, 3]

    Args:
        _ResNet: classe base che implementa l'architettura ResNet generica
        con blocchi residui configurabili.
    """

    def __init__(self,
            n_qubits, n_layers,
            n_packets, n_features,
            num_classes, n_shots=None,
            random_seed=42,
        ):
        """Initialize a new instance of the ResNet.

        Raises:
            NotImplementedError: se n_layers non è 16, 18 o 34.
        """

        # ResNet16-custom
        if n_layers == 16:
            super(ResNetModel, self).__init__(
                _ResidualBlock,
                [1, 2, 3, 1],
                num_classes
            )

        elif n_layers == 18:
            super(ResNetModel, self).__init__(
                _ResidualBlock,
                [2, 2, 2, 2],
                num_classes
            )

        elif n_layers == 34:
            super(ResNetModel, self).__init__(
                _ResidualBlock,
                [3, 4, 6, 3],
                num_classes
            )

        else:
            raise NotImplementedError(
                f"Unsupported number of layers: {n_layers}. "
                "Supported values are [16, 18, 34]."
            )

            # end if n_layers

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_packets = n_packets
        self.n_features = n_features
        self.num_classes = num_classes
        self.n_shots = n_shots
        self.random_seed = random_seed

        return

        # end
    
    def get_model_name(self):
        return compute_model_name(
            "ResNetModel",
            n_layers=self.n_layers,
            num_classes=self.num_classes
        )

        # end

    def get_model_name_short(self):
        return compute_model_name(
            "ResNetModel",
            n_layers=self.n_layers,
            num_classes=self.num_classes
        )

        # end
    
    # end class

#   ####################################################################    #
#   LeNet5 MODEL

# TODO: aggiungere LeNet5 dal paper originale.

#   ####################################################################    #
#   QUANTUM HYBRID MODEL

# TODO: Implementare un modello ibrido quantistico per la classificazione
# di istogrammi FlowPic.
