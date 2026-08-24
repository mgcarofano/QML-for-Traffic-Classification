"""

    flowpic_models.py
    di Mario Gabriele Carofano

    Questo modulo contiene la definizione di modelli classici e ibridi
    quantistici per la classificazione di istogrammi FlowPic, basati su
    strati convoluzionali bidimensionali (nn.Conv2d) per l'estrazione di
    feature spaziali dagli istogrammi, seguiti da strati fully-connected
    per la classificazione.

"""

#   ####################################################################    #
#   LIBRERIE

import torch
import torch.nn as nn
import pennylane as qml

#   ####################################################################    #
#   CLASSICAL CONV2D MODEL

class FlowPicCNN(nn.Module):
    """ CNN2D per la classificazione di istogrammi FlowPic. """

    def __init__(self, n_qubits, n_layers, n_packets, n_features, num_classes, random_seed=42):
        super(FlowPicCNN, self).__init__()

        self.num_classes = num_classes

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
        """Restituisce una stringa con il nome del modello che riassume i parametri principali.

        Returns:
            str: nome del modello.
        """

        return f"FlowPicCNN_C{self.num_classes}"

        # end

    def get_model_name_short(self):
        """Restituisce una stringa con il nome breve del modello.

        Returns:
            str: nome breve del modello.
        """

        return self.get_model_name()

        # end

    # end class

#   ####################################################################    #
#   RESNET MODEL

class ResidualBlock(nn.Module):
    """A class which implements a residual block, a building block used in convolutional neural networks (e.g. ResNet). They allow gradients to flow more easily through the network during training, which helps to mitigate the vanishing gradient problem.

    Args:
        nn (nn.Module): Inherits from nn.Module, the base class for all neural network modules in PyTorch.
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
        super(ResidualBlock, self).__init__()

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

class ResNet(nn.Module):
    """A class which implements a ResNet model, a type of CNN that uses residual connections.

    Args:
        nn (nn.Module): Inherits from nn.Module, the base class for all neural network modules in PyTorch.
    """	

    def __init__(self, block: nn.Module, layers: list[int], num_classes: int = 10):
        """Initialize a new instance of the ResNet.

        Args:
            block (nn.Module): the block to add.
            layers (list[int]): number of blocks in each layer.
            num_classes (int, optional): number of output classes. Defaults to 10.
        """	
        
        # Call the constructor of the parent class to ensure proper initialization.
        super(ResNet, self).__init__()

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

    def get_model_name(self):
        """Restituisce una stringa con il nome del modello che riassume i parametri principali.

        Returns:
            str: nome del modello.
        """

        return f"ResNet_C{self.num_classes}"

        # end

    def get_model_name_short(self):
        """Restituisce una stringa con il nome breve del modello.

        Returns:
            str: nome breve del modello.
        """

        return self.get_model_name()

        # end

    # end class

#   ####################################################################    #
#   QUANTUM HYBRID MODEL

# TODO: Implementare un modello ibrido quantistico per la classificazione
# di istogrammi FlowPic.
