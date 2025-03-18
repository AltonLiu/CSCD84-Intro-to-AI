"""
CSCD84 - Artificial Intelligence, Winter 2025, Assignment 3
B. Chan
"""


import numpy as np
import torch.nn as nn
import torch


IMG_DIM = (1, 28, 28)
FLATTENED_IMG_DIM = np.prod(IMG_DIM)
NUM_CLASSES = 10


class MLP(nn.Module):
    """
    A two-layered multilayer perceptron
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hidden_1 = nn.Linear(
            in_features=FLATTENED_IMG_DIM,
            out_features=FLATTENED_IMG_DIM,
            bias=True,
        )

        self.hidden_2 = nn.Linear(
            in_features=FLATTENED_IMG_DIM,
            out_features=FLATTENED_IMG_DIM,
            bias=True,
        )

        self.out = nn.Linear(
            in_features=FLATTENED_IMG_DIM,
            out_features=NUM_CLASSES,
            bias=True,
        )

    def forward(self, x):
        # Flatten the image
        x = x.reshape(len(x), -1)

        x = self.hidden_1(x)
        x = nn.functional.relu(x)
        x = self.hidden_2(x)
        x = nn.functional.relu(x)
        x = self.out(x)

        return x


class SkipConnectionMLP(nn.Module):
    """
    A two-layered multilayer perceptron with a skip connection
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hidden_1 = nn.Linear(
            in_features=FLATTENED_IMG_DIM,
            out_features=FLATTENED_IMG_DIM,
            bias=True,
        )

        self.hidden_2 = nn.Linear(
            in_features=FLATTENED_IMG_DIM,
            out_features=FLATTENED_IMG_DIM,
            bias=True,
        )

        self.out = nn.Linear(
            in_features=FLATTENED_IMG_DIM,
            out_features=NUM_CLASSES,
            bias=True,
        )

    def forward(self, x):
        # Flatten the image
        x = x.reshape(len(x), -1)

        out = self.hidden_1(x)
        out = nn.functional.relu(out)
        out = self.hidden_2(out) + x
        out = nn.functional.relu(out)
        out = self.out(out)

        return out

class CustomModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, 
                      out_channels=32, 
                      kernel_size=3, 
                      padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, 
                         stride=2)
        )
        
        self.hidden_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, 
                      out_channels=64, 
                      kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        
        self.out = nn.Linear(in_features=120, out_features=10)
        
    def forward(self, x):
        out = self.hidden_1(x)
        out = self.hidden_2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.out(out)
        
        return out
"""
class CustomModel(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv1 = nn.Conv2d(
            FLATTENED_IMG_DIM, 
            FLATTENED_IMG_DIM, 
            ,
            (3, 3), padding='same')
        
    def forward(self, x):
        # Flatten the image
        x = x.reshape(len(x), -1)

        return out
"""