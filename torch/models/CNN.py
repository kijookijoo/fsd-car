import torch
import torch.nn as nn

INPUT_DIMENSIONS = 1152

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=24,
                kernel_size=5,
                stride=2
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=24,
                out_channels=36,
                kernel_size=5,
                stride=2
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=36,
                out_channels=48,
                kernel_size=5,
                stride=2
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=48,
                out_channels=64,
                kernel_size=3,
                stride=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1
            ),
            nn.ReLU(),
        )
    
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(INPUT_DIMENSIONS, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 2)
        )
    
    def forward(self, x):
        x = self.cnn(x)
        x = self.regressor(x)
        return x