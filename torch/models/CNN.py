import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_shape=(3, 66, 200), output_dim=2):
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

        # Infer the linear input size from the configured input shape instead of
        # relying on a hard-coded flatten dimension.
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            n_flat = self.cnn(dummy).flatten(1).shape[1]

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flat, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, output_dim)
        )
    
    def forward(self, x):
        x = self.cnn(x)
        x = self.regressor(x)
        return x
