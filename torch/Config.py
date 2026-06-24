from dataclasses import dataclass, field
from pathlib import Path

# -----------------------------
# ML Config
# -----------------------------
@dataclass
class Config:
    ### Paths
    # Adjust the dataset root here.
    path: Path = Path("data")
    csv_path: Path = field(init=False)

    ### Training hyperparameters
    # Allows for reproducible random nubmers
    seed: int = 42
    # Number of examples before updating model weights
    batch_size: int = 64 
    # train to test split ratio
    train_test_ratio: float = 0.8
    # Controls how big the weight updates are
    learning_rate: float = 1e-3 
    # Number of times the model has seen the dataset
    epochs: int = 50
    val_frac: float = 0.15  # last 15% used as val
    # Number of CPU proccesses preparing the data 
    num_workers: int = 2

    # saving
    model_path: Path = Path("../rpi/src/self_driving_pkg/self_driving_pkg/models/model.pt")
    
    ### Image preprocessing
    out_h: int = 66
    out_w: int = 200

    ### Motor thresholds
    max_throttle: int = 40
    max_angle: int = 90

    def __post_init__(self):
        self.csv_path = self.path / "labels.csv"
