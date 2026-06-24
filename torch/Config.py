from dataclasses import dataclass

# -----------------------------
# ML Config
# -----------------------------
@dataclass
class Config:
    ### Paths
    # Adjust the dateset to be used here
    path: str = "data"
    csv_path: str = path + "/labels.csv"

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
    model_path: str = "../rpi/src/self_driving_pkg/self_driving_pkg/models/model.pt"
    
    ### Image preprocessing
    out_h: int = 66
    out_w: int = 200

    ### Motor thresholds
    max_throttle: int = 40
    max_angle: int = 90 