from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split

from Config import Config
from FSDDataset import FSDDataset
from models.CNN import CNN


class SyntheticDrivingDataset(Dataset):
    def __init__(self, size: int, input_shape=(3, 66, 200)):
        self.size = size
        self.input_shape = input_shape

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image = torch.rand(self.input_shape, dtype=torch.float32)
        label = torch.zeros(2, dtype=torch.float32)
        return image, label


def build_transforms():
    # Keep this small for now so the pipeline is easy to wire before you add
    # augmentation or normalization specific to the real dataset.
    try:
        from torchvision import transforms
    except ImportError as exc:
        raise ImportError("torchvision is required for image transforms") from exc

    return transforms.Compose(
        [
            transforms.Resize((66, 200)),
            transforms.ToTensor(),
        ]
    )


def create_datasets(config: Config):
    if not config.csv_path.exists():
        print(f"warning: {config.csv_path} not found, using synthetic data for now")
        dataset = SyntheticDrivingDataset(size=128, input_shape=(3, config.out_h, config.out_w))
        train_size = int(len(dataset) * config.train_test_ratio)
        val_size = len(dataset) - train_size
        generator = torch.Generator().manual_seed(config.seed)
        return random_split(dataset, [train_size, val_size], generator=generator)

    dataset = FSDDataset(
        csv_file_path=config.csv_path,
        img_dir=config.path / "images",
        transform=build_transforms(),
    )

    train_size = int(len(dataset) * config.train_test_ratio)
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(config.seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def create_dataloader(dataset, batch_size: int, shuffle: bool, num_workers: int):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def run_epoch(model, dataloader, loss_fn, optimizer=None, device="cpu"):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    num_batches = 0

    context = torch.enable_grad() if is_train else torch.inference_mode()
    with context:
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            pred = model(X)
            loss = loss_fn(pred, y)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


def train_test_loop(config: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, val_data = create_datasets(config)
    train_loader = create_dataloader(train_data, config.batch_size, True, config.num_workers)
    val_loader = create_dataloader(val_data, config.batch_size, False, config.num_workers)

    model = CNN(input_shape=(3, config.out_h, config.out_w)).to(device)
    loss_fn = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    best_val_loss = float("inf")
    model_path = Path(config.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.epochs):
        train_loss = run_epoch(model, train_loader, loss_fn, optimizer=optimizer, device=device)
        val_loss = run_epoch(model, val_loader, loss_fn, optimizer=None, device=device)

        print(f"epoch={epoch + 1} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config.__dict__,
                    "val_loss": best_val_loss,
                },
                model_path,
            )
            print(f"saved model to {model_path}")


def main():
    config = Config()
    train_test_loop(config)


if __name__ == "__main__":
    main()
