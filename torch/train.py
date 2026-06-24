import torch
import torch.nn as nn
from Config import Config
from models.CNN import CNN
from FSDDataset import FSDDataset
from torch.utils.data import DataLoader, random_split

SEED = 12
torch.manual_seed(seed=SEED)
BATCH_SIZE = 32
device = "cuda" if torch.cuda.is_available() else "cpu"

config = Config()

def create_train_test_data():
    label_csv = "../data/labels.csv"
    img_path = "../data/images"
    full_data = FSDDataset(label_csv, img_path)
    train_size = config.train_test_ratio * len(full_data)
    test_size = (1 - config.train_test_ratio) * len(full_data)
    train_data, test_data = random_split(
        full_data,
        [train_size, test_size]
    )
    return train_data, test_data

def create_train_dataloader(train_data):
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    return train_dataloader

def create_test_dataloader(test_data):
    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    return test_dataloader

def train(model, loss_fn, optimizer, train_dataloader):
    model.train()
    total_loss = 0
    for batch_idx, (X,y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()        
        pred = model(X)
        
        loss = loss_fn(pred, y)        
        total_loss += loss
        
        loss.backward()
        optimizer.step()

def test(model, loss_fn, test_dataloader):
    model.eval()
    min_loss = float('inf')
    total_loss = 0
    accuracy = 0
    with torch.inference_mode():
        for batch_idx, (X,y) in enumerate(test_dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            total_loss += loss
        
        unit_loss = total_loss / len(test_dataloader)
        if unit_loss < min_loss:
            min_loss = unit_loss
            torch.save({
                "model_state": model.state_dict(),
            }, config.model_path)
            print("model saved at" + config.model_path)

def train_test_loop(epochs):
    cnn = CNN()
    train_data, test_data = create_train_test_data()
    train_dataloader = create_train_dataloader(train_data)
    test_dataloader = create_test_dataloader(test_data)
    loss_fn_mse = nn.MSELoss()
    optimizer_adam = torch.optim.Adam(
        params=cnn.parameters(),
        lr=config.learning_rate
    )
    
    for epoch in range(epochs):
        train(cnn, loss_fn_mse, optimizer_adam, train_dataloader)
        test(cnn, loss_fn_mse, test_dataloader)
    
    
        
         