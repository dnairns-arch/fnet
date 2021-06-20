import os
import pandas as pd
import torch

from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F


# JSONL file paths (Exported from Doccano)
training_dataset_path = "~/python/FNet/all.jsonl"
test_dataset_path = "~/python/FNet/test.jsonl"

#
training_data = pd.read_json(path_or_buf=dataset_path, lines=True)
test_data = pd.read_json(path_or_buf=dataset_path, lines=True)

#Checking File Outputs
#print(f"data: {data['data']}")
#print(f"data: {data['label']}")

# Creating custom dataset to use with DataLoader
class ContractDataset(Dataset):
    def __init__(self, data, transform=None, target_transform=None):
        self.labels = data['label']
        self.features = data['data']  
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features.iloc[idx,0]
        label = self.labels.iloc[idx,0]
        if self.transform:
            feature = self.transform(feature)
        if self.target_transform:
            label = self.target_transform(label)
        return feature, label


# Data Paths
training_data = ContractDataset(training_data)
test_data = ContractDataset(test_data)

# Batch Size for training
batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)



class FeedForward(nn.Module):
    def __init__(self, num_features, expansion_factor, dropout):
        super().__init__()
        num_hidden = expansion_factor * num_features
        self.fc1 = nn.Linear(num_features, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_features)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout1(F.gelu(self.fc1(x)))
        out = self.dropout2(self.fc2(x))
        return out


def fourier_transform(x):
    return torch.fft.fft2(x, dim=(-1, -2)).real


class FNetEncoderLayer(nn.Module):
    def __init__(self, d_model, expansion_factor, dropout):
        super().__init__()
        self.ff = FeedForward(d_model, expansion_factor, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = fourier_transform(x)
        x = self.norm1(x + residual)
        residual = x
        x = self.ff(x)
        out = self.norm2(x + residual)
        return out


class FNet(nn.TransformerEncoder):
    def __init__(
        self, d_model=257, expansion_factor=2, dropout=0.5, num_layers=7,
    ):
        encoder_layer = FNetEncoderLayer(d_model, expansion_factor, dropout)
        super().__init__(encoder_layer=encoder_layer, num_layers=num_layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Instantiate model, loss fn, and optimizer
model = FNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        #Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss:{loss:>7f} [{current:>5d}/{size:>5d}]")

# Number of training epochs
epochs = 10

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model)
print("Done!")