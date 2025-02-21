import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

import utils


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model

def save_model(model, path):
    torch.save(model.state_dict(), path)


class MLP(nn.Module):
    
    def __init__(self):
        super(MLP, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(97, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6)
        )
        
    def forward(self, x):
        return self.model(x)
    
    
def train(model, optimizer, criterion, train_data, train_labels, test_data, test_labels, epochs=10):
    train_loss_ = []
    test_loss_ = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(train_data)
        loss = criterion(output, train_labels)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            test_output = model(test_data)
            test_loss = criterion(test_output, test_labels)
            print(f"Epoch: {epoch} Train Loss: {loss.item()} Test Loss: {test_loss.item()}")
            train_loss_.append(loss.item())
            test_loss_.append(test_loss.item())
            
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"model_{epoch}.pth")
            
    plt.plot(train_loss_, label="Train Loss")
    plt.plot(test_loss_, label="Test Loss")
    plt.yscale("log")
    plt.legend()
    plt.show()
            
    return model

if __name__ == "__main__":
    model = MLP()
    model = load_model(model, "models/model_90.pth")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    data, labels = utils.load_data(remove_na=True)
    
    train_data, test_data, train_labels, test_labels = utils.train_test_split(data, labels, test_size=0.2, random_state=42)
    
    train_data = torch.tensor(train_data.values).float()
    test_data = torch.tensor(test_data.values).float()
    train_labels = torch.tensor(train_labels["source_id"].values).long()
    test_labels = torch.tensor(test_labels["source_id"].values).long()
    
    # train(model, optimizer, criterion, train_data, train_labels, test_data, test_labels, epochs=100)
    
    test_output = model(test_data)
    print(test_output.shape)
    test_output = torch.argmax(test_output, dim=1)
    print(torch.bincount(test_output).numpy())

    print(accuracy_score(test_labels, test_output))
    
    
