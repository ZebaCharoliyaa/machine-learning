# optimizer_comparison.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Step 1: Define the Neural Network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)


# Step 2: Training Function
def train_model(optimizer_name='SGD', lr=0.01, batch_size=64):
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = SimpleNet().to(device)
    criterion = nn.CrossEntropyLoss()

    # Select optimizer
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == 'Momentum':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == 'RMSProp':
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError("Invalid optimizer name")

    losses = []
    accuracies = []

    for epoch in range(10):
        correct = 0
        total = 0
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        losses.append(running_loss / len(train_loader))
        accuracies.append(100. * correct / total)
        print(f"{optimizer_name} - Epoch {epoch+1}: Loss = {losses[-1]:.4f}, Accuracy = {accuracies[-1]:.2f}%")

    return losses, accuracies


# Step 3: Run Experiments
def run_experiments():
    optimizers = ['SGD', 'Momentum', 'RMSProp', 'Adam']
    results = {}

    for opt in optimizers:
        print(f"\nTraining with {opt} optimizer...")
        loss, acc = train_model(optimizer_name=opt, lr=0.01)
        results[opt] = (loss, acc)

    # Plotting loss
    plt.figure(figsize=(10, 5))
    for opt in optimizers:
        plt.plot(results[opt][0], label=f'{opt} Loss')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('TrainingLoss.png')
    plt.show()

    # Plotting accuracy
    plt.figure(figsize=(10, 5))
    for opt in optimizers:
        plt.plot(results[opt][1], label=f'{opt} Accuracy')
    plt.title('Training Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig('TrainingAccuracy.png')
    plt.show()


# Entry point
if __name__ == "__main__":
    run_experiments()
