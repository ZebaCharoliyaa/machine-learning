# # Filename: transfer_learning_mnist_resnet.py

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms, models
# from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
# from torchvision.models import ResNet18_Weights
# # Device

# print("Device:", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Data transform: Resize + 3 Channels + Normalize (ImageNet stats)
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.Grayscale(num_output_channels=3),  # Convert 1 channel to 3
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],   # ImageNet mean
#                          [0.229, 0.224, 0.225])   # ImageNet std
# ])

# # Datasets & Loaders
# train_dataset = datasets.MNIST(root='.', train=True, transform=transform, download=True)
# test_dataset = datasets.MNIST(root='.', train=False, transform=transform)

# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# # 🔁 Load pretrained ResNet18
# # model = models.resnet18(pretrained=True)
# model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
# # Replace final fully connected layer
# model.fc = nn.Linear(model.fc.in_features, 10)
# model = model.to(device)

# # Loss & optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0005)

# # Training loop
# num_epochs = 2
# train_losses, train_accuracies = [], []

# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0
#     correct = 0
#     total = 0

#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)

#         outputs = model(images)
#         loss = criterion(outputs, labels)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

#     avg_loss = running_loss / len(train_loader)
#     accuracy = 100 * correct / total

#     train_losses.append(avg_loss)
#     train_accuracies.append(accuracy)

#     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

# # Plot Loss & Accuracy
# plt.figure(figsize=(10, 4))
# plt.subplot(1, 2, 1)
# plt.plot(train_losses, 'r-', label='Loss')
# plt.title('Training Loss (Transfer Learning)')
# plt.grid()

# plt.subplot(1, 2, 2)
# plt.plot(train_accuracies, 'g-', label='Accuracy')
# plt.title('Training Accuracy (Transfer Learning)')
# plt.grid()

# plt.tight_layout()
# plt.show()

# # Final test accuracy
# model.eval()
# correct = 0
# total = 0
# with torch.no_grad():
#     for images, labels in test_loader:
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print(f"\n✅ Final Test Accuracy (Transfer Learning): {100 * correct / total:.2f}%")






import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Transforms (no need to resize or convert to RGB)
transform = transforms.ToTensor()

# MNIST Dataset
train_dataset = datasets.MNIST(root='.', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='.', train=False, transform=transform)

# Dataloaders (small batch size = faster on CPU)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Lightweight CNN model
class LightCNN(nn.Module):
    def __init__(self):
        super(LightCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)     # 1x28x28 → 16x28x28
        self.pool = nn.MaxPool2d(2, 2)                  # 16x28x28 → 16x14x14
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)    # 16x14x14 → 32x14x14 → 32x7x7
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Instantiate model
model = LightCNN().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 3
train_losses, train_accuracies = [], []

for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total

    train_losses.append(avg_loss)
    train_accuracies.append(accuracy)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

# Accuracy & Loss plots
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, 'r-')
plt.title("Training Loss")
plt.savefig("trainingLoss.png")
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, 'g-')
plt.title("Training Accuracy")
plt.grid()
plt.tight_layout()
plt.savefig("trainingAccurancy.png")
plt.show()

# Final evaluation
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"\n Final Test Accuracy: {100 * correct / total:.2f}%")

# Using device: cpu
# Epoch 1/3, Loss: 0.1852, Accuracy: 94.31%
# Epoch 2/3, Loss: 0.0586, Accuracy: 98.20%
# Epoch 3/3, Loss: 0.0396, Accuracy: 98.76%

# Final Test Accuracy: 98.67%