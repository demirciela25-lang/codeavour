import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# ========================
# Device Configuration
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ========================
# Dataset Paths
# ========================
# base_path = "domates\\tomato_dataset\\tomato_dataset_in"
# train_dir = "domates//tomato_dataset//tomato_dataset_in//train_small"
# val_dir = "domates//tomato_dataset//tomato_dataset_in//valid_small"

# Get the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.join(script_dir, "codeavour","domates", "tomato_dataset", "tomato_dataset_in")

train_dir = os.path.join(base_path, "train_small")
val_dir = os.path.join(base_path, "valid_small")

# ========================
# Check Dataset Existence
# ========================
if not os.path.exists(base_path):
    raise FileNotFoundError(
        f"Dataset directory not found: {base_path}\n"
        f"Current working directory: {os.getcwd()}\n"
        f"Please ensure the dataset is extracted to the correct location."
    )

if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Training directory not found: {train_dir}")

if not os.path.exists(val_dir):
    raise FileNotFoundError(f"Validation directory not found: {val_dir}")

# ========================
# Transforms
# ========================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ========================
# Datasets
# ========================
train_data = datasets.ImageFolder(train_dir, transform=train_transform)
val_data = datasets.ImageFolder(val_dir, transform=val_transform)

print("Classes:", train_data.classes)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)

# ========================
# Model Setup (Transfer Learning)
# ========================
model = models.mobilenet_v2(pretrained=True)

# Freeze feature extractor
for param in model.features.parameters():
    param.requires_grad = False

# Replace classifier
model.classifier[1] = nn.Linear(model.last_channel, len(train_data.classes))

model = model.to(device)

# ========================
# Loss & Optimizer
# ========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

# ========================
# Training Loop
# ========================
num_epochs = 10

# Storage for results
epoch_losses = []
epoch_train_accuracies = []
epoch_val_accuracies = []
epoch_model_states = []

for epoch in range(num_epochs):
    print(f"\n{'='*60}")
    print(f"Starting Epoch [{epoch+1}/{num_epochs}]")
    print(f"{'='*60}")

    # ---- TRAINING ----
    print("Training phase...")
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    total_batches = len(train_loader)
    for batch_idx, (images, labels) in enumerate(train_loader, 1):
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

        if batch_idx % 10 == 0 or batch_idx == total_batches:
            print(f"  Batch [{batch_idx}/{total_batches}] - Loss: {loss.item():.4f}")

    train_accuracy = 100 * correct / total
    print(f"Training complete - Accuracy: {train_accuracy:.2f}%")

    # ---- VALIDATION ----
    print("Validation phase...")
    model.eval()
    val_correct = 0
    val_total = 0

    total_val_batches = len(val_loader)
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader, 1):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

            if batch_idx % 10 == 0 or batch_idx == total_val_batches:
                print(f"  Batch [{batch_idx}/{total_val_batches}]")

    val_accuracy = 100 * val_correct / val_total
    print(f"Validation complete - Accuracy: {val_accuracy:.2f}%")

    # Store results
    print(f"Saving epoch {epoch+1} results...")
    epoch_losses.append(running_loss)
    epoch_train_accuracies.append(train_accuracy)
    epoch_val_accuracies.append(val_accuracy)
    epoch_model_states.append(model.state_dict().copy())

# ========================
# Print All Results
# ========================
print("\n" + "="*60)
print("TRAINING COMPLETE - ALL EPOCH RESULTS:")
print("="*60)
for i in range(num_epochs):
    print(f"Epoch [{i+1}/{num_epochs}] "
          f"Loss: {epoch_losses[i]:.4f} "
          f"Train Acc: {epoch_train_accuracies[i]:.2f}% "
          f"Val Acc: {epoch_val_accuracies[i]:.2f}%")

# ========================
# Select and Save Best Model
# ========================
best_epoch = epoch_val_accuracies.index(max(epoch_val_accuracies))
best_val_accuracy = epoch_val_accuracies[best_epoch]

print("\n" + "="*60)
print(f"BEST MODEL: Epoch {best_epoch+1}")
print(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")
print(f"Best Train Accuracy: {epoch_train_accuracies[best_epoch]:.2f}%")
print(f"Best Loss: {epoch_losses[best_epoch]:.4f}")
print("="*60)

torch.save(epoch_model_states[best_epoch], "best_tomato_model.pth")
print("Best model saved to 'best_tomato_model.pth'!")
