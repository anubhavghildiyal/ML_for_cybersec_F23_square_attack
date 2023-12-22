import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from torch.nn import Dropout

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transform with data augmentation for training
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

# Define transform for validation and test
transform_val_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load Medical MNIST dataset
#dataset = ImageFolder(root="medical_mnist", transform=transform_train)
dataset = ImageFolder(root="square_attack_yogi/medical_mnist", transform=transform_train)

# Split dataset into train, validation, and test sets
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

# DataLoader for train, validation, and test sets
train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4)

# Load pre-trained ResNet-50
resnet50 = models.resnet50(pretrained=True)
resnet50 = resnet50.to(device)

# Freeze pre-trained layers
for param in resnet50.parameters():
    param.requires_grad = False

# Modify top layers for Medical MNIST
num_ftrs = resnet50.fc.in_features
#resnet50.fc = nn.Sequential(
#    nn.Linear(num_ftrs, 256),
#    nn.ReLU(),
#    nn.Dropout(0.5),  # Add dropout to prevent overfitting
#    nn.Linear(256, 128),
#    nn.ReLU(),
#    nn.Dropout(0.5),  # Add dropout to prevent overfitting
#    nn.Linear(128, len(dataset.classes))  # Adjust output size based on the number of classes in Medical MNIST
#).to(device)


resnet50.fc = nn.Linear(num_ftrs, len(dataset.classes)).to(device)
# Define loss function, optimizer, and learning rate scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet50.fc.parameters(), lr=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # Adjust as needed

# Training loop with validation and early stopping
best_val_accuracy = 0.0
early_stopping_counter = 0
patience = 3  # Number of epochs to wait for improvement before stopping
num_epochs = 20  # Adjust as needed
for epoch in range(num_epochs):
    resnet50.train()
    total_train, correct_train = 0, 0
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = resnet50(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

        progress_bar.set_postfix({'Loss': loss.item(), 'Training Accuracy': correct_train / total_train})

    # Validation
    resnet50.eval()
    total_val, correct_val = 0, 0
    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
            val_outputs = resnet50(val_inputs)
            _, predicted = torch.max(val_outputs.data, 1)
            total_val += val_labels.size(0)
            correct_val += (predicted == val_labels).sum().item()

    val_accuracy = correct_val / total_val
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Save the model with the best validation accuracy
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(resnet50.state_dict(), "resnet50_medicalMNIST_best.pth")
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1

    # Update learning rate
    scheduler.step()

    # Check for early stopping
    if early_stopping_counter >= patience:
        print("Early stopping. No improvement in validation accuracy.")
        break

# Load the best model
resnet50.load_state_dict(torch.load("resnet50_medicalMNIST_best.pth"))

# Test the model
resnet50.eval()
total_test, correct_test = 0, 0
with torch.no_grad():
    for test_inputs, test_labels in test_loader:
        test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
        test_outputs = resnet50(test_inputs)
        _, predicted_test = torch.max(test_outputs.data, 1)
        total_test += test_labels.size(0)
        correct_test += (predicted_test == test_labels).sum().item()

test_accuracy = correct_test / total_test
print(f"Test Accuracy: {test_accuracy:.4f}")

# Save the final trained model
torch.save(resnet50.state_dict(), "resnet50_medicalMNIST_final.pth")

