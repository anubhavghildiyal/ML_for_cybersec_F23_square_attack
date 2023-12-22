import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transform for validation and test
transform_val_test = transforms.Compose([
    transforms.Resize((224, 224)),  # VGG takes 224x224 images
    transforms.ToTensor(),
])

# Load Medical MNIST dataset
dataset = ImageFolder(root="medical_mnist", transform=transform_val_test)

# Split dataset into train, validation, and test sets
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

# DataLoader for train, validation, and test sets
train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4)

# Load pre-trained VGG model
vgg = models.vgg16(pretrained=True)
vgg.classifier[6] = nn.Linear(4096, len(dataset.classes)).to(device)  # Modify last fully connected layer

# Define loss function, optimizer, and learning rate scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(vgg.parameters(), lr=0.001, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # Adjust as needed

# Training loop with validation and early stopping
best_val_accuracy = 0.0
early_stopping_counter = 0
patience = 3  # Number of epochs to wait for improvement before stopping
num_epochs = 20  # Adjust as needed

for epoch in range(num_epochs):
    # Training phase
    vgg.train()
    vgg = vgg.to(device)
    total_train, correct_train = 0, 0
    
    for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = vgg(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    # Calculate training accuracy
    train_accuracy = correct_train / total_train

    # Validation phase
    vgg.eval()
    total_val, correct_val = 0, 0

    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
            val_outputs = vgg(val_inputs)
            _, predicted = torch.max(val_outputs.data, 1)
            total_val += val_labels.size(0)
            correct_val += (predicted == val_labels).sum().item()

    # Calculate validation accuracy
    val_accuracy = correct_val / total_val

    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Save the model with the best validation accuracy
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(vgg.state_dict(), "vgg_medicalMNIST_final.pth")
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1

    # Update learning rate
    scheduler.step()

    # Check for early stopping
    if early_stopping_counter >= patience:
        print("Early stopping. No improvement in validation accuracy.")
        break

# Save the final trained model
torch.save(vgg.state_dict(), "vgg_medicalMNIST_final.pth")

# Load the best model
vgg.load_state_dict(torch.load("vgg_medicalMNIST_final.pth"))

# Test the model
vgg.eval()
total_test, correct_test = 0, 0

with torch.no_grad():
    for test_inputs, test_labels in test_loader:
        test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
        test_outputs = vgg(test_inputs)
        _, predicted_test = torch.max(test_outputs.data, 1)
        total_test += test_labels.size(0)
        correct_test += (predicted_test == test_labels).sum().item()

test_accuracy = correct_test / total_test
print(f"Test Accuracy: {test_accuracy:.4f}")


