import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import random

# Define labels for the CIFAR-10 classes
LABELS = {
    'plane': 0,
    'car': 1,
    'bird': 2,
    'cat': 3,
    'deer': 4,
    'dog': 5,
    'frog': 6,
    'horse': 7,
    'ship': 8,
    'truck': 9
}

class CustomCIFAR10(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Loop through each file in the directory
        for img_name in os.listdir(data_dir):
            if img_name.endswith('.png'):
                # Extract label from the filename (e.g., 'plane_123.png' → 'plane')
                label_name = img_name.split('_')[0]
                if label_name in LABELS:
                    self.image_paths.append(os.path.join(data_dir, img_name))
                    self.labels.append(LABELS[label_name])

        print(f"Total images found in dataset: {len(self.image_paths)}")  # Total number of images
        if len(self.image_paths) == 0:
            raise ValueError("The dataset is empty. Please check the directory structure and ensure there are images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Open image using PIL
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)
        
        return img, label

# Define image transformations for data augmentation and normalization
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Paths to your dataset
train_dir = "C:/Users/User/Desktop/cifar_train/cifar_train"  # Update this path
test_dir = "C:/Users/User/Desktop/cifar_test"   # Update this path

# Verify that directories exist
if not os.path.exists(train_dir):
    raise ValueError(f"Train directory does not exist: {train_dir}")
if not os.path.exists(test_dir):
    raise ValueError(f"Test directory does not exist: {test_dir}")

# Create the train and test datasets
train_dataset = CustomCIFAR10(data_dir=train_dir, transform=train_transform)
test_dataset = CustomCIFAR10(data_dir=test_dir, transform=test_transform)

# Check the number of samples in train and test datasets
print(f"Number of training images: {len(train_dataset)}")
print(f"Number of testing images: {len(test_dataset)}")

# Create the DataLoader objects
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Define a simple Vision Transformer model (you can replace this with a pre-trained model if needed)
class ViT(nn.Module):
    def __init__(self, num_classes=10):
        super(ViT, self).__init__()
        self.layer1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.layer2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 224 * 224, 512)  # Adjust based on your actual image size
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        print(x.shape)  # Add this line to check the tensor size
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Check if CUDA is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Instantiate the model, loss function, and optimizer
model = ViT(num_classes=10).to(device)  # Move model to GPU if available
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        # Move images and labels to GPU if available
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

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

# Testing the model
model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for images, labels in test_loader:
        # Move images and labels to GPU if available
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Testing with one random image
random_image_idx = random.randint(0, len(test_dataset) - 1)
img, label = test_dataset[random_image_idx]
img = img.unsqueeze(0).to(device)  # Add batch dimension and move image to GPU

# Predict label
model.eval()
with torch.no_grad():
    output = model(img)
    _, predicted_label = torch.max(output.data, 1)

# Display the result
print(f"Predicted Label: {list(LABELS.keys())[predicted_label.item()]}, Actual Label: {list(LABELS.keys())[label]}")
