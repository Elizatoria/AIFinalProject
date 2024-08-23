import torch
import torch.nn as nn
import torch.nn.functional as F

class FashionMNIST_CNN(nn.Module):
    def __init__(self):
        super(FashionMNIST_CNN, self).__init__()
        
        # First convolutional layer: input channels = 1 (grayscale), output channels = 32, kernel size = 3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional layer: input channels = 32, output channels = 64, kernel size = 3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # After two 2x2 pooling layers, the 28x28 image is reduced to 7x7
        self.fc2 = nn.Linear(128, 10)  # 10 output classes (for FashionMNIST)
    
    def forward(self, x):
        # Apply first convolutional layer followed by ReLU and max pooling
        x = self.pool1(self.relu1(self.conv1(x)))
        
        # Apply second convolutional layer followed by ReLU and max pooling
        x = self.pool2(self.relu2(self.conv2(x)))
        
        # Flatten the output from the conv layers to feed into fully connected layers
        x = x.view(-1, 64 * 7 * 7)
        
        # Apply first fully connected layer followed by ReLU
        x = F.relu(self.fc1(x))
        
        # Apply second fully connected layer (no activation function because this is for classification)
        x = self.fc2(x)
        
        return x

# Example usage (not required for the assignment)
# model = FashionMNIST_CNN()
# print(model)

import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Initialize hyperparameters
batch_size = 64
epochs = 3
learning_rate = 0.01

# Define transformations for the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the FashionMNIST dataset
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

# Initialize the model, loss function, and optimizer
model = FashionMNIST_CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Print statistics
        running_loss += loss.item()
        if i % 100 == 99:  # Print every 100 mini-batches
            print(f'Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(trainloader)}], Loss: {running_loss / 100:.3f}')
            running_loss = 0.0

print('Finished Training')

# Save the trained model
model_path = 'fashion_mnist_cnn.pt'
torch.save(model.state_dict(), model_path)
print(f'Model saved to {model_path}')

# Evaluate the model on the test set
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy of the network on the 10,000 test images: {accuracy:.2f}%')

'''
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to ./data\FashionMNIST\raw\train-images-idx3-ubyte.gz
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 26421880/26421880 [03:37<00:00, 121738.52it/s]
Extracting ./data\FashionMNIST\raw\train-images-idx3-ubyte.gz to ./data\FashionMNIST\raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to ./data\FashionMNIST\raw\train-labels-idx1-ubyte.gz
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 29515/29515 [00:00<00:00, 112185.68it/s]
Extracting ./data\FashionMNIST\raw\train-labels-idx1-ubyte.gz to ./data\FashionMNIST\raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to ./data\FashionMNIST\raw\t10k-images-idx3-ubyte.gz
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 4422102/4422102 [00:46<00:00, 96131.37it/s]
Extracting ./data\FashionMNIST\raw\t10k-images-idx3-ubyte.gz to ./data\FashionMNIST\raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to ./data\FashionMNIST\raw\t10k-labels-idx1-ubyte.gz
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5148/5148 [00:00<?, ?it/s]
Extracting ./data\FashionMNIST\raw\t10k-labels-idx1-ubyte.gz to ./data\FashionMNIST\raw

Epoch [1/3], Batch [100/938], Loss: 0.874
Epoch [1/3], Batch [200/938], Loss: 0.459
Epoch [1/3], Batch [300/938], Loss: 0.420
Epoch [1/3], Batch [400/938], Loss: 0.376
Epoch [1/3], Batch [500/938], Loss: 0.337
Epoch [1/3], Batch [600/938], Loss: 0.358
Epoch [1/3], Batch [700/938], Loss: 0.349
Epoch [1/3], Batch [800/938], Loss: 0.353
Epoch [1/3], Batch [900/938], Loss: 0.333
Epoch [2/3], Batch [100/938], Loss: 0.315
Epoch [2/3], Batch [200/938], Loss: 0.303
Epoch [2/3], Batch [300/938], Loss: 0.315
Epoch [2/3], Batch [400/938], Loss: 0.304
Epoch [2/3], Batch [500/938], Loss: 0.297
Epoch [2/3], Batch [600/938], Loss: 0.305
Epoch [2/3], Batch [700/938], Loss: 0.302
Epoch [2/3], Batch [800/938], Loss: 0.309
Epoch [2/3], Batch [900/938], Loss: 0.306
Epoch [3/3], Batch [100/938], Loss: 0.258
Epoch [3/3], Batch [200/938], Loss: 0.284
Epoch [3/3], Batch [300/938], Loss: 0.278
Epoch [3/3], Batch [400/938], Loss: 0.283
Epoch [3/3], Batch [500/938], Loss: 0.304
Epoch [3/3], Batch [600/938], Loss: 0.287
Epoch [3/3], Batch [700/938], Loss: 0.280
Epoch [3/3], Batch [800/938], Loss: 0.282
Epoch [3/3], Batch [900/938], Loss: 0.277
Finished Training
Model saved to fashion_mnist_cnn.pt
Accuracy of the network on the 10,000 test images: 88.89%
'''