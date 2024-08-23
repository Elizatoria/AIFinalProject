'''
Part 1 Loading the dataset and model - Due on 8/20/2024:
For this part, you will have loaded the Fashion MNIST dataset and a CNN model using
Python and the PyTorch library. This is the first checkpoint to make sure the training sessions
will be performed properly. For the convolutional layers, kernel size should be set to 5, the
number of filters should be set to 8 initially, and padding and stride should be set to their
default values.
'''
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