'''
Convolutional Neural Network (CNN)
Build Your CNN Model Structure
=====================================
This assignment does not require any training actions to be taken. 
I just am looking for the base model that we've seen in multiple examples in class.  

This should be a python class that inherits from the nn.Module object. 
Within the object, create the __init__(self) and forward(self, x) functions and flesh them out with the model structure. 

The model structure should involve a 2 convolutional layers and 2 max pooling layers (add 2 ReLU layers in between for extra points). 
After the 2 sets of convolutional and max layers, add two layers of fully connected layers (Linear). 

Remember that for the problem set we are focusing on (FashionMNIST), 
the input image is 28x28 pixels big and the model is a classification model that categorizes the images into 10 different groupings.

Feel free to use LLMs to help you, but prompt engineer with your knowledge gained from class. 
Add helpful context like the python library we are using to help us, the dataset we are using for our model, 
and the structure of the specific CNN model we are building. 
Also refer to the notebooks I have provided in the files section on Teams for some helpful insight!
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