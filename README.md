# Summary
<p>This project explores the impact of varying convolutional neural network (CNN) hyperparameters on the classification accuracy of the Fashion MNIST dataset. By experimenting with different filter sizes and batch sizes, the project aims to identify optimal configurations. The results demonstrate that increasing the number of filters significantly improves accuracy, while variations in batch size have a more nuanced impact.</p>

# Objective
<p>The primary objective of this project is to investigate how changes in CNN hyperparameters, specifically the number of filters and batch size, affect model performance on the Fashion MNIST dataset.</p>

# Process
<p>The project began by loading the Fashion MNIST dataset and constructing a CNN model in PyTorch. Various hyperparameter configurations were tested, including filter sizes of 8, 16, and 32, and batch sizes of 32, 64, and 128. Each configuration was trained for three epochs, and the model's accuracy on the test set was recorded. Below is the code snippet showing the model architecture:</p>

```Python
class FashionMNIST_CNN(nn.Module):
    def __init__(self):
        super(FashionMNIST_CNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
        # ReLU layers
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the model
model = FashionMNIST_CNN()
```

# Results
<p>The experiments revealed that increasing the number of filters generally led to an improvement in model accuracy, although diminishing returns were observed beyond a certain point. The impact of batch size was more nuanced, with smaller batch sizes leading to slightly higher accuracy but at the cost of longer training times. The results of the experiment are presented in the tables below:</p>

![accuracy_vs_filters_and_batch_size](https://github.com/user-attachments/assets/9ccdfc22-8264-44bd-8c2c-37d3be29ba24)

# Conclusion
<p>The findings suggest that the number of filters in a CNN is a critical factor in improving classification accuracy for the Fashion MNIST dataset. Future work could explore additional hyperparameters or larger filter sizes to further optimize performance.</p>

# Further Steps
<p>Future investigations could include experimenting with learning rates, exploring the impact of additional convolutional layers, or implementing regularization techniques to prevent overfitting.</p>