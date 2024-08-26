'''
Install and Import Required Libraries
pip install torch torchvision
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

'''## Part One ##'''
'''Load the Fashion MNIST Dataset'''
# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load training and test datasets
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

'''Define the CNN Model'''
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

'''Verify Model and Dataset Loading'''
# Print model architecture
print(model)

# Check the size of the first batch
dataiter = iter(trainloader)
images, labels = next(dataiter)
print(f"Batch size: {images.size()}")


'''## Part 2: First Training Session ##'''
'''Set the Initial Hyperparameters'''
# Hyperparameters
num_epochs = 3
batch_size = 64
learning_rate = 0.01

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

'''Training Loop'''
# Training the model
for epoch in range(num_epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # Get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 100 == 99:  # print every 100 mini-batches
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
            running_loss = 0.0

print("Finished Training")

'''Save the Trained Model'''
# Save the trained model
PATH = './fashion_mnist_cnn.pt'
torch.save(model.state_dict(), PATH)

'''Evaluate the Model on the Test Set'''
# Load the saved model (optional, for verification)
model.load_state_dict(torch.load(PATH))

# Switch model to evaluation mode
model.eval()

# Initialize counters
correct = 0
total = 0

# No gradient needed for evaluation
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Calculate accuracy
accuracy = 100 * correct / total
print(f"Accuracy of the network on the 10,000 test images: {accuracy:.2f}%")


'''## Part 3: Exploratory Analysis and Conclusion Drawing ##'''
'''Exploratory Analysis'''
# Define hyperparameter sets to explore
filter_options = [8, 16, 32]
batch_size_options = [32, 64, 128]
num_epochs = 3
learning_rate = 0.01

# Function to create and train the model with different hyperparameters
def train_with_hyperparams(num_filters, batch_size):
    # Redefine the model with the new number of filters
    class FashionMNIST_CNN(nn.Module):
        def __init__(self):
            super(FashionMNIST_CNN, self).__init__()
            self.conv1 = nn.Conv2d(1, num_filters, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(num_filters, num_filters * 2, kernel_size=5)
            self.fc1 = nn.Linear(num_filters * 2 * 4 * 4, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = x.view(-1, num_filters * 2 * 4 * 4)
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    model = FashionMNIST_CNN()

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Data loader with the new batch size
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    
    # Training loop (same as before)
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Finished Epoch {epoch + 1} with num_filters={num_filters}, batch_size={batch_size}")

    # Save the model
    PATH = f'./fashion_mnist_cnn_filters_{num_filters}_batch_{batch_size}.pt'
    torch.save(model.state_dict(), PATH)

    # Evaluate and return accuracy
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
    return accuracy, PATH

# Dictionary to store results
results = {}

# Perform experiments
for num_filters in filter_options:
    for batch_size in batch_size_options:
        accuracy, model_path = train_with_hyperparams(num_filters, batch_size)
        results[(num_filters, batch_size)] = {'accuracy': accuracy, 'model_path': model_path}
        print(f"Configuration: filters={num_filters}, batch_size={batch_size}, Accuracy: {accuracy:.2f}%")

'''Conclusion Drawing'''
import matplotlib.pyplot as plt

# Extract data for plotting
filter_values = [key[0] for key in results.keys()]
batch_values = [key[1] for key in results.keys()]
accuracies = [results[key]['accuracy'] for key in results.keys()]

# Create the plots
plt.figure(figsize=(10, 5))

# Plot 1: Accuracy vs Number of Filters
plt.subplot(1, 2, 1)
plt.scatter(filter_values, accuracies, c='blue')
plt.title('Accuracy vs Number of Filters')
plt.xlabel('Number of Filters')
plt.ylabel('Accuracy (%)')

# Plot 2: Accuracy vs Batch Size
plt.subplot(1, 2, 2)
plt.scatter(batch_values, accuracies, c='red')
plt.title('Accuracy vs Batch Size')
plt.xlabel('Batch Size')
plt.ylabel('Accuracy (%)')

# Adjust layout and save the plots as an image
plt.tight_layout()

# Save the entire figure
plt.savefig('accuracy_vs_filters_and_batch_size.png')

# Display the plots
plt.show()