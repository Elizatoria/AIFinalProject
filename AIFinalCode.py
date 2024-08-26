'''
Install and Import Required Libraries
pip install torch torchvision
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

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

'''
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to ./data\FashionMNIST\raw\train-images-idx3-ubyte.gz
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 26421880/26421880 [01:09<00:00, 381948.97it/s]
Extracting ./data\FashionMNIST\raw\train-images-idx3-ubyte.gz to ./data\FashionMNIST\raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to ./data\FashionMNIST\raw\train-labels-idx1-ubyte.gz
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 29515/29515 [00:00<00:00, 235693.38it/s]
Extracting ./data\FashionMNIST\raw\train-labels-idx1-ubyte.gz to ./data\FashionMNIST\raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to ./data\FashionMNIST\raw\t10k-images-idx3-ubyte.gz
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 4422102/4422102 [00:03<00:00, 1242829.04it/s]
Extracting ./data\FashionMNIST\raw\t10k-images-idx3-ubyte.gz to ./data\FashionMNIST\raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to ./data\FashionMNIST\raw\t10k-labels-idx1-ubyte.gz
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5148/5148 [00:00<?, ?it/s]
Extracting ./data\FashionMNIST\raw\t10k-labels-idx1-ubyte.gz to ./data\FashionMNIST\raw

FashionMNIST_CNN(
  (conv1): Conv2d(1, 8, kernel_size=(5, 5), stride=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(8, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=256, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
  (relu): ReLU()
)
Batch size: torch.Size([64, 1, 28, 28])
[1, 100] loss: 0.916
[1, 200] loss: 0.591
[1, 300] loss: 0.537
[1, 400] loss: 0.501
[1, 500] loss: 0.478
[1, 600] loss: 0.462
[1, 700] loss: 0.440
[1, 800] loss: 0.448
[1, 900] loss: 0.433
[2, 100] loss: 0.425
[2, 200] loss: 0.393
[2, 300] loss: 0.410
[2, 400] loss: 0.406
[2, 500] loss: 0.420
[2, 600] loss: 0.411
[2, 700] loss: 0.424
[2, 800] loss: 0.408
[2, 900] loss: 0.402
[3, 100] loss: 0.372
[3, 200] loss: 0.381
[3, 300] loss: 0.378
[3, 400] loss: 0.363
[3, 500] loss: 0.368
[3, 600] loss: 0.375
[3, 700] loss: 0.393
[3, 800] loss: 0.388
[3, 900] loss: 0.396
Finished Training
c:\Users\Eliza\Documents\MachineLearning\AIFinalProject\AIFinalCode.py:107: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(PATH))
Accuracy of the network on the 10,000 test images: 84.73%
'''