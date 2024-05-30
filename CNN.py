import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import numpy as np
import torch.utils.data as td
import random, time
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, cohen_kappa_score, confusion_matrix

# Number of epochs
num_epochs = 10
# Function to load CIFAR-10 data
def cifar_loaders(batch_size, shuffle_test=False): 
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.225, 0.225, 0.225])
    train = datasets.CIFAR10('./', train=True, download=True, 
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]))
    test = datasets.CIFAR10('./', train=False, 
        transform=transforms.Compose([transforms.ToTensor(), normalize]))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
        shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
        shuffle=shuffle_test, pin_memory=True)
    return train_loader, test_loader

# Batch sizes
batch_size = 64
test_batch_size = 64

# Load data
train_loader, _ = cifar_loaders(batch_size)
_, test_loader = cifar_loaders(test_batch_size)

# Define the CNN model
class Conv7Net(nn.Module):
    def __init__(self):
        super(Conv7Net, self).__init__()
        # Convolutional Layers
        self.conv1 = nn.Conv2d(3, 84, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(84, 256, kernel_size=5, stride=2, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=2, stride=1, padding=0)
        
        # Calculate the size of the input to the first fully connected layer
        self._to_linear = None
        self.convs(torch.randn(1, 3, 32, 32))
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(self._to_linear, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)
    
    def convs(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        if self._to_linear is None:
            self._to_linear = x.view(x.size(0), -1).shape[1]
        return x
    
    def forward(self, input):
        x = self.convs(input)
                
        # Flatten the tensor for the fully connected layer
        x = x.view(x.size(0), -1)  # flatten the tensor, keeping the batch size
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# Instantiate the model
net = Conv7Net()
print(net)

# Print layer information
params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight

# Set device to GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# Construct our model by instantiating the class defined above
ConvModel = Conv7Net()
ConvModel.to(device)

# Construct our loss function and an Optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(ConvModel.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0002)

# Training loop
for epoch in range(num_epochs):
    avg_loss_epoch = 0
    batch_loss = 0
    total_batches = 0

    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images).to(device)
        labels = Variable(labels).to(device)
        
        print(images.shape)
        outputs = ConvModel(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()   

        total_batches += 1     
        batch_loss += loss.item()

    avg_loss_epoch = batch_loss / total_batches
    print('Epoch [{}/{}], Average Loss: {:.4f}'.format(epoch + 1, num_epochs, avg_loss_epoch))

# Test the Model
correct = 0
total = 0
predicted_labels = []
true_labels = []

for images, labels in test_loader:
    images = Variable(images).to(device)
    labels = Variable(labels).to(device)
    print(images.shape)
    outputs_Conv_test = ConvModel(images)
    _, predicted = torch.max(outputs_Conv_test.data, 1)
    
    predicted_labels.extend(predicted.tolist())
    true_labels.extend(labels.tolist())
    
    total += labels.size(0) 
    correct += (predicted == labels).sum().item()
    
    

# Print model architecture layers using variables
input_layer = ConvModel.conv1
hidden_layers = [ConvModel.conv2, ConvModel.conv3, ConvModel.conv4, ConvModel.fc1, ConvModel.fc2]
output_layer = ConvModel.fc3



# Compute precision, recall, f1 score, mcc, kappa
accuracy = 100 * correct / total
precision = precision_score(true_labels, predicted_labels, average='macro')
recall = recall_score(true_labels, predicted_labels, average='macro')
f1 = f1_score(true_labels, predicted_labels, average='macro')
mcc = matthews_corrcoef(true_labels, predicted_labels)
kappa = cohen_kappa_score(true_labels, predicted_labels)




# Print results
print("************Overall Framework************")
print("Framework:", torch.__name__)
print("************Training********************")    
print('Accuracy of the network on the 10000 test images: %d %%' % (accuracy))
print("Number of epochs:", num_epochs)
print("Batch size:", batch_size)
print("Learning rate:", optimizer.param_groups[0]['lr'])
print("************Network*********************")
print("\nModel Layers:")
print(f"Input Layer: Convolutional Layer - {input_layer} with input channels {input_layer.in_channels} and output channels {input_layer.out_channels}")
print("Hidden Layers:")
for layer in hidden_layers:
    if isinstance(layer, nn.Conv2d):
        print(f" - Convolutional Layer - {layer} with input channels {layer.in_channels} and output channels {layer.out_channels}")
    elif isinstance(layer, nn.Linear):
        print(f" - Fully Connected Layer - {layer} with input features {layer.in_features} and output features {layer.out_features}")
print(f"Output Layer: Fully Connected Layer - {output_layer} with input features {output_layer.in_features} and output features {output_layer.out_features}")
print("***********Performance Metrics************")
print("Precision:", precision)
print("Recall:", recall)
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Matthew's Correlation Coefficient:", mcc)
print("Cohen's Kappa Score:", kappa)
# Calculate confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.show()







