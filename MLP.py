import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import numpy as np
import torch.utils.data as td
import random,time
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


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

batch_size = 64
test_batch_size = 64
input_size = 3072

train_loader, _ = cifar_loaders(batch_size)
_, test_loader = cifar_loaders(test_batch_size)

import torch.nn as nn
import torch.nn.functional as F

class SevenLayerFC_Net(nn.Module):
    def __init__(self, D_in,H,D_out):
        """
        In the constructor we instantiate three nn.Linear modules and assign them as
        member variables.
        """
        super(SevenLayerFC_Net, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear4 = torch.nn.Linear(H, H)
        self.linear5 = torch.nn.Linear(H, H)
        self.linear6 = torch.nn.Linear(H, H)
        self.linear7 = torch.nn.Linear(H, D_out)

        
    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = F.relu(self.linear5(x))
        x = F.relu(self.linear6(x))
        x = self.linear7(x)
        return F.log_softmax(x)  
      
      # N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = batch_size, input_size, 200, 10
num_epochs = 10

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Construct our model by instantiating the class defined above
model = SevenLayerFC_Net(D_in, H, D_out)
#print(model)

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

for epoch in range(num_epochs):
    avg_loss_epoch = 0
    batch_loss = 0
    total_batches = 0

    for i, (images, labels) in enumerate(train_loader):
        # Reshape images to (batch_size, input_size)

        images = images.reshape(-1, 32*32*3)            

        #print(images.shape)
        outputs = model(images)

        loss = criterion(outputs, labels)    
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()   

        total_batches += 1     
        batch_loss += loss.item()

    avg_loss_epoch = batch_loss/total_batches
    print ('Epoch [{}/{}], Averge Loss:for epoch[{}, {:.4f}]' 
                   .format(epoch+1, num_epochs, epoch+1, avg_loss_epoch ))

# Test the Model
correct = 0
total = 0
predicted_labels = []
true_labels = []

for images, labels in test_loader:
    images = images.reshape(-1, 3*32*32)
    #print(labels)
    outputs_test = model(images)
    _, predicted = torch.max(outputs_test.data, 1)
    #print(predicted)
    predicted_labels.extend(predicted.tolist())
    true_labels.extend(labels.tolist())
    
    total += labels.size(0) 
    correct += (predicted == labels).sum().item()

# Compute precision, recall
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, cohen_kappa_score, confusion_matrix
from torchvision import transforms

# Compute accuracy
accuracy = 100 * correct / total

# Precision
precision = precision_score(true_labels, predicted_labels, average='macro')

# Recall
recall = recall_score(true_labels, predicted_labels, average='macro')

# F1 Score
f1 = f1_score(true_labels, predicted_labels, average='macro')

# Matthew's Correlation Coefficient
mcc = matthews_corrcoef(true_labels, predicted_labels)

# Cohen's Kappa Score
kappa = cohen_kappa_score(true_labels, predicted_labels)

# Calculate confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)



print("************Overall Framework************")
print("Framework:", torch.__name__)
print("************Training********************")    
print('Accuracy of the network on the 10000 test images: %d %%' % (     100 * correct / total))
print("Number of epochs:", num_epochs)
print("Batch size:", batch_size)
print("Learning rate:", optimizer.param_groups[0]['lr'])
print("************Network*********************")
print("Input layer size:", D_in)
print("Hidden layer size:", H)
print("Output layer size:", D_out)
print("***********Preformace Metrics************")
print("Precision:", precision)
print("Recall:", recall)
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Matthew's Correlation Coefficient:", mcc)
print("Cohen's Kappa Score:", kappa)
print("***********Confusion Metrics************")
# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.show()

