import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 10)
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.log_softmax(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the data with much larger batches
train_loader = DataLoader(
    datasets.MNIST("data", train=True, download=True, transform=transforms.ToTensor()),
    generator=torch.Generator(),
    batch_size=2000,  # Massively increased from 128
    shuffle=True,
    pin_memory=True  # Added for faster GPU transfer
)

test_loader = DataLoader(
    datasets.MNIST("data", train=False, transform=transforms.ToTensor()),
    generator=torch.Generator(),
    batch_size=10000,  # Increased test batch size too
)

model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.NLLLoss().to(device)

# Train the model
epochs = 100
v = list(enumerate(train_loader))
for epoch in range(epochs):
    for batch_idx, (data, target) in v:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:  # Changed from 100 since we have fewer batches now
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )

# Test the model
model.eval()
with torch.no_grad():
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()

print("Accuracy: {:.2f}%".format(100 * correct / len(test_loader.dataset)))