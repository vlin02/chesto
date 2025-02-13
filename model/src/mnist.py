import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device('cuda')

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 10)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        return self.fc2(x).log_softmax(dim=1)

model = Net().to(device)

train_dataset = datasets.MNIST(
    'data', 
    train=True, 
    download=True,
    transform=transforms.ToTensor()
)

train_loader = DataLoader(
    train_dataset,
    batch_size=60000,
    shuffle=True,
    pin_memory=True
)

# Make optimizer capturable
optimizer = optim.Adam(model.parameters(), lr=0.001, capturable=True)
criterion = nn.NLLLoss().to(device)

data, target = next(iter(train_loader))
data, target = data.to(device), target.to(device)

s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())

static_output = None
static_loss = None

def train_step():
    global static_output, static_loss
    optimizer.zero_grad(set_to_none=True)
    static_output = model(data)
    static_loss = criterion(static_output, target)
    static_loss.backward()
    optimizer.step()

with torch.cuda.stream(s):
    train_step()
torch.cuda.current_stream().wait_stream(s)

g = torch.cuda.CUDAGraph()
with torch.cuda.stream(s):
    with torch.cuda.graph(g):
        train_step()

epochs = 1000
start_time = time.perf_counter()

def main():
    for epoch in range(epochs):
        g.replay()
        print(f'Loss: {static_loss.item():.6f}')

    end_time = time.perf_counter()
    print(f"Training time: {end_time - start_time:.2f} seconds")

main()

test_loader = DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.ToTensor()),
    batch_size=10000,
    pin_memory=True
)

model.eval()
with torch.no_grad():
    data, target = next(iter(test_loader))
    data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
    output = model(data)
    pred = output.argmax(dim=1)
    correct = pred.eq(target).sum().item()
    print(f'Accuracy: {100. * correct / len(test_loader.dataset):.2f}%')
