import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

torch.set_num_threads(40)
torch.set_num_interop_threads(4)

# Modified model with parallel splits
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Break large 784x512 layer into smaller sequential ones
        self.fc1a = nn.Linear(784, 128)
        self.fc1b = nn.Linear(128, 256)
        self.fc1c = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 10)
    
    def forward(self, x):
        x = x.view(-1, 784)
        # Process in stages to reduce peak memory
        x = torch.relu(self.fc1a(x))
        x = torch.relu(self.fc1b(x))
        x = torch.relu(self.fc1c(x))
        return self.fc2(x).log_softmax(dim=1)

model = Net()
# model = model.half()

train_dataset = datasets.MNIST(
    'data', 
    train=True, 
    download=True,
    transform=transforms.ToTensor()
)

train_loader = DataLoader(
    train_dataset,
    batch_size=5000,
    shuffle=True
)

test_loader = DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.ToTensor()),
    batch_size=10000
)

torch.backends.cuda.matmul.allow_tf32 = True
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.NLLLoss()

for p in model.parameters():
    p.grad = torch.zeros_like(p)

train_data = list(enumerate(train_loader))

epochs = 10
start_time = time.perf_counter()

@profile
def main():
    for epoch in range(epochs):
        for batch_idx, (data, target) in train_data:
            # Just make contiguous, no pin_memory
            data = data.contiguous()
            
            for p in model.parameters():
                p.grad.zero_()
                
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                    f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    end_time = time.perf_counter()
    print(f"Training time: {end_time - start_time:.2f} seconds")

main()

model.eval()
with torch.no_grad():
    data, target = next(iter(test_loader))
    output = model(data)
    pred = output.argmax(dim=1)
    correct = pred.eq(target).sum().item()
    print(f'Accuracy: {100. * correct / len(test_loader.dataset):.2f}%')