import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from functools import partial
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map

# Define the model
class Net(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape, -1))  # Flatten
        x = nn.Dense(features=512)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        x = nn.log_softmax(x)
        return x

# Data loading
transform = transforms.ToTensor()
train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000)

def numpy_collate_fn(batch):
    transposed_batch = list(zip(*batch))
    data = np.stack(transposed_batch).astype(np.float32) #Important!
    target = np.stack(transposed_batch).astype(np.int32) #Important!
    return data, target

train_loader.collate_fn = numpy_collate_fn
test_loader.collate_fn = numpy_collate_fn

# Mesh definition
mesh = Mesh(jax.devices(), ('batch',))

# Initialize parameters (sharded)
key = jax.random.PRNGKey(0)
model = Net()

@partial(shard_map, mesh=mesh, in_specs=P(), out_specs=P('batch'))
def init_params(key):
    return model.init(key, jnp.ones((1, 28, 28)))['params']

params = init_params(key)

# Create a sharded train state
optimizer = optax.adam(learning_rate=0.001)

@partial(shard_map, mesh=mesh, in_specs=P(), out_specs=P('batch'))
def create_sharded_state(params):
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

state = create_sharded_state(params)

# Loss function (sharded)
@partial(shard_map, mesh=mesh, in_specs=(P('batch'), P('batch')), out_specs=None)
def loss_fn(params, x, y):
    logits = model.apply({'params': params}, x)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=y).mean()
    return jax.lax.pmean(loss, 'batch')

# Training step (sharded)
@partial(shard_map, mesh=mesh, in_specs=(P('batch'), P('batch')), out_specs=None)
@jax.jit
def train_step(state, x, y):
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params, x, y)
    state = state.apply_gradients(grads=grads)
    return state, loss

# Testing step (sharded, gathers results)
@partial(shard_map, mesh=mesh, in_specs=(P('batch'), P('batch')), out_specs=P('batch'))
@jax.jit
def test_step(state, x, y):
    logits = state.apply_fn({'params': state.params}, x)
    predictions = jnp.argmax(logits, axis=-1)
    correct = jnp.sum(predictions == y)
    return correct

# Training loop
epochs = 3
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = jnp.array(data)
        target = jnp.array(target)

        data = data.reshape(len(jax.devices()), -1, *data.shape[1:])
        target = target.reshape(len(jax.devices()), -1)

        state, loss = train_step(state, data, target)
        loss = jax.lax.pmean(loss, 'batch')

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss:.6f}')

# Testing loop
correct_total = 0
total = 0
for data, target in test_loader:
    data = jnp.array(data)
    target = jnp.array(target)

    data = data.reshape(len(jax.devices()), -1, *data.shape[1:])
    target = target.reshape(len(jax.devices()), -1)

    correct = test_step(state, data, target)
    correct_total += correct.sum()
    total += target.shape * len(jax.devices()) # Corrected total calculation

accuracy = correct_total / total
print(f'Accuracy of the model on the 10000 test images: {100 * accuracy:.2f} %')