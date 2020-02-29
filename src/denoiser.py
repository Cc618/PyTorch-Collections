# GAN which generates handwritten digits

import os
import numpy as np
import matplotlib.pyplot as plt
import torch as T
import torch.distributions as D
import torch.nn.functional as F
from torch import nn
from torch.nn import ReLU, Linear, Sequential, Sigmoid
from torchvision.datasets import MNIST
from torchvision import transforms
import torchvision
from utils import models_dir, dataset_dir, bw2img, stack_show


# The values are in [0, 1]
def add_noise(batch, strength=.4):
    global device

    return T.clamp(batch + D.Normal(0, strength).sample(batch.size()).to(device), min=0, max=1)


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.brain = Sequential(
            Linear(28 * 28, 256),
            ReLU(),
            Linear(256, 28 * 28),
            Sigmoid()
        )

    def forward(self, x):
        return self.brain(x.view(-1, 28 * 28)).view(-1, 1, 28, 28)


# Hyper params
epochs = 3
batch_size = 10
learning_rate = .0002
path = models_dir + '/denoiser'

# Training device
device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

# Dataset
trans = transforms.ToTensor()
dataset = MNIST(root=dataset_dir, train=True, download=True, transform=trans)
loader = T.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# Model
net = Net()
net.to(device)

# Load
if os.path.exists(path):
    net.load_state_dict(T.load(path))
    print('Model loaded')

# Train
optim = T.optim.Adam(net.parameters(), lr=learning_rate, betas=(.9, .999))
criterion = nn.MSELoss()
for e in range(epochs):
    avg_loss = 0
    for i, data in enumerate(loader, 0):
        # Only inputs (no labels)
        inputs, _ = data

        # Zero the parameter gradients
        optim.zero_grad()

        # Predictions
        denoised = inputs.to(device)
        noised = add_noise(denoised)
        y = net(noised)

        # Back prop
        loss = criterion(y, denoised)
        loss.backward()
        optim.step()

        avg_loss += loss.item()

        # Stats
        print_freq = 100
        if i % print_freq == print_freq - 1:
            print(f'Epoch {e + 1:2d}, Batch {i + 1:5d}, Loss {avg_loss / print_freq:.3f}')
            
            avg_loss = 0.0

    # Save
    T.save(net.state_dict(), path)

print('Model trained and saved')

denoised, _ = iter(loader).next()
denoised = denoised.to(device)

noised = add_noise(denoised)
stack_show([denoised, noised, net(noised)], ['Ground Truth', 'Input', 'Output'], bw=True)
