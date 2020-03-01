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


# Random input for the generator
def sample_latent(batch_size, latent_size, device):
    return T.distributions.Normal(0, 1).sample([batch_size, latent_size]).to(device)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        h_size = 512
        self.dense = Sequential(
            Linear(784, h_size),
            ReLU(True),

            Linear(h_size, 1),
            Sigmoid(),
        )

    def forward(self, x):
        x = self.dense(x.view(-1, 784))

        # 1 for real, 0 for fake
        return x


class Generator(nn.Module):
    def __init__(self, latent_size):
        super().__init__()

        h_size = 2048
        self.dense = Sequential(
            Linear(latent_size, h_size),
            ReLU(True),

            Linear(h_size, 784),
            Sigmoid()
        )

    def forward(self, x):
        x = self.dense(x).view(-1, 1, 28, 28)

        return x


# Hyper params
epochs = 10
batch_size = 100
# test_batch_size <= batch_size
test_batch_size = 4
learning_rate = .0001
path = models_dir + '/gan'
dis_path = path + '_dis'
gen_path = path + '_gen'
latent_size = 10

# Training device
device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

# Dataset
trans = transforms.ToTensor()
dataset = MNIST(root=dataset_dir, train=True, download=True, transform=trans)
loader = T.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# Models
discriminator = Discriminator()
discriminator.to(device)
generator = Generator(latent_size)
generator.to(device)

# Load
if os.path.exists(dis_path) and os.path.exists(gen_path):
    discriminator.load_state_dict(T.load(dis_path))
    generator.load_state_dict(T.load(gen_path))
    print('Models loaded')

# Train
d_opti = T.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(.5, .999))
g_opti = T.optim.Adam(generator.parameters(), lr=learning_rate, betas=(.5, .999))
criterion = nn.BCELoss()

# Labels
real_label = T.ones([batch_size, 1], device=device)
fake_label = T.zeros([batch_size, 1], device=device)

for e in range(epochs):
    d_avg_loss = 0
    g_avg_loss = 0
    for i, data in enumerate(loader, 0):
        # Only inputs (no labels)
        inputs, _ = data

        # Generator #
        g_opti.zero_grad()

        # Generate images from random input
        z = sample_latent(batch_size, latent_size, device)
        fake = generator(z)

        # The generator wants to generate real images
        # So the discriminator has to predict 1 for real
        gen_class = discriminator(fake)

        # Create generator loss and back prop
        gen_loss = criterion(gen_class, real_label)
        gen_loss.backward()
        g_opti.step()

        # Discriminator #
        d_opti.zero_grad()

        # Real images
        real = inputs.to(device)

        # Real discriminator loss
        real_class = discriminator(real)
        real_loss = criterion(real_class, real_label)

        # Fake discriminator loss
        fake_class = discriminator(fake.detach())
        fake_loss = criterion(fake_class, fake_label)

        # Combine losses
        dis_loss = real_loss + fake_loss
        dis_loss.backward()
        d_opti.step()

        d_avg_loss += dis_loss.item()
        g_avg_loss += gen_loss.item()

        # Stats
        print_freq = 100
        if i % print_freq == print_freq - 1:
            print(f'Epoch {e + 1:2d}, Batch {i + 1:5d}, Loss {(d_avg_loss + g_avg_loss) / print_freq:.3f} : (Dis = {d_avg_loss / print_freq:.3f}, Gen = {g_avg_loss / print_freq:.3f})')

            d_avg_loss = g_avg_loss = 0.0

    # Save
    T.save(discriminator.state_dict(), dis_path)
    T.save(generator.state_dict(), gen_path)

print('Models trained and saved')

ground_truth, _ = iter(loader).next()
ground_truth = ground_truth[:test_batch_size].to(device)

with T.no_grad():
    stack_show([ground_truth, generator(sample_latent(test_batch_size, latent_size, device))], ['Ground Truth', 'Generated'], bw=True)
