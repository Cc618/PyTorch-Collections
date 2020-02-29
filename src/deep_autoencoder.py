# Autoencoder using convolutional layers
# Dataset : MNIST
# Requires : PIL, matplotlib
# Inspired by https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# To compress data : net.encode(data)
# To decompress data : net.decode(data)
# To mutate data : net(data)

import os
import numpy as np
import matplotlib.pyplot as plt
import torch as T
from torch import nn
from torch import cuda
import torch.nn.functional as F
from torchvision import transforms
import torchvision
from torchvision.datasets import MNIST
from torch.nn import ReLU, Linear, Sigmoid, Conv2d, ConvTranspose2d, MaxPool2d
import PIL.Image as im
from utils import dataset_dir, models_dir


# Displays an image (1 dim tensor)
# t has values in [0, 1]
def imshow(t):
    transforms.ToPILImage()(t).show()


# Show in matplotlib
def gridshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Net(nn.Module):
    def __init__(self, hidden_size, latent_size):
        super().__init__()

        self.latent_size = latent_size

        self.encodeConv1 = Conv2d(1, 16, 4)
        self.encodeConv2 = Conv2d(16, 32, 2)
        self.encodeFC1 = Linear(800, hidden_size)
        self.encodeFC2 = Linear(hidden_size, self.latent_size)

        self.decodeFC1 = Linear(self.latent_size, 13 * 13)
        self.decodeConv1 = ConvTranspose2d(1, 1, 2)
        self.decodeFC2 = Linear(14 * 14, 28 * 28)

    def encode(self, x):
        x = MaxPool2d(2)(F.relu(self.encodeConv1(x)))
        x = MaxPool2d(2)(F.relu(self.encodeConv2(x)))
        x = x.view(-1, 800)
        x = F.relu(self.encodeFC1(x))
        x = T.sigmoid(self.encodeFC2(x))

        return x

    def decode(self, x):
        x = F.relu(self.decodeFC1(x))
        x = x.view(-1, 1, 13, 13)
        x = F.relu(self.decodeConv1(x))
        x = x.view(-1, 14 * 14)
        x = T.sigmoid(self.decodeFC2(x))
        x = x.view(-1, 1, 28, 28)

        return x

    def forward(self, x):
        return self.decode(self.encode(x))


# Hyper params
latent_size = 10
hidden_size = 256
epochs = 3
batch_size = 10
learning_rate = .0002
train_or_test = 'test'
path = models_dir + '/deep_autoencoder'

# Training device
device = T.device('cuda:0' if cuda.is_available() else 'cpu')

# Dataset
trans = transforms.ToTensor()
dataset = MNIST(root=dataset_dir, train=True, download=True, transform=trans)
loader = T.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# Model
net = Net(hidden_size, latent_size)
net.to(device)

if train_or_test == 'train':
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
            x = inputs.to(device)
            y = net(x)

            # Back prop
            loss = criterion(y, x)
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

else:
    # Load
    net.load_state_dict(T.load(path))

    # Test
    dataiter = iter(loader)
    images, _ = dataiter.next()

    # Show ground truth
    gridshow(torchvision.utils.make_grid(images))

    # Show predictions
    preds = T.cat([net(images[i].view(1, 1, 28, 28).to(device)).view(1, 1, 28, 28).cpu() for i in range(batch_size)])
    preds = T.tensor(preds)
    gridshow(torchvision.utils.make_grid(preds))
