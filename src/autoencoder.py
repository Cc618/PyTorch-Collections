# Autoencoder using only fully connected layers
# Dataset : CIFAR10
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
from torchvision import transforms
import torchvision
from torchvision.datasets import CIFAR10
from torch.nn import ReLU, Linear, Sigmoid
import PIL.Image as im
from globals import dataset_dir, models_dir


# Displays an image (3 dim tensor)
# t has values in [0, 1]
def imshow(t):
    transforms.ToPILImage()(t).show()


# Show in matplotlib
def gridshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Net(nn.Module):
    def __init__(self, im_width, im_height, hidden_size, latent_size):
        super().__init__()

        self.latent_size = latent_size
        self.im_width = im_width
        self.im_height = im_height

        self.encoder1 = Linear(3 * im_width * im_height, hidden_size)
        self.encoder2 = Linear(hidden_size, latent_size)
        self.decoder1 = Linear(latent_size, hidden_size)
        self.decoder2 = Linear(hidden_size, 3 * im_width * im_height)
    
    def encode(self, x):
        x = x.view([-1])
        
        encoded = ReLU()(self.encoder1(x))
        encoded = Sigmoid()(self.encoder2(encoded))

        return encoded

    def decode(self, encoded):
        decoded = ReLU()(self.decoder1(encoded))
        decoded = Sigmoid()(self.decoder2(decoded))
        
        decoded = decoded.view([3, self.im_width, self.im_height])

        return decoded

    def forward(self, x):
        return self.decode(self.decode(x))


# Hyper params
latent_size = 32 * 32 // 3
epochs = 1
batch_size = 4
hidden_size = 512
train_or_test = 'train'
path = models_dir + '/autoencoder'

# Training device
device = T.device('cuda:0' if cuda.is_available() else 'cpu')

# Dataset
trans = transforms.ToTensor()
dataset = CIFAR10(root=dataset_dir, train=True, download=True, transform=trans)
loader = T.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# Model
net = Net(32, 32, hidden_size, latent_size)
net.to(device)

if train_or_test == 'train':
    # Load
    if os.path.exists(path):
        net.load_state_dict(T.load(path))
        print('Model loaded')

    # Train
    optim = T.optim.RMSprop(net.parameters())
    criterion = nn.MSELoss()
    for e in range(epochs):
        avg_loss = 0
        for i, data in enumerate(loader, 0):
            # Only inputs (no labels)
            inputs, _ = data

            for batch in range(batch_size):
                # Zero the parameter gradients
                optim.zero_grad()

                # Predictions
                x = inputs[0].to(device)
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
    preds = T.cat([net(images[i].to(device)).view(1, 3, 32, 32).cpu() for i in range(batch_size)])
    preds = T.tensor(preds)
    gridshow(torchvision.utils.make_grid(preds))
