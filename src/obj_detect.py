# Cats are red, dogs are blue

import os
import torch as T
from torch import nn
from torch import optim
from torch.nn import Conv2d, Linear, Module, ReLU, Softmax, Sigmoid, MaxPool2d, BatchNorm2d
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
import PIL.Image as im
from utils import dataset_dir, models_dir, img_load, img_show, stack_show, img_div, img_undiv


def colorize(img, class_ratio, alpha=1):
    '''
        Colorizes in blue / red the image with the ratio
        that this image belongs to a cat / dog
        (.5 = unknown, 0/1 = cat/dog at 100% chance)
    - alpha : Opacity of the color
    '''
    global red_img, blue_img

    if class_ratio > .5:
        # [0, 1], 1 = 100% chance
        ratio = (class_ratio - .5) * 2
        ratio *= alpha

        return img * (1 - ratio) + blue_img * ratio
    
    # [0, 1], 1 = 100% chance
    ratio = (.5 - class_ratio) * 2
    ratio *= alpha

    return img * (1 - ratio) + red_img * ratio


def detect_colorize(net, img, div_width, div_height, alpha=.5):
    '''
        Divides an image into subdivisions, detects and colorizes each
    subdivision with the color associated to the prediction and
    reconstructs the image with colors
    - img : Tensor
    - return : Tensor
    '''
    _, crop_width, _ = img.size()
    crop_width //= div_width
    
    # Divide
    divs = img_div(img, div_width, div_height)
    
    # Classify
    for i in range(len(divs)):
        divs[i] = colorize(divs[i], net(divs[i].unsqueeze(0)), alpha=alpha)

    # Reconstruct
    return img_undiv(divs, crop_width, device)


class Net(Module):
    def __init__(self, im_width, im_height):
        super().__init__()

        self.im_width = im_width
        self.im_height = im_height

        self.flatten_size = 32 * 13 * 13
        hidden_size = 1024
        hidden_size2 = 256

        self.conv1 = Conv2d(3, 16, 4)
        self.conv2 = Conv2d(16, 24, 5)
        self.conv3 = Conv2d(24, 32, 4)
        
        self.norm1 = BatchNorm2d(16)
        self.norm2 = BatchNorm2d(24)
        self.norm3 = BatchNorm2d(32)

        self.fc1 = Linear(self.flatten_size, hidden_size)
        self.fc2 = Linear(hidden_size, hidden_size2)
        self.fc3 = Linear(hidden_size2, 1)

    def forward(self, x):
        x = self.norm1(MaxPool2d(2)(ReLU(True)(self.conv1(x))))
        x = self.norm2(MaxPool2d(2)(ReLU(True)(self.conv2(x))))
        x = self.norm3(MaxPool2d(2)(ReLU(True)(self.conv3(x))))

        x = x.view(-1, self.flatten_size)

        x = ReLU(True)(self.fc1(x))
        x = ReLU(True)(self.fc2(x))
        x = Sigmoid()(self.fc3(x))

        return x


# Params
im_width = 128
im_height = 128
epochs = 1
batch_size = 100
learning_rate = .0002
train_or_test = 'test'
path = models_dir + '/obj_detect'

# Training device
device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

# Data
test_dataset_path = dataset_dir + '/cats_dogs/test'
train_dataset_path = dataset_dir + '/cats_dogs/train'
blue_img = T.tensor([0, 0, 1], dtype=T.float, device=device).reshape(3, 1, 1).repeat([1, im_width, im_height])
red_img = T.tensor([1, 0, 0], dtype=T.float, device=device).reshape(3, 1, 1).repeat([1, im_width, im_height])

# Network
net = Net(im_width, im_height)
net.to(device)

# Load
if os.path.exists(path):
    net.load_state_dict(T.load(path))
    print('Model loaded')

if train_or_test == 'train':

    # Dataset
    dataset = ImageFolder(
        root=train_dataset_path,
        transform=transforms.Compose([transforms.Resize((im_width, im_height)), transforms.ToTensor()])
    )

    loader = T.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True
    )

    # Train
    optim = T.optim.Adam(net.parameters(), lr=learning_rate, betas=(.9, .999))
    criterion = nn.BCELoss()
    for e in range(epochs):
        avg_loss = 0
        for i, data in enumerate(loader, 0):
            # Data
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device).to(T.float32).view(-1, 1)

            # Zero the parameter gradients
            optim.zero_grad()

            # Predictions
            y = net(inputs)

            # Back prop
            loss = criterion(y, labels)
            loss.backward()
            optim.step()

            avg_loss += loss.item()

            # Stats
            print_freq = 10
            if i % print_freq == print_freq - 1:
                print(f'Epoch {e + 1:2d}, Batch {i + 1:5d}, Loss {avg_loss / print_freq:.3f}')

                avg_loss = 0.0

        # Save
        T.save(net.state_dict(), path)

    print('Model trained and saved')

elif train_or_test == 'test':
    # Dataset
    dataset = ImageFolder(
        root=test_dataset_path,
        transform=transforms.Compose([transforms.Resize((im_width, im_height)), transforms.ToTensor()])
    )

    loader = T.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False
    )

    # Test
    ok = 0
    with T.no_grad():
        for i, (x, y) in enumerate(loader, 0):
            x, y = x.to(device), y.to(device)
            preds = net(x)
            oks = [round(pred.detach().cpu().item()) == label for pred, label in zip(preds, y)]
            for o in oks:
                if o:
                    ok += 1

    print(f'Accuracy : {ok / len(dataset) * 100:.1f} %')

elif train_or_test == 'misc':
    # Show image with colors
    tst_path = './data/cats_dogs.jpg'
    img = im.open(tst_path)
    img = transforms.Compose([transforms.Resize((128 * 6, 128 * 8)), transforms.ToTensor()])(img).to(device)

    img_show(detect_colorize(net, img, im_width, im_height))
