# Detects where cats or dogs are within an image
# Cats are red, dogs are blue
# Uses ./classification's network to detect cats / dogs
# The model must be already trained

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
import classification


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


# Params
im_width = 128
im_height = 128
img_path = './data/cats_dogs.jpg'

# Training device
device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

# Data
blue_img = T.tensor([0, 0, 1], dtype=T.float, device=device).reshape(3, 1, 1).repeat([1, im_width, im_height])
red_img = T.tensor([1, 0, 0], dtype=T.float, device=device).reshape(3, 1, 1).repeat([1, im_width, im_height])

# Network
net = classification.Net(im_width, im_height)
net.to(device)

# Load
if os.path.exists(classification.path):
    net.load_state_dict(T.load(classification.path))
    print('Model loaded')

# Show image with colors
img = im.open(img_path)
img = transforms.Compose([transforms.Resize((128 * 6, 128 * 8)), transforms.ToTensor()])(img).to(device)

img_show(detect_colorize(net, img, im_width, im_height))
