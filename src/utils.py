import os
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as im
import torch as T
from torch.nn import Module, Linear
import torch.nn.functional as F


# Constants #
# Download folder of the data
dataset_dir = './data'
models_dir = './models'

# Minimum value > 0 of float32
eps = np.finfo(np.float32).eps.item()


# IO #
def try_load_agent(agent, path):
    '''
        Loads the agent if the file exists
    '''
    if os.path.exists(path):
        agent.load_state_dict(T.load(path))


def save_agent(agent, path):
    T.save(agent.state_dict(), path)


# Image #
def tens2img(img):
    '''
    [channel, width, height] tensor to [width, height, channel] np.array
    '''
    return np.transpose(img.cpu().detach().numpy(), (1, 2, 0))


def bw2img(img):
    '''
        Black and white image to RGB
    - img : a tensor [channel, width, height]
    - return : a np.array [width, height, channel]
    '''
    return np.repeat(tens2img(img), 3, axis=2)


def img_load(path):
    '''
        Loads the image in a tensor.
    '''
    return T.from_numpy(np.transpose(np.array(im.open(path), dtype='float32'), (2, 0, 1))) / 255


def img_div(img, div_width, div_height):
    '''
        Divides an image in multiple subdivisions (subimages)
    - img : List of tensors
    - return : List of tensors
    !!! The borders may be ignored if the image's dimension is
    !!! not a multiple of div_width / div_height
    '''
    divs = []
    _, w, h = img.size()
    for y in range(0, h - div_height + 1, div_height):
        for x in range(0, w - div_width + 1, div_width):
            div = img[:, x : x + div_width, y : y + div_height]
            divs.append(div)

    return divs


def img_undiv(divs, width, device):
    '''
        Inverse of img_div, creates an image from subdivisions
    - width : Number of divs per width
    - return : Tensor
    - !!! Only for RGB images
    '''
    if len(divs) == 0:
        return

    if len(divs) % width != 0:
        raise Exception("width argument doesn't match the number of divs")

    height = len(divs) // width
    _, div_width, div_height = divs[0].size()
    img = T.empty(3, div_width * width, div_height *height)
    
    for y in range(height):
        for x in range(width):
            div = divs[y * width + x]
            img[:, x * div_width : (x + 1) * div_width, y * div_height : (y + 1) * div_height] = div

    return img


# Display #
def stack_show(batches, titles=None, bw=False):
    '''
        Shows a chart with all images in batches
        batches is a list of batch
    - bw : for black and white images
    '''
    h, w = len(batches), batches[0].size()[0]
    _, axarr = plt.subplots(h, w)

    trans = tens2img if not bw else bw2img

    for y in range(h):
        for x in range(w):
            axarr[y, x].imshow(trans(batches[y][x]))
            
            if titles:
                axarr[y, x].title.set_text(titles[y] + ' #' + str(x))

    plt.show()


def img_show(img, title=None, bw=False):
    '''
    Shows an image in a chart
    '''
    trans = tens2img if not bw else bw2img
    plt.imshow(trans(img))

    if title:
        plt.title(title)
    
    plt.show()


# Net #
class DenseBlock(Module):
    '''
        Dense block : [FC(x, h), FC(h, h), FC(h, y)]
    '''
    def __init__(self, n_x, n_h, n_y, activation=lambda x: x):
        super().__init__()

        self.activation = activation

        self.fc1 = Linear(n_x, n_h)
        self.fc2 = Linear(n_h, n_h)
        self.fc3 = Linear(n_h, n_y)

    def forward(self, x):
        x = F.relu(self.fc1(x), True)
        x = F.relu(self.fc2(x), True)
        x = self.activation(self.fc3(x))
    
        return x


class DenseLayer(Module):
    '''
        Dense layer : [FC(x, h), ReLU(), FC(h, y)]
    '''
    def __init__(self, n_x, n_h, n_y):
        super().__init__()

        self.fc1 = Linear(n_x, n_h)
        self.fc2 = Linear(n_h, n_y)

    def forward(self, x):
        x = F.relu(self.fc1(x), True)
        x = self.fc2(x)
    
        return x
