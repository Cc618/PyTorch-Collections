import numpy as np
import matplotlib.pyplot as plt

# Constants #
# Download folder of the data
dataset_dir = './data'
models_dir = './models'


# Image #
# [channel, width, height] tensor to [width, height, channel] np.array
def tens2img(img):
    return np.transpose(img.numpy(), (1, 2, 0))


# Black and white image to RGB
# img is a tensor [channel, width, height]
# return is a np.array [width, height, channel]
def bw2img(img):
    return np.repeat(tens2img(img), 3, axis=2)

# Shows a chart with all images
# bw for black and white images
def batchshow(top_batch, bot_batch, batch_size, bw=False):
    f, axarr = plt.subplots(2, batch_size)

    trans = (lambda x: x) if not bw else bw2img

    for x in range(batch_size):
        axarr[0, x].imshow(trans(top_batch[x]))
        axarr[1, x].imshow(trans(bot_batch[x]))

    plt.show()


