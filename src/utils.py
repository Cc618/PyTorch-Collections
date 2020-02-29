import numpy as np
import matplotlib.pyplot as plt

# Constants #
# Download folder of the data
dataset_dir = './data'
models_dir = './models'


# Image #
# [channel, width, height] tensor to [width, height, channel] np.array
def tens2img(img):
    return np.transpose(img.cpu().detach().numpy(), (1, 2, 0))


# Black and white image to RGB
# img is a tensor [channel, width, height]
# return is a np.array [width, height, channel]
def bw2img(img):
    return np.repeat(tens2img(img), 3, axis=2)


# Shows a chart with all images in batches
# batches is a list of batch
# bw for black and white images
def stack_show(batches, titles=None, bw=False):
    h, w = len(batches), batches[0].size()[0]
    f, axarr = plt.subplots(h, w)

    trans = (lambda x: x) if not bw else bw2img

    for y in range(h):
        for x in range(w):
            axarr[y, x].imshow(trans(batches[y][x]))
            
            if titles:
                axarr[y, x].title.set_text(titles[y] + ' #' + str(x))

    plt.show()


