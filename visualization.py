import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt


def highlight_mask(image, mask):
    '''
    :param image: numpy.array(M x N x 3) [0, 255]
    :param mask: numpy.array(M x N, dtype=np.bool)
    :return: image with highlighted mask
    '''

    M, N = mask.shape

    im = image.copy()
    mask_vec = mask.ravel()

    im_vec = im.reshape((M * N, 3))
    im_vec[mask_vec, :] /= 2.0
    im_vec[mask_vec, 0] += 128

    return im


def make_video(frames, masks):
    '''
    Play video using frames with highlighted masks
    :param frames: numpy.array(D x M x N x 3) [0, 255]
    :param masks: numpy.array(D x M x N, dtype=np.bool)
    D - number of frames.

    Example of usage:
    video = make_video(test_img, ans)
    video()
    '''

    highlighted_frames = np.zeros(frames.shape)
    for i in range(frames.shape[0]):
        highlighted_frames[i, :, :, :] = highlight_mask(frames[i, :, :, :], masks[i, :, :])

    fig = plt.figure()
    im = plt.imshow(highlighted_frames[0, :, :, :].astype(np.uint8))

    def updatefig(j):
        im.set_array(highlighted_frames[j, :, :, :].astype(np.uint8))
        return im,

    return lambda: animation.FuncAnimation(fig, updatefig, frames=highlighted_frames.shape[0], interval=100, blit=True)

# Example:
# video = make_video(test_img, ans)
# video()
