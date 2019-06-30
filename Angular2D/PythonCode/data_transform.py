"""Contains data transforms which can be passed to the data loader"""
import math
import random

import numpy as np
import torch
import matplotlib.cm as cm

import image_warping


def torch_stack(input_t, channels=64):
    # This has shape im_size, im_size, num_colours * num_images
    out = torch.squeeze(
        torch.cat(torch.chunk(input_t, channels, dim=0), dim=-1))

    # This has shape num_colour * num_images, im_size, im_size
    out = out.transpose_(0, 2).transpose_(1, 2)
    return out


def torch_unstack(input_t, channels=64):
    # This has shape num_images, num_channels, im_size, im_size
    input_back = torch.stack(torch.chunk(input_t, 64, dim=0))

    # This has shape num_images, im_size, im_size, num_channels
    input_back = input_back.transpose_(1, 3).transpose_(1, 2)

    return input_back


def stack(sample, channels=64):
    sample['inputs'] = torch_stack(sample['inputs'], channels)
    sample['targets'] = torch_stack(sample['targets'], channels)
    return sample


def normalise_sample(sample):
    """Coverts an lf in the range 0 to maximum into 0 1"""
    maximum = 255.0
    lf = sample['colour']
    lf.div_(maximum)
    return sample


def get_random_crop(sample, patch_size):
    pixel_end = sample['colour'].shape[1]
    high = pixel_end - 1 - patch_size
    start_h = random.randint(0, high)
    start_v = random.randint(0, high)
    end_h = start_h + patch_size
    end_v = start_v + patch_size
    sample['colour'] = sample['colour'][:, :, start_v:end_v, start_h:end_h]
    return sample


def random_gamma(sample):
    maximum = 255
    gamma = random.uniform(0.4, 1.0)
    sample['colour'] = torch.pow(
        sample['colour'].div_(maximum), gamma).mul_(maximum)
    return sample


def denormalise_lf(lf):
    """Coverts an lf in the range 0 1 to 0 to maximum"""
    maximum = 255.0
    lf.mul_(maximum)
    return lf


def create_remap(in_tensor, dtype=torch.uint8):
    """
    Remaps an input tensor of shape B, W, H, C into
    W * sqrt(B), H * sqrt(B), C
    Where each grid of size sqrt(B) * sqrt(B) in the remap
    contains pixel information from the 64 input images
    """
    num, channels, im_height, im_width = in_tensor.shape
    one_way = int(math.floor(math.sqrt(num)))
    out_tensor = torch.zeros(
        size=(channels, im_height * one_way, im_width * one_way),
        dtype=dtype)
    for i in range(num):
        out_tensor[
            :, i % one_way::one_way, i // one_way::one_way] = in_tensor[i]
    return out_tensor


def undo_remap(in_tensor, desired_shape, dtype=torch.uint8):
    """
    Remaps an input tensor to undo create_remap
    """
    num, _, _, _ = desired_shape
    one_way = int(math.floor(math.sqrt(num)))
    out_tensor = torch.zeros(size=desired_shape, dtype=dtype)
    for i in range(num):
        out_tensor[i] = in_tensor[
            :, i % one_way::one_way, i // one_way::one_way]
    return out_tensor


if __name__ == "__main__":
    "Test the remapping"
    loc = "LOCATION"
    out_loc = "LOCATION2"
    import h5py
    from PIL import Image
    with h5py.File(
            loc, mode='r',
            libver='latest', swmr=True) as h5_file:
        inp = h5_file["train/images"][0]
        data = create_remap(inp).numpy()
        data = np.swapaxes(data, 0, 2)
        data = np.swapaxes(data, 0, 1)
        out = Image.fromarray(data)
        out.save(out_loc)
