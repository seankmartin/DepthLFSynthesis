"""Contains data transforms which can be passed to the data loader"""
import math
import random

import numpy as np
import torch


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
    lf = sample['warped']
    lf.div_(maximum)
    return sample


def subsample_channels(sample, num_channels):
    """Extracts a subsample of channels from the sample"""
    sample['colour'] = sample['colour'][:, :num_channels, ...]
    sample['warped'] = sample['warped'][:, :num_channels, ...]
    return sample

def create_random_coords(pixel_end, n_samples, patch_size):
    """Return n random co-ords as V1, H1, V2, H2"""
    sample_co_ords = np.array(
        [n_samples, 4], np.uint16)
    for i in range(n_samples):
        high = pixel_end - 1 - patch_size
        start_h = random.randint(0, high)
        start_v = random.randint(0, high)
        end_h = start_h + patch_size
        end_v = start_v + patch_size
        sample_co_ords[i] = np.array([start_v, start_h, end_v, end_h])
    return sample_co_ords


def crop(sample, crop_cords):
    """co-ords should be V1, H1, V2, H2"""
    sample["colour"] = sample["colour"][:, :,
                                        crop_cords[0]:crop_cords[2],
                                        crop_cords[1]:crop_cords[3]]
    sample["warped"] = sample["warped"][:, :,
                                        crop_cords[0]:crop_cords[2],
                                        crop_cords[1]:crop_cords[3]]
    return sample


def angular_remap(sample):
    shape = sample['colour'].shape
    inputs = create_remap(sample['colour'], dtype=torch.float32)
    targets = create_remap(sample['warped'], dtype=torch.float32)
    return {'inputs': inputs, 'targets': targets, 'shape': shape}


def get_random_crop(sample, patch_size):
    pixel_end = sample['colour'].shape[1]
    high = pixel_end - 1 - patch_size
    start_h = random.randint(0, high)
    start_v = random.randint(0, high)
    end_h = start_h + patch_size
    end_v = start_v + patch_size
    sample = crop(sample, [start_v, start_h, end_v, end_h])
    return sample


def random_gamma(sample):
    maximum = 255
    gamma = random.uniform(0.4, 1.0)
    sample['colour'] = torch.pow(
        sample['colour'].div_(maximum), gamma).mul_(maximum)
    sample['warped'] = torch.pow(
        sample['warped'].div_(maximum), gamma).mul_(maximum)
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
