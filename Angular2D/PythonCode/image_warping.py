"""Image warping based on a disparity"""
import configparser
import os
from enum import Enum
import math
import argparse
from time import time

import h5py
from PIL import Image
import numpy as np
from skimage.transform import warp
import torch

import welford
import evaluate

class WARP_TYPE(Enum):
    FW = 1
    SK = 2
    SLOW = 3
    TORCH = 4
    TORCH_ALL = 5
    TORCH_GPU = 6

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return WARP_TYPE[s]
        except KeyError:
            raise ValueError()

def repeat_weights(weights, shape):
    """
    Takes in a one dimensional array of weights and repeats them to match shape
    Shape is expected to have the channels as the final dimension.
    """
    weights_extra = weights.unsqueeze(-1)
    return weights_extra.expand(
        weights.shape[0], shape[-1]).reshape(shape)

def torch_big_sample(array, indexes, desired_shape):
    """
    Samples a large torch array using indexes, which are x, y arrays
    Reshapes output to desired_shape
    """
    torch_arr = torch.tensor(array, dtype=torch.float32)
    indexed = torch_arr[[indexes[0], indexes[1]]]
    return indexed.reshape(desired_shape)
    #chunked = torch.chunk(indexed, desired_shape[0])
    #chunked = [chunk.reshape(desired_shape[1:]) for chunk in chunked]
    #out = torch.stack(chunked)

#Think I have slightly the wrong idea here
def depth_rendering(ref_view, disparity_map, lf_size = (64, 512, 512, 3)):
    """
    Perform full depth based rendering for a light field

    Keyword arguments:
    Ref_view - the central reference view,
    disparity_map - the central view disparity map,
    lf size (B, W, H, C)
    """
    lf_one_way = int(math.floor(math.sqrt(lf_size[0])))

    x_indices = np.arange(lf_size[1])
    y_indices = np.arange(lf_size[2])
    b_indices = np.arange(lf_size[0])

    #Create a grid of size lf_size[:3] consisting of the pixel co ordinates of each image
    _, x, y = np.meshgrid(b_indices, x_indices, y_indices, indexing= 'ij')

    # Create a grid of size (lf_size[0], 2) consiting of the row, col lf positions
    grid = np.meshgrid(np.arange(lf_one_way), np.arange(lf_one_way), indexing= 'ij')
    stacked = np.stack(grid, 2)
    positions = stacked.reshape(-1, 2)

    # Compute the distance from each lf position from the reference view
    # Repeat the elements of this to match the size of the disparity map
    ref_pos = np.array(
        [lf_one_way // 2, lf_one_way // 2])
    distance = (np.tile(ref_pos, (lf_size[0], 1)) - positions).T
    dis_repeated = np.repeat(distance, lf_size[1] * lf_size[2], axis = 1)
    dis_repeated = dis_repeated.reshape(2, lf_size[0], lf_size[1], lf_size[2])


    # Tile the disparity map so that there is one for each lf_position - lf_size[0]
    tiled_map = np.tile(disparity_map, (lf_size[0], 1, 1))

    # Compute the shifted pixels
    x_shifted = (x.astype(np.float32) - tiled_map * dis_repeated[0]).flatten()
    y_shifted = (y.astype(np.float32) - tiled_map * dis_repeated[1]).flatten()

    #indices for linear interpolation in a square around the central point
    x_low = np.floor(x_shifted).astype(int)
    x_high = x_low + 1

    y_low = np.floor(y_shifted).astype(int)
    y_high = y_low + 1

    #Place co-ordinates outside the image back into the image
    x_low_clip = np.clip(x_low, 0, ref_view.shape[0] - 1)
    x_high_clip = np.clip(x_high, 0, ref_view.shape[0] - 1)
    y_low_clip = np.clip(y_low, 0, ref_view.shape[1] - 1)
    y_high_clip = np.clip(y_high, 0, ref_view.shape[1] - 1)

    #Gather the interpolation points
    interp_pts_1 = np.stack((x_low_clip, y_low_clip))
    interp_pts_2 = np.stack((x_low_clip, y_high_clip))
    interp_pts_3 = np.stack((x_high_clip, y_low_clip))
    interp_pts_4 = np.stack((x_high_clip, y_high_clip))

    #Index into the images
    desired_shape = lf_size
    res_1 = torch_big_sample(ref_view, interp_pts_1, desired_shape)
    res_2 = torch_big_sample(ref_view, interp_pts_2, desired_shape)
    res_3 = torch_big_sample(ref_view, interp_pts_3, desired_shape)
    res_4 = torch_big_sample(ref_view, interp_pts_4, desired_shape)

    #Compute interpolation weights
    x_low_f = x_low.astype(np.float32)
    d_x_low = 1.0 - (x_shifted.astype(np.float32) - x_low_f)
    d_x_high = 1.0 - d_x_low
    y_low_f = y_low.astype(np.float32)
    d_y_low = 1.0 - (y_shifted.astype(np.float32) - y_low_f)
    d_y_high = 1.0 - d_y_low

    w1 = torch.from_numpy(d_x_low * d_y_low)
    w2 = torch.from_numpy(d_x_low * d_y_high)
    w3 = torch.from_numpy(d_x_high * d_y_low)
    w4 = torch.from_numpy(d_x_high * d_y_high)

    #THEY AGREE AT THIS POINT
    weighted_1 = torch.mul(repeat_weights(w1, desired_shape), res_1)
    weighted_2 = torch.mul(repeat_weights(w2, desired_shape), res_2)
    weighted_3 = torch.mul(repeat_weights(w3, desired_shape), res_3)
    weighted_4 = torch.mul(repeat_weights(w4, desired_shape), res_4)

    novel_view = torch.add(torch.add(weighted_1, weighted_2), weighted_3)
    torch.add(novel_view, weighted_4, out=novel_view)
    return novel_view

def torch_tensor_sample(tensor, indexes, desired_shape):
    indexed = tensor[[indexes[0], indexes[1]]]
    return indexed.reshape(desired_shape)
    #chunked = torch.chunk(indexed, desired_shape[0])
    #chunked = [chunk.reshape(desired_shape[1:]) for chunk in chunked]
    #out = torch.stack(chunked)

def depth_rendering_gpu(ref_view, disparity_map, lf_size = (64, 512, 512, 3)):
    """
    Perform full depth based rendering for a light field

    Keyword arguments:
    Ref_view - the central reference view,
    disparity_map - the central view disparity map,
    lf size (B, W, H, C)
    """
    lf_one_way = int(math.floor(math.sqrt(lf_size[0])))

    x_indices = np.arange(lf_size[1])
    y_indices = np.arange(lf_size[2])
    b_indices = np.arange(lf_size[0])

    #Create a grid of size lf_size[:3] consisting of the pixel co ordinates of each image
    _, x, y = np.meshgrid(b_indices, x_indices, y_indices, indexing= 'ij')

    # Create a grid of size (lf_size[0], 2) consiting of the row, col lf positions
    grid = np.meshgrid(np.arange(lf_one_way), np.arange(lf_one_way), indexing= 'ij')
    stacked = np.stack(grid, 2)
    positions = stacked.reshape(-1, 2)

    # Compute the distance from each lf position from the reference view
    # Repeat the elements of this to match the size of the disparity map
    ref_pos = np.array(
        [lf_one_way // 2, lf_one_way // 2])
    distance = (np.tile(ref_pos, (lf_size[0], 1)) - positions).T
    dis_repeated = np.repeat(distance, lf_size[1] * lf_size[2], axis = 1)
    dis_repeated = dis_repeated.reshape(2, lf_size[0], lf_size[1], lf_size[2])

    # Tile the disparity map so that there is one for each lf_position - lf_size[0]
    tiled_map = np.tile(disparity_map, (lf_size[0], 1, 1))

    # Compute the shifted pixels
    x_shifted = (x.astype(np.float32) - tiled_map * dis_repeated[0]).flatten()
    y_shifted = (y.astype(np.float32) - tiled_map * dis_repeated[1]).flatten()

    #indices for linear interpolation in a square around the central point
    x_low = np.floor(x_shifted).astype(int)
    x_high = x_low + 1

    y_low = np.floor(y_shifted).astype(int)
    y_high = y_low + 1

    #Place co-ordinates outside the image back into the image
    x_low_clip = np.clip(x_low, 0, ref_view.shape[0] - 1)
    x_high_clip = np.clip(x_high, 0, ref_view.shape[0] - 1)
    y_low_clip = np.clip(y_low, 0, ref_view.shape[1] - 1)
    y_high_clip = np.clip(y_high, 0, ref_view.shape[1] - 1)

    #Gather the interpolation points
    interp_pts_1 = np.stack((x_low_clip, y_low_clip))
    interp_pts_2 = np.stack((x_low_clip, y_high_clip))
    interp_pts_3 = np.stack((x_high_clip, y_low_clip))
    interp_pts_4 = np.stack((x_high_clip, y_high_clip))

    #Index into the images
    desired_shape = lf_size
    ref_view = torch.tensor(ref_view, dtype=torch.float32).cuda()
    res_1 = torch_tensor_sample(ref_view, interp_pts_1, desired_shape)
    res_2 = torch_tensor_sample(ref_view, interp_pts_2, desired_shape)
    res_3 = torch_tensor_sample(ref_view, interp_pts_3, desired_shape)
    res_4 = torch_tensor_sample(ref_view, interp_pts_4, desired_shape)

    #Compute interpolation weights
    x_low_f = x_low.astype(np.float32)
    d_x_low = 1.0 - (x_shifted.astype(np.float32) - x_low_f)
    d_x_high = 1.0 - d_x_low
    y_low_f = y_low.astype(np.float32)
    d_y_low = 1.0 - (y_shifted.astype(np.float32) - y_low_f)
    d_y_high = 1.0 - d_y_low

    w1 = torch.from_numpy(d_x_low * d_y_low)
    w2 = torch.from_numpy(d_x_low * d_y_high)
    w3 = torch.from_numpy(d_x_high * d_y_low)
    w4 = torch.from_numpy(d_x_high * d_y_high)

    #THEY AGREE AT THIS POINT
    weighted_1 = torch.mul(repeat_weights(w1, desired_shape).cuda(), res_1)
    weighted_2 = torch.mul(repeat_weights(w2, desired_shape).cuda(), res_2)
    weighted_3 = torch.mul(repeat_weights(w3, desired_shape).cuda(), res_3)
    weighted_4 = torch.mul(repeat_weights(w4, desired_shape).cuda(), res_4)

    novel_view = torch.add(torch.add(weighted_1, weighted_2), weighted_3)
    torch.add(novel_view, weighted_4, out=novel_view)
    return novel_view

def shift_disp(xy, disp, distance, dtype):
    #Repeat the elements of the disparity_map to match the distance
    size_x, size_y = disp.shape[0:2]

    #Needs to be tranposed to match expected cols, rows
    repeated = np.repeat(disp.T, 2, -1).reshape((size_x * size_y, 2))

    #Convert to desired dtype
    result = (repeated * distance).astype(dtype)
    return xy - result

def sk_warp(
    ref_view, disparity_map, ref_pos, novel_pos,
    dtype=np.float32, blank=0, preserve_range=False):
    """
    Uses skimage to perform backward warping:

    Keyword arguments:
    ref_view -- colour image data at the reference position
    disparity_map -- a disparity map at the reference position
    ref_pos -- the grid co-ordinates of the ref_view
    novel_pos -- the target grid position for the novel view
    dtype -- data type to consider disparity as
    blank -- value to use at positions not seen in reference view
    preserve_range -- Keep the data in range 0, 255 or convert to 0 1
    """
    distance = ref_pos - novel_pos

    novel_view = warp(
        image=ref_view, inverse_map=shift_disp,
        map_args={
            "disp": disparity_map, "distance": np.flipud(distance),
            "dtype": dtype},
        cval=blank, preserve_range=preserve_range, order=1
        )
    if preserve_range:
        novel_view = np.around(novel_view).astype(np.uint8)
    return novel_view

def valid_pixel(pixel, img_size):
    """Returns true if the pixel co-ordinate lies inside the image grid"""
    size_x, size_y = img_size
    valid = (((pixel[0] > -1) and (pixel[0] < size_x)) and
             ((pixel[1] > -1) and (pixel[1] < size_y)))
    return valid

def torch_sample(array, indexes, desired_shape):
    """
    From a numpy array, and a set of indices of shape [2, X]
    Where indexes[0] denotes the x indices of the array
    and indexes[1] denotes the y indices of the array
    return array indexed at these positions
    """
    torch_arr = torch.tensor(array, dtype=torch.float32)
    indexed = torch_arr[[indexes[0], indexes[1]]]
    return indexed.reshape(desired_shape)

def torch_warp(
        ref_view, disparity_map, ref_pos, novel_pos):
    s_indices = np.arange(ref_view.shape[0])
    t_indices = np.arange(ref_view.shape[0])

    x, y = np.meshgrid(s_indices, t_indices, indexing= 'ij')

    distance = ref_pos - novel_pos

    #print(np.array([x.reshape(ref_view.shape[:-1]),
    #      y.reshape(ref_view.shape[:-1])]))
    x_shifted = (x.astype(np.float32) - disparity_map * distance[0]).flatten()
    y_shifted = (y.astype(np.float32) - disparity_map * distance[1]).flatten()
    #print(np.array([x_shifted.reshape(ref_view.shape[:-1]),
    #      y_shifted.reshape(ref_view.shape[:-1])]))

    #indices for linear interpolation in a square around the central point
    x_low = np.floor(x_shifted).astype(int)
    x_high = x_low + 1

    y_low = np.floor(y_shifted).astype(int)
    y_high = y_low + 1

    #Place co-ordinates outside the image back into the image
    x_low_clip = np.clip(x_low, 0, ref_view.shape[0] - 1)
    x_high_clip = np.clip(x_high, 0, ref_view.shape[0] - 1)
    y_low_clip = np.clip(y_low, 0, ref_view.shape[1] - 1)
    y_high_clip = np.clip(y_high, 0, ref_view.shape[1] - 1)
    #print(np.array([x_low_clip.reshape(ref_view.shape[:-1]),
    #      y_low_clip.reshape(ref_view.shape[:-1])]))

    #Gather the interpolation points
    interp_pts_1 = np.stack((x_low_clip, y_low_clip))
    interp_pts_2 = np.stack((x_low_clip, y_high_clip))
    interp_pts_3 = np.stack((x_high_clip, y_low_clip))
    interp_pts_4 = np.stack((x_high_clip, y_high_clip))

    #Index into the images
    desired_shape = ref_view.shape
    res_1 = torch_sample(ref_view, interp_pts_1, desired_shape)
    res_2 = torch_sample(ref_view, interp_pts_2, desired_shape)
    res_3 = torch_sample(ref_view, interp_pts_3, desired_shape)
    res_4 = torch_sample(ref_view, interp_pts_4, desired_shape)

    #Compute interpolation weights
    x_low_f = x_low.astype(np.float32)
    d_x_low = 1.0 - (x_shifted - x_low_f)
    d_x_high = 1.0 - d_x_low
    y_low_f = y_low.astype(np.float32)
    d_y_low = 1.0 - (y_shifted - y_low_f)
    d_y_high = 1.0 - d_y_low

    w1 = torch.from_numpy(d_x_low * d_y_low)
    w2 = torch.from_numpy(d_x_low * d_y_high)
    w3 = torch.from_numpy(d_x_high * d_y_low)
    w4 = torch.from_numpy(d_x_high * d_y_high)

    weighted_1 = torch.mul(repeat_weights(w1, desired_shape), res_1)
    weighted_2 = torch.mul(repeat_weights(w2, desired_shape), res_2)
    weighted_3 = torch.mul(repeat_weights(w3, desired_shape), res_3)
    weighted_4 = torch.mul(repeat_weights(w4, desired_shape), res_4)

    novel_view = torch.add(torch.add(weighted_1, weighted_2), weighted_3)
    torch.add(novel_view, weighted_4, out=novel_view)
    return novel_view

def fw_warp_image(
    ref_view, disparity_map, ref_pos, novel_pos,
    dtype=np.uint8, blank=0):
    """
    Returns a forward warped novel from an input image and disparity_map
    For each pixel position in the reference view, shift it by the disparity,
    and assign the value in the reference at that new pixel position to the
    novel view.

    Keyword arguments:
    ref_view -- colour image data at the reference position
    disparity_map -- a disparity map at the reference position
    ref_pos -- the grid co-ordinates of the ref_view
    novel_pos -- the target grid position for the novel view
    """
    size_x, size_y = ref_view.shape[0:2]
    distance = ref_pos - novel_pos

    #Initialise an array of blanks
    novel_view = np.full(ref_view.shape, blank, dtype=dtype)

    #Create an array of pixel positions
    grid = np.meshgrid(np.arange(size_x), np.arange(size_y), indexing='ij')
    stacked = np.stack(grid, 2)
    pixels = stacked.reshape(-1, 2)

    #Repeat the elements of the disparity_map to match the distance
    repeated = np.repeat(disparity_map, 2, -1).reshape((size_x * size_y, 2))

    #Round to the nearest integer value
    result = (repeated * distance).astype(int)
    novel_pixels = pixels + result

    #Move the pixels from the reference view to the novel view
    for novel_coord, ref_coord in zip(novel_pixels, pixels):
        if valid_pixel(novel_coord, ref_view.shape[0:2]):
            novel_view[novel_coord[0], novel_coord[1]] = (
                ref_view[ref_coord[0], ref_coord[1]])

    return novel_view

def slow_fw_warp_image(ref_view, disparity_map, ref_pos, novel_pos):
    """
    Returns a forward warped novel from an input image and disparity_map
    For each pixel position in the reference view, shift it by the disparity,
    and assign the value in the reference at that new pixel position to the
    novel view.
    Has a very large for loop, performance is much slower than
    fw_warp_image

    Keyword arguments:
    ref_view -- colour image data at the reference position
    disparity_map -- a disparity map at the reference position
    ref_pos -- the grid co-ordinates of the ref_view
    novel_pos -- the target grid position for the novel view
    """
    size_x, size_y = ref_view.shape[0:2]
    distance = ref_pos - novel_pos

    novel_view = np.zeros(ref_view.shape, dtype=np.uint8)
    for x in range(size_x):
        for y in range(size_y):
            res = np.repeat(disparity_map[x, y], 2, -1) * distance
            new_pixel = ((x, y) + res).astype(int)
            if valid_pixel(new_pixel, (size_x, size_y)):
                novel_view[new_pixel[0], new_pixel[1]] = ref_view[x, y]
    return novel_view

def save_array_as_image(array, save_location):
    """Saves an array as an image at the save_location using pillow"""
    image = Image.fromarray(array)
    image.save(save_location)
    image.close()

def get_diff_image(im1, im2):
    diff = np.subtract(im1.astype(float), im2.astype(float))
    diff = np.around(abs(diff)).astype(np.uint8)
    return diff

def get_diff_image_floatint(im1_float, im2_int):
    diff = np.subtract(im1_float, im2_int.astype(float) / 255.0)
    diff = np.around(abs(diff)).astype(np.uint8)
    return diff

def get_sub_dir_for_saving(base_dir):
    """
    Returns the number of sub directories of base_dir, n, in format
    base_dir + path_separator + n
    Where n is padded on the left by zeroes to be of length four

    Example: base_dir is /home/sean/test with two sub directories
    Output: /home/sean/test/0002
    """
    num_sub_dirs = sum(os.path.isdir(os.path.join(base_dir, el))
                   for el in os.listdir(base_dir))

    sub_dir_to_save_to_name = str(num_sub_dirs)
    sub_dir_to_save_to_name = sub_dir_to_save_to_name.zfill(4)

    sub_dir_to_save_to = os.path.join(base_dir, sub_dir_to_save_to_name)
    os.mkdir(sub_dir_to_save_to)

    return sub_dir_to_save_to

def main(args, config):
    hdf5_path = os.path.join(config['PATH']['output_dir'],
                             config['PATH']['hdf5_name'])
    #warp_type = WARP_TYPE.TORCH_GPU
    warp_type = args.warp_type
    print("Performing image warping using {}".format(warp_type))
    with h5py.File(hdf5_path, mode='r', libver='latest') as hdf5_file:
        grid_size = 64
        grid_one_way = 8
        sample_index = grid_size // 2 + (grid_one_way // 2)
        depth_grp = hdf5_file['val']['disparity']

        overall_psnr_accum = (0, 0, 0)
        overall_ssim_accum = (0, 0, 0)
        for sample_num in range(args.nSamples):
            SNUM = sample_num
            print("Working on image", SNUM)
            depth_image = np.squeeze(depth_grp['images'][SNUM, sample_index])

            #Hardcoded some values for now
            colour_grp = hdf5_file['val']['colour']
            colour_image = colour_grp['images'][SNUM, sample_index]

            #Can later expand like 0000 if needed
            base_dir = os.path.join(config['PATH']['output_dir'], 'warped')
            get_diff = (config['DEFAULT']['should_get_diff'] == 'True')

            if not args.no_save:
                save_dir = get_sub_dir_for_saving(base_dir)
                print("Saving images to {}".format(save_dir))
            else:
                print("Not saving images, only evaluating output")

            psnr_accumulator = (0, 0, 0)
            ssim_accumulator = (0, 0, 0)

            start_time = time()
            if warp_type == WARP_TYPE.TORCH_ALL:
                final = depth_rendering(colour_image, depth_image)
                print("Time taken was {:4f}".format(time() - start_time))

            if warp_type == WARP_TYPE.TORCH_GPU:
                final = depth_rendering_gpu(colour_image, depth_image).cpu()
                print("Time taken was {:4f}".format(time() - start_time))

            ref_pos = np.asarray([4, 4])
            print("Reference position is ({}, {})".format(*ref_pos))
            for i in range(8):
                for j in range(8):
                    if warp_type == WARP_TYPE.FW:
                        res = fw_warp_image(colour_image, depth_image,
                                            ref_pos, np.asarray([i, j]))
                    elif warp_type == WARP_TYPE.SK:
                        res = sk_warp(
                            colour_image, depth_image,
                            ref_pos, np.asarray([i, j]),
                            preserve_range=True
                        )
                    elif warp_type == WARP_TYPE.SLOW:
                        res = slow_fw_warp_image(
                            colour_image, depth_image,
                            ref_pos, np.asarray([i, j])
                        )
                    elif warp_type == WARP_TYPE.TORCH:
                        res = np.around(torch_warp(
                            colour_image, depth_image,
                            ref_pos, np.asarray([i, j])
                        ).numpy()).astype(np.uint8)
                    elif (warp_type == WARP_TYPE.TORCH_ALL or
                          warp_type == WARP_TYPE.TORCH_GPU):
                        res = np.around(final[i * 8 + j].numpy()).astype(np.uint8)

                    if not args.no_save:
                        file_name = 'Warped_Colour{}{}.png'.format(i, j)
                        save_location = os.path.join(save_dir, file_name)
                        save_array_as_image(res, save_location)
                        idx = i * 8 + j
                        file_name = 'GT_Colour{}{}.png'.format(i, j)
                        save_location = os.path.join(save_dir, file_name)
                        save_array_as_image(
                            colour_grp['images'][SNUM][idx], save_location)
                        if get_diff:
                            colour = colour_grp['images'][SNUM, i * 8 + j]
                            diff = get_diff_image(colour, res)
                            #diff = get_diff_image_floatint(res, colour)
                            file_name = 'Diff{}{}.png'.format(i, j)
                            save_location = os.path.join(save_dir, file_name)
                            save_array_as_image(diff, save_location)
                    psnr = evaluate.my_psnr(
                        res,
                        colour_grp['images'][SNUM, i * 8 + j])
                    ssim = evaluate.ssim(
                        res,
                        colour_grp['images'][SNUM, i * 8 + j])
                    print("Position ({}, {}): PSNR {:4f}, SSIM {:4f}".format(
                        i, j, psnr, ssim))
                    psnr_accumulator = welford.update(psnr_accumulator, psnr)
                    ssim_accumulator = welford.update(ssim_accumulator, ssim)

            psnr_mean, psnr_var, _ = welford.finalize(psnr_accumulator)
            ssim_mean, ssim_var, _ = welford.finalize(ssim_accumulator)
            print("\npsnr average {:5f}, stddev {:5f}".format(
                psnr_mean, math.sqrt(psnr_var)))
            print("ssim average {:5f}, stddev {:5f}".format(
                ssim_mean, math.sqrt(ssim_var)))
            overall_psnr_accum = welford.update(
                overall_psnr_accum, psnr_mean)
            overall_ssim_accum = welford.update(
                overall_ssim_accum, ssim_mean)
        if args.nSamples > 1:
            psnr_mean, psnr_var, _ = welford.finalize(overall_psnr_accum)
            ssim_mean, ssim_var, _ = welford.finalize(overall_ssim_accum)
            print("\nOverall psnr average {:5f}, stddev {:5f}".format(
                psnr_mean, math.sqrt(psnr_var)))
            print("Overall ssim average {:5f}, stddev {:5f}".format(
                ssim_mean, math.sqrt(ssim_var)))

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(
        description='Process modifiable parameters from command line')
    PARSER.add_argument("--no_save", "--ns", action='store_true',
                        help="Should not save images")
    PARSER.add_argument("--nSamples", "--n", type=int, default=1,
                        help="Number of sample images to warp")
    PARSER.add_argument("--warp_type",
        type=lambda warp_type: WARP_TYPE[warp_type],
        choices=list(WARP_TYPE),
        default="TORCH_ALL",
        help="Which type of warping to use")
    ARGS, UNPARSED = PARSER.parse_known_args()
    if len(UNPARSED) is not 0:
        print("Unrecognised command line argument passed")
        print(UNPARSED)
        exit(-1)

    CONFIG = configparser.ConfigParser()
    CONFIG.read(os.path.join('config', 'hdf5.ini'))
    DIRTOMAKE = os.path.join(CONFIG['PATH']['output_dir'], 'warped')
    if not os.path.exists(DIRTOMAKE):
        os.makedirs(DIRTOMAKE)
    main(ARGS, CONFIG)
