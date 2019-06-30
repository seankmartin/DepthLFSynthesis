import inviwopy
from inviwopy.glm import vec3, length
import ivw.utils as inviwo_utils

import sys
from time import time
# Location of py_modules folder
ivw_custom_py_modules_location = "/home/sean/LF_Volume_Synthesis/Inviwo/py_modules"
sys.path.append(ivw_custom_py_modules_location)

import h5py
import numpy as np

import os
import pathlib
from random import seed, random
from time import sleep, time

from random_camera import create_random_camera
from random_clip import random_clip_look_from, restore_clip
from random_clip import random_plane_clip_cam
from config_gen import choose_cfg
import welford
import ivw_helpers

def get_volume(network, config):
    if config["should_use_numpy_vol"]:
            ivw_volume = network.NumpyVolumeLoader
            if ivw_volume == None:
                print("Attempting to use Python Script Proccessor as id")
                print("Recommended as Numpy Volume Loader")
                ivw_volume = network.PythonScriptProcessor
                if ivw_volume == None:
                    print(
                    "Please change the identifier of the volume loader to:",
                    "Numpy Volume Loader",
                    "Or add a python script processor to the network")
                    exit(-1)
    else:
        ivw_volume = network.VolumeSource
    return ivw_volume

def create_hdf_storage(hdf5_file, config):
    pixel_dim_x = config["pixel_dim_x"]
    pixel_dim_y = config["pixel_dim_y"]
    spatial_rows = config["spatial_rows"]
    spatial_cols = config["spatial_cols"]
    cam = inviwopy.app.network.MeshClipping.camera

    for set_type in config["set_types"]:    
        num_samples = config["num_samples"][set_type]

        # Setup the hdf5
        chs = config['channels']
        colour = hdf5_file.create_group(set_type)
        colour.attrs['lf_shape'] = [
            num_samples, spatial_cols * spatial_rows,
            chs, pixel_dim_y, pixel_dim_x]
        colour.attrs['focal_length'] = cam.projectionMatrix[0][0]
        colour.attrs['near_plane'] = cam.nearPlane
        colour.attrs['far_plane'] = cam.farPlane
        mean_shape = [num_samples, chs, pixel_dim_y, pixel_dim_x]
        time_shape = [num_samples]
        cam_shape = [num_samples, 9]
        colour.create_dataset('images', colour.attrs['lf_shape'], np.uint8,
                                chunks = (1, 1, chs, pixel_dim_y, pixel_dim_x),
                                compression = "lzf",
                                shuffle = True)
        colour.create_dataset('warped', colour.attrs['lf_shape'], np.uint8,
                                chunks = (1, 1, chs, pixel_dim_y, pixel_dim_x),
                                compression = "lzf",
                                shuffle = True)
        colour.create_dataset('mean', mean_shape, np.float32,
                                chunks = (1, chs, pixel_dim_y, pixel_dim_x),
                                compression = "lzf",
                                shuffle = True)
        colour.create_dataset('var', mean_shape, np.float32,
                                chunks = (1, chs, pixel_dim_y, pixel_dim_x),
                                compression = "lzf",
                                shuffle = True)
        colour.create_dataset('timing', time_shape, np.float32)
        colour.create_dataset('camera_extrinsics', cam_shape, np.float32)

def save_looking_to_hdf5_group(
    sample_index, h5_canvas_list, camera, config):
        """Saves lf information to hdf5 group with images over columns then rows
        
        Keyword arguments:
        h5_canvas_list - tuples of groups and canvas numbers in a list
                         for example (colour_group, 0)
        """
        overall_start_time = time()
        cam = inviwopy.app.network.MeshClipping.camera
        
        accumulator_list = []
        # Gathers accumulators for welford mean and var
        for i, group_tuple in enumerate(h5_canvas_list):
            group = group_tuple[0]
            mean_shape = (1, ) + group["mean"].shape[1:]
            mean = np.zeros(mean_shape, np.float32)
            accumulator = (0, mean, 0)
            accumulator_list.append(accumulator)

        start_time = time()
        cam.lookFrom = camera[0]
        cam.lookTo = camera[1]
        cam.lookUp = camera[2]
        inviwo_utils.update()
        time_taken = time() - start_time
        sleep(1)

        for i, group_tuple in enumerate(h5_canvas_list):
            #assuming that there is two canvases
            group = group_tuple[0]
            canvas = ivw_helpers.get_canvas_by_id(
                inviwopy.app.network, group_tuple[1])
            w_canvas = ivw_helpers.get_canvas_by_id(
                inviwopy.app.network, group_tuple[2])
            assert (canvas.inputSize.customInputDimensions.value[0] // config["spatial_cols"] == 
                    group.attrs["lf_shape"][-1]), \
                    "canvas size x {} is not pixel dimension {} of h5".format(
                        canvas.inputSize.customInputDimensions.value[0] // config["spatial_cols"],
                        group.attrs["lf_shape"][-1]
                    )
            assert (canvas.inputSize.customInputDimensions.value[1] // config["spatial_rows"] == 
                    group.attrs["lf_shape"][-2]), \
                    "canvas size y {} is not pixel dimension {} of h5".format(
                        canvas.inputSize.customInputDimensions.value[1] // config["spatial_rows"],
                        group.attrs["lf_shape"][-2]
                    )
            total_im_data = ivw_helpers.get_image(canvas)

            group['camera_extrinsics'][sample_index] = \
            [cam.lookFrom.x, cam.lookFrom.y, cam.lookFrom.z,
            cam.lookTo.x, cam.lookTo.y, cam.lookTo.z,
            cam.lookUp.x, cam.lookUp.y, cam.lookUp.z]

            # Inviwo stores data in a different indexing to regular
            # Store as descending y: C, H, W
            for idx in range(64):
                y_start = config["pixel_dim_y"] * (idx // 8)
                y_end = y_start + (config["pixel_dim_y"])
                x_start = config["pixel_dim_x"] * (idx % 8)
                x_end = x_start + (config["pixel_dim_x"])
                
                im_data = total_im_data[x_start:x_end, y_start:y_end]
                im_data = np.flipud(np.swapaxes(im_data, 0, 2))
                im_data = im_data[::-1, ::-1, ...]
                group['images'][sample_index, idx] = im_data[:3, ...]
                group['timing'][sample_index] = time_taken
                accumulator_list[i] = welford.update(
                    accumulator_list[i], np.asarray(im_data, dtype=np.float32))

            total_im_data = ivw_helpers.get_image(w_canvas)
            # Inviwo stores data in a different indexing to regular
            # Store as descending y: C, H, W
            for idx in range(64):
                y_start = config["pixel_dim_y"] * (idx // 8)
                y_end = y_start + (config["pixel_dim_y"])
                x_start = config["pixel_dim_x"] * (idx % 8)
                x_end = x_start + (config["pixel_dim_x"])
                
                im_data = total_im_data[x_start:x_end, y_start:y_end]
                im_data = np.flipud(np.swapaxes(im_data, 0, 2))
                im_data = im_data[::-1, ::-1, ...]
                group['warped'][sample_index, idx] = im_data[:3, ...]

        for i, group_tuple in enumerate(h5_canvas_list):
            group = group_tuple[0]
            mean, var, _ = welford.finalize(accumulator_list[i])
            group['mean'][sample_index, :, :, :] = mean[:3, ...]
            group['var'][sample_index, :, :, :] = var[3:, ...]

        print("Finished writing LF {0} in {1:.2f}".format(
            sample_index, time() - overall_start_time))
        sleep(1)


def random_float():
    """returns a random float between -1 and 1"""
    return (random() - 0.5) * 2

def capture_lf_samples(hdf5_file, set_type, config, network, count):
    random_cams = []
    for _ in range(config["num_samples"][set_type]):
        radii = (config["max_look_from"], config["min_look_from"])
        random_cam = create_random_camera(
                        radii,
                        max_look_to_origin=0,
                        look_up=config["look_up"],
                        random_look_up=True)
        random_cams.append(random_cam)

    colour = hdf5_file[set_type]
    for camera in random_cams:
        if config["clip"]:
            _, clip_type = random_clip_look_from(network, camera[0])
        elif config["plane"]:
            random_plane_clip_cam(network, camera)
        save_looking_to_hdf5_group(
            sample_index=count[set_type],
            h5_canvas_list=[(colour, "LF", "Warp")],
            camera=camera, config=config)
        count[set_type] += 1
        if config["clip"]:
            restore_clip(network, clip_type)
        elif config["plane"]:
            mesh_clip = network.MeshClipping
            mesh_clip.getPropertyByIdentifier(
                "clippingEnabled").value = False
    return count

def main(config):
    """ Generate a number of random train val lf samples, saving to hdf5"""
    save_main_dir = config["save_main_dir"]

    if not os.path.isdir(save_main_dir):
        print("{} does not exist, creating it now...", save_main_dir)
        pathlib.Path(save_main_dir).mkdir(parents=True, exist_ok=True)

    app = inviwopy.app
    network = app.network
    
    # Resize the canvas to improve rendering speed, only affects visual output
    if config["should_resize"]:
        ivw_helpers.set_canvas_sizes(128, 128)

    hdf5_path = os.path.join(save_main_dir, config["hdf5_name"])
    with h5py.File(hdf5_path, mode='w', libver='latest') as hdf5_file:
        hdf5_file.swmr_mode = True
        create_hdf_storage(hdf5_file, config)
        count = {"train": 0, "val": 0}
        for set_type in 'train', 'val':
            capture_lf_samples(hdf5_file, set_type, config, network, count)    
    print("Finished writing to HDF5 in {}".format(hdf5_path))

if __name__ == '__main__':
    config = choose_cfg("generic")
    lu = config['look_up']
    config['look_up'] = vec3(lu[0], lu[1], lu[2])
    if config["constant_seed"]:
        seed(100)
    else:
        seed(time())

    main(config)
