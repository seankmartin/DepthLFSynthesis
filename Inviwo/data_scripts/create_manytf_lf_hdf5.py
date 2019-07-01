import inviwopy
import ivw.utils as inviwo_utils
from inviwopy.glm import vec3

import sys
# Location of py_modules folder
ivw_custom_py_modules_location = "/home/sean/repos/DepthLFSynthesis/Inviwo/py_modules"
sys.path.append(ivw_custom_py_modules_location)

import h5py
import numpy as np

import os
import configparser
import pathlib
from random import seed, random
from time import sleep, time

from random_camera import create_random_camera
from random_clip import random_clip_look_from, restore_clip
from random_clip import random_plane_clip_cam
from common import get_all_files_in_dir
from modify_transfer_func import modify_tf
from config_gen import choose_cfg
import welford
import ivw_helpers

def generate_random_tfs(base_tf_names, network, num_random, should_view=False):
    ivw_tf = network.VolumeRaycaster.isotfComposite.transferFunction
    tf_values = []
    if len(base_tf_names) > 0:
        for base_tf_name in base_tf_names:
            ivw_tf.load(base_tf_name)
            tf_values.append(ivw_tf.getValues())
            for _ in range(num_random):
                tf_values.append(modify_tf(ivw_tf))
                if should_view:
                    inviwo_utils.update()
                    sleep(0.2)
    else:
        tf_values.append(ivw_tf.getValues())
        for _ in range(num_random):
            tf_values.append(modify_tf(ivw_tf))
            if should_view:
                inviwo_utils.update()
                sleep(0.2)
    return tf_values

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

def create_hdf_storage(hdf5_file, config, network, num_vol_tfs):
    pixel_dim_x = config["pixel_dim_x"]
    pixel_dim_y = config["pixel_dim_y"]
    spatial_rows = config["spatial_rows"]
    spatial_cols = config["spatial_cols"]
    cam = network.MeshClipping.camera

    chs = config['channels']
    for set_type in config['set_types']:
        num_samples = (
            config["num_samples"][set_type] 
            * max(num_vol_tfs[set_type][0] * num_vol_tfs[set_type][1], 1) 
            * (config["num_random_tfs"][set_type] + 1)
        )
        colour = hdf5_file.create_group(set_type)
        colour.attrs['lf_shape'] = [
            num_samples, spatial_cols * spatial_rows,
            chs, pixel_dim_y, pixel_dim_x]
        colour.attrs['baseline'] = config["baseline"]
        colour.attrs['focal_length'] = cam.projectionMatrix[0][0]
        colour.attrs['near_plane'] = cam.nearPlane
        colour.attrs['far_plane'] = cam.farPlane
        colour.attrs['num_tfs'] = (
            num_vol_tfs[set_type][1] 
            * (config["num_random_tfs"][set_type] + 1))
        colour.attrs['num_vols'] = num_vol_tfs[set_type][0]
        colour.attrs['samples_per_tf'] = config["num_samples"][set_type]

        mean_shape = [num_samples, chs, pixel_dim_y, pixel_dim_x]
        time_shape = [num_samples, spatial_cols * spatial_rows]
        cam_shape = [num_samples, spatial_cols * spatial_rows, 9]
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
        chs = config['channels']
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
                group['images'][sample_index, idx] = im_data[:chs, ...]
                group['timing'][sample_index] = time_taken
                accumulator_list[i] = welford.update(
                    accumulator_list[i], 
                    np.asarray(im_data[:chs, ...], dtype=np.float32))

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
                group['warped'][sample_index, idx] = im_data[:chs, ...]

        for i, group_tuple in enumerate(h5_canvas_list):
            group = group_tuple[0]
            mean, var, _ = welford.finalize(accumulator_list[i])
            group['mean'][sample_index, :, :, :] = mean
            group['var'][sample_index, :, :, :] = var

        print("Finished writing LF {0} in {1:.2f}".format(
            sample_index, time() - overall_start_time))
        sleep(1)

def capture_lf_samples(hdf5_file, set_type, config, network, count):
    spatial_rows = config["spatial_rows"]
    spatial_cols = config["spatial_cols"]

    colour = hdf5_file[set_type]
    for _ in range(config["num_samples"][set_type]):
        random_cams = []
        radii = (config["max_look_from"], config["min_look_from"])
        random_cam = create_random_camera(
                        radii,
                        max_look_to_origin=0,
                        look_up=config["look_up"],
                        random_look_up=True)
        random_cams.append(random_cam)
    
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

    # Find the list of volumes and tfs form the corresponding directories
    volume_names = get_all_files_in_dir(
        config["volume_dir"], None, True)
    base_tf_names = get_all_files_in_dir(
        config["base_tf_dir"], ".itf", True)
    unseen_volume_names = get_all_files_in_dir(
        config["unseen_volume_dir"], None, True)
    unseen_tf_names = get_all_files_in_dir(
        config["unseen_tf_dir"], ".itf", True)

    num_vol_tfs = {}
    num_vol_tfs['train'] = \
        [len(volume_names), len(base_tf_names)]
    num_vol_tfs['unseen_vol'] = \
        [len(unseen_volume_names), len(base_tf_names)]
    num_vol_tfs['unseen_tf'] = \
        [len(volume_names), len(unseen_tf_names)]

    print("Generating and previewing the tranfer funcs")
    tf_values = generate_random_tfs(
        base_tf_names, network, 
        config["num_random_tfs"]["train"], should_view=True)
    new_tf_values = generate_random_tfs(
        unseen_tf_names, network, 
        config["num_random_tfs"]["unseen_tf"], should_view=True)
    
    utf8_list = lambda mylist : [a.encode('utf8') for a in mylist]

    hdf5_path = os.path.join(save_main_dir, config["hdf5_name"])
    with h5py.File(hdf5_path, mode='w', libver='latest') as hdf5_file:
        hdf5_file.swmr_mode = True
        create_hdf_storage(hdf5_file, config, network, num_vol_tfs)
        if "train" in config['set_types']:
            hdf5_file["train"].attrs["vol_names"] = utf8_list(volume_names)
            hdf5_file["train"].attrs["tf_names"] = utf8_list(base_tf_names)
        if "unseen_vol" in config['set_types']:
            hdf5_file["unseen_vol"].attrs["vol_names"] = utf8_list(unseen_volume_names)
            hdf5_file["unseen_vol"].attrs["tf_names"] = utf8_list(base_tf_names)
        if "unseen_tf" in config['set_types']:
            hdf5_file["unseen_tf"].attrs["vol_names"] = utf8_list(volume_names)
            hdf5_file["unseen_tf"].attrs["tf_names"] = utf8_list(unseen_tf_names)

        ivw_tf = network.VolumeRaycaster.isotfComposite.transferFunction
        ivw_volume = get_volume(network, config)

        count = {}
        for set_type in config['set_types']:
            count[set_type] = 0

        # Capture the data
        # There is more than one volume or transfer function to evaluate
        if num_vol_tfs['train'][0] or num_vol_tfs['train'][1] > 1:
            for volume_name in volume_names:
                    if config["should_use_numpy_vol"]:
                        ivw_volume.properties.location.value = volume_name
                    else:
                        ivw_volume.filename.value = volume_name
                        ivw_volume.reload.press()
                    for tf_value in tf_values:
                        ivw_tf.setValues(tf_value)
                        set_type = 'train'
                        count = capture_lf_samples(
                            hdf5_file, set_type, config, network, count)
                    for tf_value in new_tf_values:
                        ivw_tf.setValues(tf_value)
                        set_type = 'unseen_tf'
                        count = capture_lf_samples(
                            hdf5_file, set_type, config, network, count)

            for volume_name in unseen_volume_names:
                    if config["should_use_numpy_vol"]:
                        ivw_volume.properties.location.value = volume_name
                    else:
                        ivw_volume.filename.value = volume_name
                        ivw_volume.reload.press()
                    for tf_value in tf_values:
                        ivw_tf.setValues(tf_value)
                        set_type = 'unseen_vol'
                        count = capture_lf_samples(
                            hdf5_file, set_type, config, network, count)

        # Do a small test for training, can set part of it aside for testing
        else: 
            for tf_val in tf_values:
                set_type = config['set_types'][0]
                ivw_tf.setValues(tf_val)
                count = capture_lf_samples(
                    hdf5_file, set_type, config, network, count)

if __name__ == '__main__':
    config = choose_cfg("head")
    lu = config['look_up']
    config['look_up'] = vec3(lu[0], lu[1], lu[2])

    if config["constant_seed"]:
        seed(100)
    else:
        seed(time())

    main(config)