import configparser
import os
import pathlib

"""
    Config file members:
    constant_seed -- should the fixed seed of 100 be used
    should_resize -- resize the canvas to speed up display
    should_use_numpy_vol -- Indicates if using IVW volume loader or custom
    pixel_dim_* -- the size of the captured images
    clip -- boolean indicating if the volume should be clipped
    plane -- boolean indicating if clipping should be along a plane
                instead of a uniform box. If plane and clip are both set to
                true, uniform box clipping will be performed.
    set_types -- the groups to create in the hdf5 file
    num_samples -- the number of lf samples to generate as a dict of
                    {'train': num, 'val': num} for each volume tf combination
    hdf5_name -- the name of the output file.
    baseline -- the space between images in the LF grid.
    look_up -- the direction of the camera up vector.
    max_look_from -- how far away from the origin the camera may be
    min_look_from -- how close to the origin the camera may be
    spatial_* -- the number of rows or cols in the grid
    num_random_tfs -- the number of random transfer functions to generate
                        for each base transfer function
    save_main_dir -- the directory to save the hdf5 files to
    volume_dir -- the directory which contains the volume files to generate over
    base_tf_dir -- the directory which contains the ivw transfer functions to 
                    generate over
"""

def choose_cfg(choice):
    if choice == "head":
        return setup_head_cfg()
    elif choice == "looking":
        return setup_looking_cfg()
    elif choice == "default":
        return setup_default_cfg()
    elif choice == "generic":
        return setup_generic_cfg()
    elif choice == "tiny":
        return setup_tiny_cfg()
    else:
        print("Choice not found for cfg, using default")
        return setup_default_cfg() 

def setup_head_cfg():
    home = os.path.expanduser('~')

    config = {
            "constant_seed": True,
            "should_resize": False,
            "should_use_numpy_vol" : True,
            "pixel_dim_x": 256,
            "pixel_dim_y": 256,
            "clip": False,
            "plane": True,
            "set_types" : ['train', 'unseen_vol', 'unseen_tf'],
            "num_samples": {'train': 25, 'unseen_vol': 10, 'unseen_tf': 10},
            "num_random_tfs": {'train': 1, 'unseen_vol': 1, 'unseen_tf': 1},
            "hdf5_name": "head_set_new.h5",
            "baseline": 0.01,
            "look_up": [0, 0, 1],
            "max_look_from": 1.1,
            "min_look_from": 0.35,
            "spatial_rows": 8,
            "spatial_cols": 8,
            "channels": 3
        }
    config["save_main_dir"] = os.path.join(
        home, 'lf_datasets', 'test_sets')

    config["volume_dir"] = os.path.join(
        home, 'lf_data', 'head_volumes')

    config["unseen_volume_dir"] = os.path.join(
        home, 'lf_data', 'new_head_volumes')

    config["base_tf_dir"] = os.path.join(
        home, 'lf_data', 'head_tfs')

    config["unseen_tf_dir"] = os.path.join(
        home, 'lf_data', 'new_head_tfs')

    dirs = [
        config["volume_dir"], config["unseen_volume_dir"], 
        config["base_tf_dir"], config["unseen_tf_dir"]]

    for dir_ in dirs:
        if not os.path.isdir(dir_):
            print("{} does not exist, creating it now...".format(dir_))
            pathlib.Path(dir_).mkdir(parents=True, exist_ok=True)

    return config

def setup_looking_cfg():
    home = os.path.expanduser('~')

    config = {
        "constant_seed": True,
        "should_resize": False,
        "plane": False,
        "clip": False,
        "num_samples": {'train': 2000, 'val': 800},
        "hdf5_name": "test_looking.h5",
        "pixel_dim_x": 408,
        "pixel_dim_y": 226,
        "look_up": [0, -1, 0],
        "max_look_from": 0.8,
        "min_look_from": 0.25,
        "spatial_rows": 9,
        "spatial_cols": 5,
        "looking_size": [0.08, 0.42],
        "looking_cone": 40,
        "looking_vert_angle": 0
    }
    config["save_main_dir"] = os.path.join(
        home, 'lf_datasets', 'test_sets')

    return config

def setup_generic_cfg():
    home = os.path.expanduser('~')

    config = {
        "constant_seed": True,
        "should_resize": False,
        "plane": True,
        "clip": False,
        "set_types": {"train", "val"},
        "num_samples": {'train': 1000, 'val': 400},
        "hdf5_name": "test_new.h5",
        "look_up": [0, 1, 0],
        "pixel_dim_x": 512,
        "pixel_dim_y": 512,
        "max_look_from": 1.0,
        "min_look_from": 0.35,
        "spatial_rows": 8,
        "spatial_cols": 8,
        "random_light": False,
        "channels": 3
    }
    config["save_main_dir"] = os.path.join(
        home, 'lf_datasets', 'test_sets')

    return config

def setup_tiny_cfg():
    home = os.path.expanduser('~')

    config = {
        "constant_seed": True,
        "should_resize": False,
        "plane": True,
        "clip": False,
        "set_types": {"train", "val"},
        "num_samples": {'train': 10, 'val': 5},
        "hdf5_name": "tiny.h5",
        "look_up": [0, 1, 0],
        "pixel_dim_x": 512,
        "pixel_dim_y": 512,
        "max_look_from": 1.0,
        "min_look_from": 0.4,
        "spatial_rows": 8,
        "spatial_cols": 8,
        "random_light": False,
        "channels": 3
    }
    config["save_main_dir"] = os.path.join(
        home, 'lf_datasets', 'test_sets')

    return config

def setup_default_cfg():
    home = os.path.expanduser('~')

    config = {
            "constant_seed": True,
            "should_resize": False,
            "should_use_numpy_vol" : False,
            "pixel_dim_x": 256,
            "pixel_dim_y": 256,
            "clip": False,
            "plane": True,
            "set_types" : ['train', 'unseen_vol', 'unseen_tf'],
            "num_samples": {'train': 20, 'unseen_vol': 1, 'unseen_tf': 1},
            "num_random_tfs": {'train': 1, 'unseen_vol': 1, 'unseen_tf': 1},
            "hdf5_name": "deafult_set.h5",
            "baseline": 0.01,
            "look_up": [0, 1, 0],
            "max_look_from": 1.8,
            "min_look_from": 0.6,
            "spatial_rows": 8,
            "spatial_cols": 8,
        }
    config["save_main_dir"] = os.path.join(
        home, 'lf_datasets', 'test_sets')

    config["volume_dir"] = os.path.join(
        home, 'lf_data', 'vols')

    config["unseen_volume_dir"] = os.path.join(
        home, 'lf_data', 'new_vols')

    config["base_tf_dir"] = os.path.join(
        home, 'lf_data', 'tfs')

    config["unseen_tf_dir"] = os.path.join(
        home, 'lf_data', 'new_tfs')

    return config

def write_cfg(filename):
    choice = "head"

    config = configparser.ConfigParser()
    config['DEFAULT'] = choose_cfg(choice)

    with open(filename, 'w') as configfile:
        config.write(configfile)

if __name__ == "__main__":
    out_name = "config.ini"
    write_cfg(out_name)