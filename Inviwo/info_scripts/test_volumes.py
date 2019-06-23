import inviwopy
import ivw.utils as inviwo_utils
from inviwopy.glm import vec3

import sys
# Location of py_modules folder
ivw_custom_py_modules_location = "/home/sean/LF_Volume_Synthesis/Inviwo/py_modules"
sys.path.append(ivw_custom_py_modules_location)

import h5py
import numpy as np

import os
import configparser
import pathlib
from random import seed, random
from time import sleep, time

from common import get_all_files_in_dir
from modify_transfer_func import modify_tf
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

def main(config):
    app = inviwopy.app
    network = app.network
    
    # Resize the canvas to improve rendering speed, only affects visual output
    if config["should_resize"]:
        ivw_helpers.set_canvas_sizes(128, 128)

    # Find the list of volumes and tfs form the corresponding directories
    volume_names = get_all_files_in_dir(
        config["volume_dir"], None, True)

    ivw_volume = get_volume(network, config)

    for volume_name in volume_names:
            if config["should_use_numpy_vol"]:
                ivw_volume.properties.location.value = volume_name
            else:
                ivw_volume.filename.value = volume_name
                ivw_volume.reload.press()
            inviwo_utils.update()
            sleep(1.2)

if __name__ == '__main__':
    home = os.path.expanduser('~')
    config = choose_cfg("head")
    lu = config['look_up']
    config['look_up'] = vec3(lu[0], lu[1], lu[2])

    main(config)