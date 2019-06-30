#Inviwo Python script 
import inviwopy
from inviwopy.glm import vec3, ivec2
import ivw.utils as inviwo_utils

import sys
import os

ivw_custom_py_modules_location = "/home/sean/LF_Volume_Synthesis/Inviwo/py_modules"
sys.path.append(ivw_custom_py_modules_location)

from lf_camera import LightFieldCamera

def main(pixel_dim):
    #Setup
    app = inviwopy.app
    network = app.network
    cam = network.EntryExitPoints.camera
    cam.lookUp = vec3(0, -1, 0)
    cam.nearPlane = 6.0
    cam.farPlane = 1000.0
    canvases = inviwopy.app.network.canvases
    for canvas in canvases:
        canvas.inputSize.dimensions.value = ivec2(pixel_dim, pixel_dim)
    inviwo_utils.update()

    # Create a light field camera at the current camera position
    lf_camera_here = LightFieldCamera(
        cam.lookFrom, cam.lookTo, cam.lookUp,
        interspatial_distance=0.5)

    #Preview the lf camera array
    save_dir = os.path.join(os.path.expanduser('~'), "lftset")
    lf_camera_here.view_array(cam, save=True, should_time=True, save_dir=save_dir)
    lf_camera_here.move_to_centre()

if __name__ == '__main__':
    PIXEL_DIM = 512
    main(PIXEL_DIM)
