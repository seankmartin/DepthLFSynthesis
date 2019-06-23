from random import random

from math import ceil

from inviwopy.glm import vec3, normalize

from lf_camera import LightFieldCamera, cross_product
from random_camera import create_random_camera

def create_random_lf_cameras(num_to_create, look_from_radii,
                             max_look_to_origin=1.0,
                             interspatial_distance=1.0,
                             spatial_rows=8, spatial_cols=8,
                             look_up = vec3(0.0, 1.0, 0.0)):
    """Create a list of randomnly positioned lf cameras

    Keyword arguments:
    num_to_create -- the number of lf cameras to create
    look_from_radii -- min, max distance from the origin to camera look from
    max_look_to_origin -- max distance from the origin to camera look to    
    interspatial_distance -- distance between cameras in array (default 1.0)
    spatial_rows, spatial_cols -- dimensions of camera array (default 8 x 8)
    """
    lf_cameras = []
    for _ in range(num_to_create):
        look_from, look_to, look_up = create_random_camera(
            look_from_radii, max_look_to_origin, look_up, False)
        
        # Centre the light field on the centre image
        view_direction = look_to - look_from
        right_vec = normalize(cross_product(view_direction, look_up))
        correction_x = -ceil(spatial_cols / 2) * interspatial_distance
        correction_y = ceil(spatial_rows / 2) * interspatial_distance
        correction = (
            normalize(look_up) * correction_y +
            right_vec * correction_x)
        look_from = look_from + correction
        look_to = look_to + correction
        lf_cam = LightFieldCamera(look_from, look_to, look_up,
                                  interspatial_distance,
                                  spatial_rows, spatial_cols)
        lf_cameras.append(lf_cam)
    return lf_cameras