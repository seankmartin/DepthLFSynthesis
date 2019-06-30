from random import random

from inviwopy.glm import vec3, normalize

from lf_camera import cross_product

def random_float():
    """returns a random float between -1 and 1"""
    return (random() - 0.5) * 2

def rand_vec_in_unit_sphere():
    return vec3(random_float(), random_float(), random_float())

def rand_vec_between_spheres(big_radius, small_radius):
    """
    Generates a vec3 in between the shell of two spheres
    Input is the radius of the big and small sphere
    """
    radius_diff = big_radius - small_radius
    point_on_unit_sphere = normalize(rand_vec_in_unit_sphere())
    scale_factor = (random() * radius_diff + small_radius)
    point_in_between = point_on_unit_sphere * scale_factor    
    return point_in_between

def create_random_camera(look_from_radii,
                        max_look_to_origin=1.0,
                        look_up=vec3(0, 1, 0),
                        fix_look_up=False,
                        random_look_up=False):
    """Create a randomly positioned camera"""
    d = max_look_to_origin
    look_from = rand_vec_between_spheres(*look_from_radii)
    look_to = vec3(random_float(), random_float(), random_float()) * d
    reverse_direction = normalize(look_from - look_to)
    if random_look_up:
        look_up = normalize(
            vec3(random_float(), random_float(), random_float()))
    if fix_look_up:
        right = normalize(cross_product(look_up, reverse_direction))
        look_up = cross_product(reverse_direction, right)
    return (look_from, look_to, look_up)