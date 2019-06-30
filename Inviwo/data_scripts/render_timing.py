import inviwopy
from inviwopy.glm import vec3
import ivw.utils as inviwo_utils

import random
import time

import sys
# Location of py_modules folder
ivw_custom_py_modules_location = "/home/sean/LF_Volume_Synthesis/Inviwo/py_modules"
sys.path.append(ivw_custom_py_modules_location)

import welford

def random_f():
    rand_num = random.random()
    if rand_num < 0.5:
        sign = -1
    else: 
        sign = 1
    return sign * random.random()

def random_vec3(mult = 1.0):
    return vec3(random_f() * mult, random_f() * mult, random_f() * mult)

def time_one():
    mc = inviwopy.app.network.MeshClipping
    cam = mc.camera
    start_time = time.time()
    cam.setLook(random_vec3(), random_vec3(0), random_vec3())
    mc.alignPlaneNormalToCameraNormal.press()
    inviwo_utils.update()
    end_time = time.time() - start_time
    #print("Rendering complete in {:4f}".format(end_time))
    return end_time

def main(num_samples):
    random.seed(time.time())
    time_accumulator = (0, 0, 0)
    for _ in range(num_samples):
        last_time = time_one()
        time_accumulator = welford.update(time_accumulator, last_time)
    if num_samples > 1:
        mean_time, std_dev_time, _ = welford.finalize(time_accumulator)
        print("Overall time mean: {:4f}, stdev: {:4f}".format(mean_time, std_dev_time))

if __name__ == '__main__':
    NUM_SAMPLES = 300
    main(NUM_SAMPLES)