import common
import os

import numpy as np
from PIL import Image

def main(in_img_loc):
    out_dir = os.path.dirname(in_img_loc)
    im = Image.open(in_img_loc)
    arr = common.decompose(
        np.array(im), [64, 512, 512, 4], 8
    )
    for i in range(arr.shape[0]):
        loc = os.path.join(
            out_dir, "out{}.png".format(i)
        )
        common.save_numpy_image(arr[i], loc)

if __name__ == "__main__":
    place = r"/home/sean/lftset/Test1/full.png"
    main(place)