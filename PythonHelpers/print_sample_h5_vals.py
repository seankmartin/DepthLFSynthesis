import os

import h5py
import numpy as np

import common

def main(file_location, out_loc):
    with h5py.File(file_location, 'r') as f:
        for i in range(3):
            for j in range(64):
                im_out_location = os.path.join(
                    out_loc, "{}/gt{}.png".format(i, j))
                data = np.swapaxes(f["train"]["images"][i][j], 0, 2)
                data = np.swapaxes(data, 0, 1)
                common.save_numpy_image(
                    array=data,
                    location=im_out_location
                )
                im_out_location = os.path.join(
                    out_loc, "{}/warp{}.png".format(i, j))
                data = np.swapaxes(f["train"]["warped"][i][j], 0, 2)
                data = np.swapaxes(data, 0, 1)
                common.save_numpy_image(
                    array=data,
                    location=im_out_location
                )

if __name__ == "__main__":
    file_location = os.path.join(
        "/home/sean/lf_datasets/test_sets/", "test_new.h5"
    )
    out_loc = "/home/sean/lf_datasets/seentest/"
    main(file_location, out_loc)
