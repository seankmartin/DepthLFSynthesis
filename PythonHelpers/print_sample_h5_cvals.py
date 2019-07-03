import os

import h5py
import numpy as np

import common

def main(file_location, out_loc):
    with h5py.File(file_location, 'r') as f:
        for i in range(80):
            j = 28
            im_out_location = os.path.join(
                out_loc, "{}gt{}.png".format(i, j))
            data = np.swapaxes(f["unseen_vol"]["images"][i][j], 0, 2)
            data = np.swapaxes(data, 0, 1)
            common.save_numpy_image(
                array=data,
                location=im_out_location
            )
            im_out_location = os.path.join(
                out_loc, "{}warp{}.png".format(i, j))
            data = np.swapaxes(f["unseen_vol"]["warped"][i][j], 0, 2)
            data = np.swapaxes(data, 0, 1)
            common.save_numpy_image(
                array=data,
                location=im_out_location
            )

if __name__ == "__main__":
    file_location = os.path.join(
        "/home/sean/lf_datasets/test_sets/", "tiny_head_low.h5"
    )
    out_loc = "/home/sean/lf_datasets/all_centres_nvol/"
    main(file_location, out_loc)
