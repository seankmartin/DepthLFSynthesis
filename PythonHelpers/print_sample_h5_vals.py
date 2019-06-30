import os

import h5py
import numpy as np

import common

def main(file_location):
    with h5py.File(file_location, 'r') as f:
        for i in range(1):
            for j in range(64):
                im_out_location = os.path.join(
                    "/home/sean/lf_datasets/full_test1/test_warped" + str(j) + ".png")
                data = np.swapaxes(f["train"]["images"][i][j], 0, 2)
                data = np.swapaxes(data, 0, 1)
                common.save_numpy_image(
                    array=data,
                    location=im_out_location
                )

if __name__ == "__main__":
    file_location = os.path.join(
        "/home/sean/lf_datasets/test_sets/", "test_new.h5"
    )
    main(file_location)