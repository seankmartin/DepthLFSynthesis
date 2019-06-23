import os

import h5py
import numpy as np

import common

def main(file_location):
    with h5py.File(file_location, 'r') as f:
        for i in range(2000):
            im_out_location = os.path.join(
                "/home/sean/lf_datasets/lf_images/test_train_im" + str(i) + ".png")
            central_idx = 36
            data = np.swapaxes(f["unseen_tf/images"][i][central_idx], 0, 2)
            data = np.swapaxes(data, 0, 1)
            common.save_numpy_image(
                array=data,
                location=im_out_location
            )

if __name__ == "__main__":
    file_location = os.path.join(
        "/home/sean/lf_datasets/test_sets/", "head_set.h5"
    )
    main(file_location)