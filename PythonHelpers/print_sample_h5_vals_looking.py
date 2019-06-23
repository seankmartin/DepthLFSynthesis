import os

import h5py
import numpy as np

import common

def main(file_location):
    with h5py.File(file_location, 'r') as f:
        sample_idx = 22
        base_path = "/home/sean/lf_datasets/images"
        for i in range(100):
            final_path = "test_im_looking" + str(i) + ".png"
            im_out_location = os.path.join(
                base_path, final_path)
            data = np.swapaxes(f["train/images"][i][sample_idx], 0, 2)
            data = np.swapaxes(data, 0, 1)
            common.save_numpy_image(data, im_out_location)   
        im_out_location = os.path.join(
            base_path, "swizzle.png")
        data = np.swapaxes(f["train/swizzles"][i], 0, 2)
        data = np.swapaxes(data, 0, 1)
        common.save_numpy_image(data, im_out_location) 


if __name__ == "__main__":
    file_location = os.path.join(
        "/home/sean/lf_datasets/test_sets/", "test_looking.h5"
    )
    main(file_location)