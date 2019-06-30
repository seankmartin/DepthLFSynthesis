import os

import h5py
import numpy as np

import argparse
import evaluate

def main(args):
    with h5py.File(args.loc, 'r') as f:
        for i in range(args.n):
            for j in range(64):
                im1 = np.swapaxes(
                    f["train"]["images"][i][j], 0, 2)
                im1 = np.swapaxes(im1, 0, 1)
                im2 = np.swapaxes(
                    f["train"]["warped"][i][j], 0, 2)
                im2 = np.swapaxes(im2, 0, 1)
                psnr = evaluate.my_psnr(im1, im2)
                ssim = evaluate.ssim(im1, im2)
                print("{}, {}: PSNR {:4f}, SSIM {:4f}".format(
                    i, j, psnr, ssim))

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description="modifiable params")
    PARSER.add_argument("--loc", type=str, default=None,
                        help="hdf5 path for comparison")
    PARSER.add_argument("--n", type=int, default=1,
                        help="num_samples")
    ARGS, _ = PARSER.parse_known_args()
    main(ARGS)