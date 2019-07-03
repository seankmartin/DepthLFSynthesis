import os
import math

import h5py
import numpy as np

import argparse
import evaluate
import welford

import common
def main(args):
    with h5py.File(args.loc, 'r') as f:
        overall_psnr_accum = (0, 0, 0)
        overall_ssim_accum = (0, 0, 0)
        i = args.n
        psnr_accumulator = (0, 0, 0)
        ssim_accumulator = (0, 0, 0)
        group1 = f["train"]["images"][i, :, :args.channels, ...]
        group2 = f["train"]["warped"][i, :, :args.channels, ...]
        for j in range(64):
            im1_out_location = os.path.join(
                args.out_loc, "{}/gt{}.png".format(i, j))
            im2_out_location = os.path.join(
                args.out_loc, "{}/warp{}.png".format(i, j))
            im1 = group1[j]
            im2 = group2[j]
            im1 = np.swapaxes(im1, 0, 2)
            im1 = np.swapaxes(im1, 0, 1)
            im2 = np.swapaxes(im2, 0, 2)
            im2 = np.swapaxes(im2, 0, 1)
            common.save_numpy_image(
                array=im1,
                location=im1_out_location
            )
            common.save_numpy_image(
                array=im2,
                location=im2_out_location
            )
            psnr = evaluate.psnr(im1, im2)
            ssim = evaluate.ssim(im1, im2)
            psnr_accumulator = welford.update(psnr_accumulator, psnr)
            ssim_accumulator = welford.update(ssim_accumulator, ssim)
            if args.verbose:
                print("{};{};PSNR {:4f};SSIM {:4f}".format(
                    i, j, psnr, ssim))
        psnr_mean, psnr_var, _ = welford.finalize(psnr_accumulator)
        ssim_mean, ssim_var, _ = welford.finalize(ssim_accumulator)
        if args.verbose:
            print()
        print(
            "{};psnr average {:5f};stddev {:5f}".format(
                i, psnr_mean, math.sqrt(psnr_var)) +
            ";ssim average {:5f};stddev {:5f}".format(
                ssim_mean, math.sqrt(ssim_var)))
        overall_psnr_accum = welford.update(
            overall_psnr_accum, psnr_mean)
        overall_ssim_accum = welford.update(
            overall_ssim_accum, ssim_mean)


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description="modifiable params")
    PARSER.add_argument("--loc", type=str, default=None,
                        help="hdf5 path for comparison")
    PARSER.add_argument("--n", type=int, default=0,
                        help="sample to print")
    PARSER.add_argument("--verbose", "-v", action="store_true",
                        help="Whether to print many values")
    PARSER.add_argument("--group", type=str, default="train",
                        help="hdf5 group to get images from")
    PARSER.add_argument("--channels", "-c", type=int, default=3,
                        help="How many channels to use in the image")
    PARSER.add_argument("--out_loc", type=str, default=None,
                        help="Where to save images")
    ARGS, _ = PARSER.parse_known_args()
    main(ARGS)