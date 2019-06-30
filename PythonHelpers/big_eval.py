import argparse
from os.path import join

import numpy as np
from PIL import Image

import image_warping
import evaluate

def main(args):
    if args.dir is None:
        print("Please enter a dir to compare in! use --dir flag")
        exit(-1)
    for i in range(64):
        loc1 = join(args.dir, "Warp{}.png".format(i))
        loc2 = join(args.dir, "Full{}.png".format(i))
        im1 = Image.open(loc1)
        im2 = Image.open(loc2)
        im1 = np.array(im1)
        im2 = np.array(im2)
        im_a1 = Image.fromarray(im1.astype(np.uint8))
        im_a2 = Image.fromarray(im2.astype(np.uint8))
        psnr = evaluate.my_psnr(im1, im2)
        ssim = evaluate.ssim(im1, im2)
        print("{}: PSNR {:4f}, SSIM {:4f}".format(i, psnr, ssim))


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description="modifiable params")
    PARSER.add_argument("--dir", type=str, default=None,
                        help="image path for comparison")
    ARGS, _ = PARSER.parse_known_args()
    main(ARGS)