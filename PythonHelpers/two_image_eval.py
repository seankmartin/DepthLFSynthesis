import argparse

import numpy as np
from PIL import Image

import image_warping
import evaluate

def main(args):
    if args.im1 is None or args.im2 is None:
        print("Please enter two image paths through cmd")
        exit(-1)
    print("Comparing {} and {}".format(args.im1, args.im2))
    im1 = Image.open(args.im1)
    im2 = Image.open(args.im2)
    im1 = np.array(im1)
    im2 = np.array(im2)
    im_a1 = Image.fromarray(im1.astype(np.uint8))
    im_a1.save("t1.png")
    im_a2 = Image.fromarray(im2.astype(np.uint8))
    im_a2.save("t2.png")
    psnr = evaluate.my_psnr(im1, im2)
    ssim = evaluate.ssim(im1, im2)
    print("PSNR {:4f}, SSIM {:4f}".format(psnr, ssim))

    if args.save:
        diff = image_warping.get_diff_image(im1, im2)
        file_name = 'DifferenceImage.png'
        image_warping.save_array_as_image(diff, file_name)

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description="modifiable params")
    PARSER.add_argument("--im1", type=str, default=None,
                        help="first image path for comparison")
    PARSER.add_argument("--im2", type=str, default=None,
                        help="second image path for comparison")   
    PARSER.add_argument("--save", action="store_true",
                        help="Should save a difference image?")
    ARGS, _ = PARSER.parse_known_args()
    main(ARGS)