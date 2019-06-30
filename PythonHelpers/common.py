import os
from os import listdir
from os.path import isfile, isdir, join

from PIL import Image
import numpy as np

def decompose(arr, out_shape, y_dim):
    sample = np.empty(out_shape)
    for j in range(y_dim):
        for i in range(out_shape[0] // y_dim):
            start_x = out_shape[1] * j
            end_x = out_shape[1] * (j + 1)
            start_y = out_shape[2] * i
            end_y = out_shape[2] * (i + 1)
            sample[i + j * y_dim] = arr[start_x:end_x, start_y:end_y]
    return sample

def save_numpy_image(array, location):
    """Saves numpy image in 0 - 255 using PIL at location"""
    # Save a sample image
    make_dir_if_not_exists(location)
    im = Image.fromarray(array.astype(np.uint8))
    im.save(location)

def make_dir_if_not_exists(location):
    """Makes directory structure for given location"""
    os.makedirs(os.path.dirname(location), exist_ok=True)

def has_ext(filename, ext):
    """Check if the filename ends in the extension"""
    if ext == None:
        return True
    if ext[0] != ".":
        ext = "." + ext
    return filename[-len(ext):].lower() == ext.lower()

def get_all_files_in_dir(in_dir, ext=None):
    """Gets all files in the directory with the given extension"""
    if not isdir(in_dir):
        print("Non existant directory " + str(in_dir))
        return []
    ok_file = lambda f : isfile(join(in_dir, f)) and has_ext(f, ext)
    onlyfiles = [f for f in listdir(in_dir) if ok_file(f)]
    return onlyfiles

def do_tests():
    results = []
    passed = has_ext("hahahah.jpg", "jpg")
    results.append(passed)
    passed = has_ext("Billboghahahah.jpG", "jpg")
    results.append(passed)
    passed = has_ext("hahahah.jpg", ".jpg")
    results.append(passed)
    passed = has_ext("hahahah.j.pg", ".jpg")
    results.append(passed)
    return results

if __name__ == "__main__":
    print(do_tests())
    home = os.path.expanduser('~')
    print(get_all_files_in_dir(home+ "banaa", "ini"))
