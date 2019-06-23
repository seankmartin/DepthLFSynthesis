import h5py
import os

def walk_dict(d,depth=0):
    for k,v in sorted(d.items(),key=lambda x: x[0]):
        spaces = ("  ")*depth
        if hasattr(v, "items"):
            print(spaces + ("%s" % k))
            walk_dict(v,depth+1)
        else:
            print(spaces + "%s %s" % (k, v))

def get_attrs(d, depth=0):
    for k,v in sorted(d.items(),key=lambda x: x[0]):
        spaces = ("  ")*depth
        if len(v.attrs) is not 0:
            print("{}{} has attrs {}".format(
                    spaces, k, list(v.attrs.items())
                ))
        if hasattr(v, "items"):
            get_attrs(v, depth=depth+1)

def main(file_location):
    with h5py.File(file_location, 'r', libver='latest') as f:
        walk_dict(f)
        print()
        get_attrs(f)
        print()

if __name__ == "__main__":
    file_location = os.path.join(
        "/home/sean/lf_datasets/test_sets/", "head_set_h5209.h5"
    )
    main(file_location)

