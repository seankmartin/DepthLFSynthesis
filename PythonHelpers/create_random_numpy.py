import numpy as np
import os

def main(save_location, dim):
    np.random.seed(546465)
    volume = np.random.rand(dim[0], dim[1], dim[2]).astype(np.float32)
    print(volume)
    np.save(save_location, volume)

if __name__ == "__main__":
    save_location = os.path.join(os.getcwd(), "example.npy")
    main(save_location, dim=(256, 256, 256))