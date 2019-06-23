import numpy as np
import os

def main(location):
    data = np.load(location)
    print(data)

if __name__ == "__main__":
    location = os.path.join(
        os.getcwd(), "example.npy"
    )
    main(location)