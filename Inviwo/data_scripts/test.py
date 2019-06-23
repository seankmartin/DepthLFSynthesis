#Inviwo Python script 
import inviwopy
import numpy as np

app = inviwopy.app
network = app.network

d = network.canvases[0].image.colorLayers[0].data
print(d.shape)
non_zero = np.nonzero(d)
print(max(non_zero[1]))

print((d[200, 2033, :]))

d_flip = np.flipud(np.swapaxes(d, 0, 1))

#print(d_flip[225])

d_col = np.swapaxes(d, 0, 2)
print(d_col[:, ::-1, :][:, 0, 200])
print (d_col[:, ::-1, :].shape)

config = {
        "constant_seed": True,
        "pixel_dim_x": 408,
        "pixel_dim_y": 226,
        "clip": False,
        "plane": True,
        "num_samples": {'train': 10, 'val': 10},
        "hdf5_name": "chest_clipped.h5",
        "baseline": 1.0,
        "spatial_rows": 9,
        "spatial_cols": 9,
    }

idx = 44
y_start = config["pixel_dim_y"] * (idx // 5)
y_end = y_start + (config["pixel_dim_y"])                
x_start = config["pixel_dim_x"] * (idx % 5)                
x_end = x_start + (config["pixel_dim_x"])
print(x_end)

im_data = d[x_start:x_end, y_start:y_end]
im_data = np.flipud(np.swapaxes(im_data, 0, 2))
im_data = im_data[:, ::-1, :]
print(im_data.shape)
print(im_data[:, 60, 5])