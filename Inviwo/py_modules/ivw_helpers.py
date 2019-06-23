"""
A set of functions to help wrap some common Inviwopy usages
"""
import inviwopy
import ivw.utils as inviwo_utils
from inviwopy.glm import ivec2

app = inviwopy.app
network = app.network

def get_canvas_image():
    """ 
    Returns the colour image from the first canvas in
    the Inviwo network with values in 0 - 255 RGBA
    """

    canvas = network.canvases[0]
    return canvas.image.colorLayers[0].data

def get_image(canvas):
    """
    Returns the colour image from the canvas in
    the Inviwo network with values in 0 - 255 RGBA
    """
    return canvas.image.colorLayers[0].data

def set_canvas_sizes(pixel_size_x, pixel_size_y):
    """Sets the canvases in the network to pixel size (square image)"""

    canvases = network.canvases
    for canvas in canvases:
        canvas.inputSize.dimensions.value = ivec2(pixel_size_x, pixel_size_y)
    inviwo_utils.update()

def get_canvas_by_id(network, id):
    """Returns the canvas in the network with the given id"""
    
    canvases = network.canvases
    for canvas in canvases:
        if canvas.identifier == id:
            return canvas
    print("No canvas found with name {}", id)
    return None

if __name__ == "__main__":
    print(get_canvas_image())
