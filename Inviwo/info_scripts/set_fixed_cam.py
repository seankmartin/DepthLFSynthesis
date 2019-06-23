#Inviwo Python script 
import inviwopy

from inviwopy.glm import vec3

app = inviwopy.app
network = app.network

cam = inviwopy.app.network.MeshClipping.camera
print(cam.lookFrom)
print(cam.lookUp)
print(cam.lookTo)
cam.lookFrom = vec3(-0.3, -1, -0.3)
cam.lookTo = vec3(0)
cam.lookUp = vec3(0, 0, 1)