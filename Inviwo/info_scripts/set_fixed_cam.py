#Inviwo Python script 
import inviwopy

from inviwopy.glm import vec3

app = inviwopy.app
network = app.network

cam = inviwopy.app.network.MeshClipping.camera
print(cam.lookFrom)
print(cam.lookUp)
print(cam.lookTo)
cam.lookFrom = vec3(20,-70,-90)
cam.lookTo = vec3(0)
cam.lookUp = vec3(0, -1, 0)