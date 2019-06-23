#Inviwo Python script 
import inviwopy
import ivw.utils as inviwo_utils
from time import time

app = inviwopy.app
network = app.network

cam = network.MeshClipping.camera

lg = network.LookingGlassEntryExitPoints
lg.individual_view.value = True

start_time = time()
for i in range(45):
    lg.view.value = i
    inviwo_utils.update()

print(time() - start_time)

start_time = time()
lg.individual_view.value = False
inviwo_utils.update()
print(time() - start_time)