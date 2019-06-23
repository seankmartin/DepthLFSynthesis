#Inviwo Python script 
import inviwopy

import sys
# Location of py_modules folder
ivw_custom_py_modules_location = "/home/sean/LF_Volume_Synthesis/Inviwo/py_modules"
sys.path.append(ivw_custom_py_modules_location)

from modify_transfer_func import modify_tf

app = inviwopy.app
network = app.network

tf = network.VolumeRaycaster.isotfComposite.transferFunction

modify_tf(tf)