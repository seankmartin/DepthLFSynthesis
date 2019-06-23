import inviwopy
from inviwopy.glm import ivec3, dvec2, mat4, vec4, vec3
from inviwopy.properties import IntVec3Property, FileProperty, InvalidationLevel, PropertySemantics, BoolProperty, FloatVec3Property, IntProperty
from inviwopy.data import VolumeOutport
from inviwopy.data import Volume
import numpy as np
np.set_printoptions(precision=2, suppress=True)

import os
import sys

import nibabel as nib

"""
The PythonScriptProcessor will run this script on construction and whenever this
it changes. Hence one needs to take care not to add ports and properties multiple times.
The PythonScriptProcessor is exposed as the local variable 'self'.
"""

if not "location" in self.properties:
    self.addProperty(
        FileProperty("location", "Numpy Volume Location", 
            os.path.join(os.getcwd(), "default_volume.npy"), 
            "npy",
            InvalidationLevel.InvalidOutput,
            PropertySemantics("PythonEditor")
        )
    )

if not "outport" in self.outports:
    self.addOutport(VolumeOutport("outport"))

if not "use_nifti" in self.properties:
    self.addProperty(
        BoolProperty("use_nifti", "Should load a Nifti", False, 
        InvalidationLevel.InvalidOutput,
        PropertySemantics("PythonEditor"))
    )

if not "use_scan_basis" in self.properties:
    self.addProperty(
        BoolProperty("use_scan_basis", "Use the scanned file basis", False, 
        InvalidationLevel.InvalidOutput,
        PropertySemantics("PythonEditor"))
    )

if not "basis" in self.properties:
    self.addProperty(
        FloatVec3Property("basis", "X, Y, Z basis scales", 
        vec3(1.0), vec3(0.0), vec3(1.0), vec3(0.001))
    )

if not "max" in self.properties:
    self.addProperty(
        IntProperty("max", "Defined data max", 600, 1, 1500)
    )

def process(self):
    """
    The PythonScriptProcessor will call this process function whenever the processor process 
    function is called. The argument 'self' represents the PythonScriptProcessor.
    """
    if self.properties.use_nifti.value == True:
        print("Loading nii data at " + self.properties.location.value)
        img = nib.load(self.properties.location.value)
        print(img.header)
        print(img.get_data_dtype())
        data = img.get_fdata()
    else:
        print("Loading numpy data at " + self.properties.location.value)
        data = np.load(self.properties.location.value)
    
    dim = data.shape
    max_desired_val = self.properties.max.value
    max_val = min(data.max(), max_desired_val)
    print("Max data value is {}, cut down to {}".format(
        data.max(), max_val))
    data = np.where(data > max_desired_val, np.zeros_like(data), data)
    # data = np.where(data < (max_val / 10), np.zeros_like(data), data)
    volume = Volume(data.astype(np.uint16))
    volume.dataMap.dataRange = dvec2(0.0, max_val)
    volume.dataMap.valueRange = dvec2(0.0, max_desired_val)
    self.outports.outport.setData(volume)

    # shifting the model by a set translation along all axes
    
    if self.properties.use_scan_basis.value == False:
        basis_vals = self.properties.basis.value
        volume.modelMatrix = mat4(
            basis_vals.x, 0, 0, 0,
            0, basis_vals.y, 0, 0,
            0, 0, basis_vals.z, 0,
            -(basis_vals.x / 2), -(basis_vals.y / 2), -(basis_vals.z / 2), 1
        )

    else:
        affine_trans = img.affine.transpose()
        volume.modelMatrix = mat4(*affine_trans.reshape(16))
    
    volume.worldMatrix = mat4(
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    )

def initializeResources(self):
    pass

# Tell the PythonScriptProcessor about the 'initializeResources' function we want to use
self.setInitializeResources(initializeResources)

# Tell the PythonScriptProcessor about the 'process' function we want to use
self.setProcess(process)