import inviwopy
from inviwopy.glm import ivec3, dvec2, mat4, vec4, vec3
from inviwopy.properties import IntVec3Property, FileProperty, InvalidationLevel, PropertySemantics, BoolProperty, FloatVec3Property, TransferFunctionProperty, ButtonProperty, IntProperty
from inviwopy.data import VolumeOutport, VolumeInport
from inviwopy.data import Volume, TransferFunction

from math import ceil
import numpy as np

from random import random

if not "outport" in self.outports:
    self.addOutport(VolumeOutport("outport"))

if not "volinport" in self.inports:
    self.addInport(VolumeInport("volinport"))

if not "num_peaks" in self.properties:
    self.addProperty(
        IntProperty("num_peaks", "Number of peaks in Transfer Function", 3, 1, 20)
    )

if not "start" in self.properties:
    self.addProperty(
        IntProperty("start", "Peak to start at", 3, 1, 20)
    )

if not "tf" in self.properties:
    self.addProperty(
        #TransferFunctionProperty("tf", "Transfer Function", TransferFunction(),# self.inports.volinport)
        TransferFunctionProperty("tf", "Transfer Function", TransferFunction())
    )

if not "button" in self.properties:
    self.addProperty(
        ButtonProperty("button", "Generate TF")
    )

def random_colour():
    return vec4(random(), random(), random(), 0.1 + random() * 0.7)

def make_peak(tf, point, colour):
    vol = self.getInport("volinport").getData()
    scaled_point = point / vol.dataMap.valueRange[1]
    tf.add(scaled_point, colour)
    black = vec4(0, 0, 0, 0)
    #TODO clamp to range 0 1
    left = scaled_point - 0.03
    right = scaled_point + 0.03
    tf.add(left, black)
    tf.add(right, black)

def calc_peaks():
    if not "volinport" in self.inports:
        print("inport error")
        print("Inviwo is trying to use the other python processor as self")
    else:
        vol = self.getInport("volinport").getData()
    vol_data = vol.data
    hist, bin_edges = np.histogram(
        vol_data, bins=20
    )
    pairs = []
    start = self.properties.start.value
    for i, val in enumerate(hist[start:]):
        idx = start - 1 + i
        pairs.append(
            (val, (bin_edges[idx] + bin_edges[idx + 1] / 2))
            )
    pairs.sort(key=lambda tup: tup[0], reverse=True)
    final_values = []
    for i in range(self.properties.num_peaks.value):
        final_values.append(pairs[2 * i][1])
    return final_values

def GenerateTF():
    peaks = calc_peaks()
    tf = self.properties.tf
    tf.clear()
    for point in peaks:
        colour = random_colour()
        make_peak(tf, point, colour)

self.properties.button.onChange(GenerateTF)

def process(self):
    #pass inport to outport
    self.outports.outport.setData(
        self.getInport("volinport").getData()
    )

def initializeResources(self):
    pass

# Tell the PythonScriptProcessor about the 'initializeResources' function we want to use
self.setInitializeResources(initializeResources)

# Tell the PythonScriptProcessor about the 'process' function we want to use
self.setProcess(process)