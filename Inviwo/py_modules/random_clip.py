from inviwopy.glm import ivec2
from random import randint, random

def random_subset(network, type):
    """type is expected to be "X", "Y" or "Z" """
    if type is "X":
        clip = network.VolumeSubset.rangeX
    if type is "Y":
        clip = network.VolumeSubset.rangeY
    if type is "Z":
        clip = network.VolumeSubset.rangeZ
    max = (clip.rangeMax // 2) - (clip.rangeMax // 10)
    start = randint(0, max)
    end = randint(0, max)
    end = clip.rangeMax - end
    clip_range = ivec2(start, end)
    clip.value = clip_range

def restore_volume(network, type):
    """type is expected to be "X", "Y" or "Z" """
    if type is "X":
        clip = network.VolumeSubset.rangeX
    if type is "Y":
        clip = network.VolumeSubset.rangeY
    if type is "Z":
        clip = network.VolumeSubset.rangeZ
    clip.value = ivec2(0, clip.rangeMax)
    
def random_clip(network, type):
    """type is expected to be "X", "Y" or "Z" """
    if type is "X":
        clip = network.CubeProxyGeometry.clipX
    if type is "Y":
        clip = network.CubeProxyGeometry.clipY
    if type is "Z":
        clip = network.CubeProxyGeometry.clipZ
    max = (clip.rangeMax // 2) - (clip.rangeMax // 10)
    start = randint(0, max)
    end = randint(0, max)
    end = clip.rangeMax - end
    clip_range = ivec2(start, end)
    clip.value = clip_range

def restore_clip(network, type):
    """type is expected to be "X", "Y" or "Z" """
    if type is "X":
        clip = network.CubeProxyGeometry.clipX
    if type is "Y":
        clip = network.CubeProxyGeometry.clipY
    if type is "Z":
        clip = network.CubeProxyGeometry.clipZ
    clip.value = ivec2(0, clip.rangeMax)

def random_clip_look_from(network, look_from):
    max_val, max_type = -1, None

    val = abs(look_from.x)
    if val > max_val:
        max_val, max_type = val, "X"

    val = abs(look_from.y)
    if val > max_val:
        max_val, max_type = val, "Y"

    val = abs(look_from.z)
    if val > max_val:
        max_val, max_type = val, "Z"

    random_clip(network, max_type)

    return max_val, max_type

def random_clip_cam(network, camera):
    return random_clip_look_from(network, camera.lookFrom)

def random_clip_lf(network, lf):
    """Randomly cips a volume for a given lf camera"""
    look_from = lf.look_from
    return random_clip_look_from(network, look_from)

def random_plane_clip(network, lf):
    mesh_clip = network.MeshClipping
    assert (mesh_clip != None), \
        "Please add a Mesh Clipping processor to use plane clipping"
    cam = network.MeshClipping.camera
    cam.setLook(lf.look_from, lf.look_to, lf.look_up)
    mesh_clip.alignPlaneNormalToCameraNormal.press()
    mesh_clip.getPropertyByIdentifier("movePointAlongNormal").value = True
    mesh_clip.getPropertyByIdentifier("moveCameraAlongNormal").value = False
    mesh_clip.getPropertyByIdentifier("clippingEnabled").value = True
    max_val = mesh_clip.getPropertyByIdentifier("pointPlaneMove").maxValue

    #TODO this div value is very dependent on volume at hand, 2 or 2.5 common
    div_value = 2.0
    rand_clip = (max_val / 10) + random() * (max_val * 9 / (div_value * 10))
    mesh_clip.getPropertyByIdentifier("pointPlaneMove").value = rand_clip

def random_plane_clip_cam(network, camera):
    mesh_clip = network.MeshClipping
    assert (mesh_clip != None), \
        "Please add a Mesh Clipping processor to use plane clipping"
    cam = network.MeshClipping.camera
    cam.setLook(*camera)
    mesh_clip.alignPlaneNormalToCameraNormal.press()
    mesh_clip.getPropertyByIdentifier("movePointAlongNormal").value = True
    mesh_clip.getPropertyByIdentifier("moveCameraAlongNormal").value = False
    mesh_clip.getPropertyByIdentifier("clippingEnabled").value = True
    max_val = mesh_clip.getPropertyByIdentifier("pointPlaneMove").maxValue

    #TODO this div value is very dependent on the volume at hand, 2 common
    div_value = 2.0
    rand_clip = (max_val / 10) + random() * (max_val * 9 / (div_value * 10))
    mesh_clip.getPropertyByIdentifier("pointPlaneMove").value = rand_clip
