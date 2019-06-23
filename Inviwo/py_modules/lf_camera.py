import os
from time import sleep, time
import sys

import inviwopy
from inviwopy.glm import vec3, normalize
import ivw.utils as inviwo_utils

import numpy as np

import ivw_helpers
import welford

class LightFieldCamera:
    def __init__(self, look_from, look_to, look_up = vec3(0.0, 1.0, 0.0),
                 interspatial_distance = 1.0,
                 spatial_rows = 8, spatial_cols = 8):
        """Create a light field camera array.

        Keyword arguments:
        look_from, look_to, look_up -- vectors for top left cam (default up y)
        interspatial_distance -- distance between cameras in array (default 1.0)
        spatial_rows, spatial_cols -- camera array dimensions (default 8 x 8)
        """
        self.set_look(look_from, look_to, look_up)
        self.spatial_rows = spatial_rows
        self.spatial_cols = spatial_cols
        self.interspatial_distance = interspatial_distance

    def set_look(self, look_from, look_to, look_up):
        """Set the top left camera to look_from, look_to and look_up"""
        self.look_from = look_from
        self.look_to = look_to
        self.look_up = look_up

    def get_row_col_number(self, index):
        row_num = index // self.spatial_cols
        col_num = index % self.spatial_cols
        return row_num, col_num

    def __str__(self):
        lf_string = ("baseline;{}\n").format(self.interspatial_distance)
        lf_string += ("grid_rows;{}\n").format(self.spatial_rows)
        lf_string += ("grid_cols;{}\n").format(self.spatial_rows)
        lf_string += ("look_from;{}\n").format(self.look_from)
        lf_string += ("look_to;{}\n").format(self.look_to)
        lf_string += ("look_up;{}").format(self.look_up)
        return lf_string

    def print_metadata(self, camera, pixel_size, file = sys.stdout):
        """
        Prints the metadata about a this grid with a camera

        Keyword arguments:
        camera -- input inviwo camera object
        pixel_size -- output image size (assumed x_dim = y_dim)
        file -- the file to print to (default sys.stdout)
        """
        print(self, end = '\n', file = file)
        print(cam_to_string(camera), end = '\n', file = file)
        print("pixels;{}".format(pixel_size), file = file)

    def view_array(self, cam=inviwopy.app.network.MeshClipping.camera,
                   save=False, save_dir=os.path.expanduser('~'),
                   should_time=False):
        """Move the inviwo camera through the array for the current workspace.

        Keyword arguments:
        cam -- the camera to move through the light field array
        save -- save the images to png files (default False)
        save_dir -- the main directory to save the png images to (default home)
        """
        if not os.path.isdir(save_dir):
            raise ValueError("save_dir is not a valid directory.")

        print("Viewing array for lf camera with:")
        print(self)
        # Save the current camera position
        prev_cam_look_from = cam.lookFrom
        prev_cam_look_to = cam.lookTo
        prev_cam_look_up = cam.lookUp
        
        cam.lookUp = self.look_up
        if should_time:
            start_time = time()
        for idx, val in enumerate(self.calculate_camera_array()):
            (look_from, look_to) = val
            cam.lookFrom = look_from
            cam.lookTo = look_to
            inviwo_utils.update()
            row_num, col_num = self.get_row_col_number(idx)

            if save:
                #Loop over canvases in the workspace
                canvases = inviwopy.app.network.canvases
                same_names = any(
                    [(canvases[i].displayName == canvases[i + 1].displayName)
                    for i in range(len(canvases) - 1)]
                )
                #The identifier names could be different
                identifier_same_names = False
                if same_names:
                    identifier_same_names = any(
                    [(canvases[i].identifier == canvases[i + 1].identifier)
                    for i in range(len(canvases) - 1)]
                    )
                for canvas_idx, canvas in enumerate(canvases):
                    if same_names and identifier_same_names:
                        file_name = ('Canvas_'
                                  + str(canvas_idx)
                                  + '_'
                                  + str(row_num)
                                  + str(col_num)
                                  + '.png')
                    elif identifier_same_names:
                        file_name = (canvas.displayName
                                  + str(row_num)
                                  + str(col_num)
                                  + '.png')
                    else:
                        file_name = (canvas.identifier
                                  + str(row_num)
                                  + str(col_num)
                                  + '.png')
                    if 'Depth' in file_name:
                        str_list = list(file_name)
                        str_list[-4:] = list('.npy')
                        file_name = ''.join(str_list)
                        full_save_dir = os.path.abspath(save_dir)
                        file_path = os.path.join(full_save_dir, file_name)
                        print('Saving to: ' + file_path)
                        np.save(
                            file_path,
                            np.flipud(np.transpose(canvas.image.depth.data)),
                            fix_imports=False)
                    else:
                        full_save_dir = os.path.abspath(save_dir)
                        file_path = os.path.join(full_save_dir, file_name)
                        print('Saving to: ' + file_path)
                        canvas.snapshot(file_path)
            else:
                #Smooths the viewing process
                if not should_time:
                    print('Viewing position ({}, {})'.format(row_num, col_num))
                    sleep(0.1)

        canvas = inviwopy.app.network.canvases[0]
        pixel_dim = canvas.inputSize.dimensions.value[0]
        if save:
            metadata_filename = os.path.join(full_save_dir, 'metadata.csv')
            with open(metadata_filename, 'w') as f:
                self.print_metadata(cam, pixel_dim, f)

        # Reset the camera to original position
        time_taken = 0
        if should_time:
            time_taken = time() - start_time
            print("Overall time taken to render grid was {:4f}".format(
                time_taken))
        print()
        cam.lookFrom = prev_cam_look_from
        cam.lookTo = prev_cam_look_to
        cam.lookUp = prev_cam_look_up
        return time_taken

    def save_to_hdf5_group(self, sample_index, h5_canvas_list, config):
        """Saves lf information to hdf5 group with images over columns then rows
        
        Keyword arguments:
        h5_canvas_list - tuples of groups and canvas numbers in a list
                         for example (colour_group, 0)
        """
        overall_start_time = time()
        cam = inviwopy.app.network.MeshClipping.camera
        cam.lookUp = self.look_up
        
        accumulator_list = []
        # Gathers accumulators for welford mean and var
        for i, group_tuple in enumerate(h5_canvas_list):
            group = group_tuple[0]
            mean_shape = (1, ) + group["mean"].shape[1:]
            mean = np.zeros(mean_shape, np.float32)
            accumulator = (0, mean, 0)
            accumulator_list.append(accumulator)

        for idx, val in enumerate(self.calculate_camera_array()):
            (look_from, look_to) = val
            start_time = time()
            cam.lookFrom = look_from
            cam.lookTo = look_to
            inviwo_utils.update()
            time_taken = time() - start_time
            #row_num, col_num = self.get_row_col_number(idx)

            for i, group_tuple in enumerate(h5_canvas_list):
                #assuming that there is one canvas
                group = group_tuple[0]
                canvas = inviwopy.app.network.canvases[group_tuple[1]]
                im_data = ivw_helpers.get_image(canvas)

                # Inviwo stores data in a different indexing to regular
                im_data = np.flipud(np.swapaxes(im_data, 0, 2))
                im_data = im_data[::-1, ::-1, ...]
                assert (canvas.inputSize.customInputDimensions.value[0] ==
                    group.attrs["lf_shape"][-1]), \
                    "canvas size x {} is not pixel dimension {} of h5".format(
                        canvas.inputSize.customInputDimensions.value[0],
                        group.attrs["lf_shape"][-1]
                    )
                assert (canvas.inputSize.customInputDimensions.value[1] == 
                    group.attrs["lf_shape"][-2]), \
                    "canvas size y {} is not pixel dimension {} of h5".format(
                        canvas.inputSize.customInputDimensions.value[1],
                        group.attrs["lf_shape"][-2]
                    )
                group['images'][sample_index, idx] = im_data
                group['timing'][sample_index, idx] = time_taken
                group['camera_extrinsics'][sample_index, idx] = \
                [cam.lookFrom.x, cam.lookFrom.y, cam.lookFrom.z,
                cam.lookTo.x, cam.lookTo.y, cam.lookTo.z,
                cam.lookUp.x, cam.lookUp.y, cam.lookUp.z] 
                accumulator_list[i] = welford.update(
                    accumulator_list[i], np.asarray(im_data, dtype=np.float32))
        
        for i, group_tuple in enumerate(h5_canvas_list):
            group = group_tuple[0]
            mean, var, _ = welford.finalize(accumulator_list[i])
            group['mean'][sample_index, :, :, :] = mean
            group['var'][sample_index, :, :, :] = var

        print("Finished writing LF {0} in {1:.2f} to {2}".format(
            sample_index, time() - overall_start_time, h5_canvas_list[0][0].name
        ))

    def get_look_right(self):
        """Get the right look vector for the top left camera"""
        view_direction = self.look_to - self.look_from
        right_vec = normalize(cross_product(view_direction, self.look_up))
        return right_vec

    def calculate_camera_array(self):
        """Returns list of (look_from, look_to) tuples for the camera array"""
        look_list = []

        row_step_vec = normalize(self.look_up) * self.interspatial_distance
        col_step_vec = self.get_look_right() * self.interspatial_distance

        #Start at the top left camera position
        for i in range(self.spatial_rows):
            row_movement = row_step_vec * (-i)
            row_look_from = self.look_from + row_movement
            row_look_to = self.look_to + row_movement

            for j in range(self.spatial_cols):
                col_movement = col_step_vec * j
                cam_look_from = row_look_from + col_movement
                cam_look_to = row_look_to + col_movement
                
                look_list.append((cam_look_from, cam_look_to))

        return look_list

def cross_product(vec_1, vec_2):
    result =  vec3(
        (vec_1.y * vec_2.z) - (vec_1.z * vec_2.y),
        (vec_1.z * vec_2.x) - (vec_1.x * vec_2.z),
        (vec_1.x * vec_2.y) - (vec_1.y * vec_2.x))
    return result

def cam_to_string(cam):
    """Returns some important Inviwo camera properties as a string"""
    cam_string = ("near;{:8f}\n").format(cam.nearPlane)
    cam_string += ("far;{:8f}\n").format(cam.farPlane)
    cam_string += ("focal_length;{:8f}\n"
                  .format(cam.projectionMatrix[0][0]))
    cam_string += ("fov;{}").format(cam.fov)
    return cam_string