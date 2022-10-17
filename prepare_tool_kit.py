from ctypes import sizeof
from lib2to3.pytree import Base
from re import template
from typing import ValuesView
from utils import get_ray_fn, init_ray
import hydra
from omegaconf import DictConfig
from pathlib import Path
from environment.baseEnv import BaseEnv
from environment.camera import SimCameraBase
import trimesh
import numpy as np
from environment.meshRendererEnv import MeshRendererEnv
from utils.tsdfHelper import TSDFHelper
import pybullet as p
from skimage.morphology import label
import ray
from os.path import exists
from scipy import ndimage
import matplotlib.pyplot as plt

def generate_tool_kit(obj_paths, output_path, tool_info, kit_size, voxel_size, image_size):
    """ Generate tool kits"""
    
    env = BaseEnv(gui=True)
    p.setGravity(0,0,0)

    # Setup an orthographic camera
    focal_length = 64e4
    z=-200
    view_matrix = p.computeViewMatrix((0, 0, z), (0, 0, 0), (1, 0, 0))
    camera = SimCameraBase(view_matrix, image_size,
                           z_near=-z-0.05, z_far=-z+0.05, focal_length=focal_length)

    # Sample objects randonly
    tools = ['hammer', 'plier', 'screwdriver', 'wrench']
    n_objects = [1, 1, 1, 1]
    obj_shapes = []
    




    return


class KitGenerator():
    def __init__(self, 
                 mode,
                 data_path, 
                 output_path, 
                 tool_info, 
                 kit_size, 
                 voxel_size, 
                 image_size):
        # tool relatived
        self.train_set = list(np.arange(0, 10))
        self.test_set = list(np.arange(10, 15))
        self.tools = ['hammer', 'plier', 'screwdriver', 'wrench']
        self.targ_length = [0.2, 0.18, 0.18, 0.18]
        self.n_objects = [1, 1, 1, 1]

        self.data_path = data_path
        self.output_path = output_path
        self.tool_info = tool_info
        self.kit_size = kit_size
        self.voxel_size = voxel_size
        self.image_size = image_size
        self.mode = mode
        self.homogeneous = False


    def reset(self):
        env = BaseEnv(gui=True)
        p.setGravity(0,0,0)

        # Setup an orthographic camera
        focal_length = 64e4
        z=-200
        view_matrix = p.computeViewMatrix((0, 0, z), (0, 0, 0), (1, 0, 0))
        camera = SimCameraBase(view_matrix, self.image_size,
                           z_near=-z-0.05, z_far=-z+0.05, focal_length=focal_length)

        # Select objects
        obj_shapes = []
        for i, tool in enumerate(self.tools):
            if self.mode == 'train':
                obj_shape = np.random.choice(self.train_set, self.n_objects[i])
            else:
                if self.homogeneous:
                    obj_shape = [np.random.choice(self.test_set, self.n_objects[i])] * 1
                else:
                    obj_shape = np.random.choice(self.test_set, self.n_objects[i])
            obj_shapes.append(obj_shape)

        kit_vol_shape = np.ceil(self.kit_size / self.voxel_size).astype(np.int)
        print('kit_vol_shape', kit_vol_shape)
        
        # Build Kit
        targ_pos = [[-0.02, 0.03, 0.0],
                    [0.02, -0.03, 0.0],
                    [0.02, 0.09, 0.0],
                    [-0.02, -0.09, 0.0]]
        for i, tool in enumerate(self.tools):
            for j in range(self.n_objects[i]):
                shape = self.data_path / tool / f'{obj_shapes[i][j]}' / 'object.obj'
                print(shape)
                
                mesh = trimesh.load(shape, force='mesh')
                # Center mesh around origin
                scale_factor = self.targ_length / (mesh.vertices.max(axis=0)-mesh.vertices.min(axis=0))[1]
                mesh.vertices *= scale_factor
                mesh.vertices[:, 0:2] -= ( (mesh.vertices.max(axis=0) + mesh.vertices.min(axis=0)) / 2)[0:2]
                mesh.vertices[:, 2] -= mesh.vertices.min(axis=0)[2] + 0

                # Scale mesh to fit the desired obj_bounds precisely



        
        

        

        


        


@hydra.main(config_path="conf/data_gen", config_name="tool_kit")
def main(cfg: DictConfig):

    # def get_obj_paths(data_path, tool_size):
    #     """Find the folder path of object"""
    #     obj_paths = dict()
    #     for tool, num in tool_size.items():
    #         tool_path = data_path / tool
    #         obj_paths[tool] = []
    #         for i in range(num):
    #             obj_path = tool_path / str(i)
    #             obj_path = obj_path / f"object.obj"
    #             obj_paths[tool].append(obj_path)
    #             print(obj_path)
    #     return obj_paths

    # get object info
    tool_info = cfg.tool_info
    tool_size = {tool['type']:tool['num'] for tool in tool_info}
    # print(tool_info)

    # get object paths
    mode = cfg.mode
    data_path = Path(cfg.root_data_path)
    output_path = Path(cfg.output_data_path)

    # get parameters
    n_samples = cfg.n_samples
    image_size = np.array(cfg.image_size)
    voxel_size = cfg.voxel_size
    kit_size = np.array(cfg.kit_size)

    # for sample_id in len(n_samples):
        # output_folder = output_path / f'{sample_id:04d}' #e.g. 000
        # output_folder.mkdir(parents=True, exist_ok=True)

        # print("Iteration: ", sample_id)
        # generate_tool_kit(obj_paths, output_folder, tool_info, kit_size, voxel_size, image_size)
    kit_gen = KitGenerator(mode, data_path, output_path, tool_info, 
                            kit_size, voxel_size, image_size)

    for i in range(n_samples):
        kit_gen.reset()


    return


if __name__ == "__main__":
    main()