from ctypes import sizeof
from lib2to3.pytree import Base
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

def generate_tn_object(obj_path, obj_bounds, voxel_size, image_size, target_height):
    """ Generate object for TransporterNet
    """

    if not exists(obj_path):
        return

    env=BaseEnv(gui=True)
    p.setGravity(0,0,0)

    # Setup an orthographic camera:
    # (Using an orthographic camera removes the anglular volume when captured from only one image)
    focal_length = 63e4
    z=-200
    view_matrix = p.computeViewMatrix((0, 0, z), (0, 0, 0), (1, 0, 0))
    camera = SimCameraBase(view_matrix, image_size,
                           z_near=-z-0.05, z_far=-z+0.05, focal_length=focal_length)

    mesh = trimesh.load(obj_path, force='mesh')
    scale_factor = 0.25 / (mesh.vertices.max(axis=0)-mesh.vertices.min(axis=0))[1]
    mesh.vertices *= scale_factor
    mesh.vertices[:, 2] -= mesh.vertices.min(axis=0)[2] + 0
    processed_obj_path = obj_path.parent / f"obj.obj"
    mesh.export(processed_obj_path)

    urdf_path = MeshRendererEnv.dump_obj_urdf(processed_obj_path, 
                                              rgba=np.array([0, 1, 0, 1]),
                                              load_collision_mesh=True)
    p.loadURDF(str(urdf_path))
    color_im, depth_im, _ = camera.get_image()

    # generate the obj_bounds
    obj_bounds = np.zeros((3,2), np.float32)
    obj_bounds[:,0] = mesh.vertices.min(axis=0)
    obj_bounds[:,1] = mesh.vertices.max(axis=0)
    obj_bounds_mod = np.abs(obj_bounds).max(axis=1)
    obj_bounds[:2, 0] = -obj_bounds_mod[:2]
    obj_bounds[:2, 1] = +obj_bounds_mod[:2]

    # 
    delta = 0.01
    obj_bounds_zoomed = np.copy(obj_bounds)
    obj_bounds_zoomed[:2, 0] -= delta  # Only along x,y axis
    obj_bounds_zoomed[:2, 1] += delta

    print(obj_bounds)
    part_tsdf = TSDFHelper.tsdf_from_camera_data(
        views=[(color_im, depth_im,
                camera.intrinsics,
                camera.pose_matrix)],
        bounds=obj_bounds_zoomed,
        voxel_size=voxel_size,
    )

    # 1 indicates empty while -1 indicates occupied
    occ_grid = np.ones_like(part_tsdf)
    occ_grid[part_tsdf < 0] = -1
    occ_grid_proj = occ_grid.min(axis=2)
    print(occ_grid_proj.shape)

    height = target_height
    occ_grid = np.repeat(occ_grid_proj[:,:,np.newaxis], int(height/voxel_size), axis=2)
    # occ_grid[occ_grid == 0] = 1
    # occ_grid = -occ_grid

    tn_obj_path = obj_path.parent / "tn_obj.obj"
    if TSDFHelper.to_mesh(occ_grid, tn_obj_path, voxel_size, vol_origin=[0, 0, height/2]):
        print(tn_obj_path)
    else:
        print("Failed")

    return

@hydra.main(config_path="conf/data_gen", config_name="tn")
def main(cfg: DictConfig):

    tool_size = {"hammer":15}

    def get_obj_paths(data_path, tool_size):
        """Find the folder path of object"""
        obj_paths = []
        for tool, size in tool_size.items():
            data_path = data_path / tool
            for i in range(size):
                obj_path = data_path / str(i)
                obj_path = obj_path / f"object.obj"
                obj_paths.append(obj_path)
                print(obj_path)
        return obj_paths

    # get object paths
    data_path = Path(cfg.root_data_path)
    object_paths = get_obj_paths(data_path, tool_size)

    # get parameters
    image_size = np.array(cfg.image_size)
    voxel_size = cfg.voxel_size
    target_height = cfg.target_height

    for i in range(tool_size["hammer"]):
        generate_tn_object(object_paths[i], None, voxel_size, image_size, target_height)
    return


if __name__ == "__main__":
    main()