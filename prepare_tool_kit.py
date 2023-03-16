from ctypes import sizeof
from lib2to3.pytree import Base
from re import template
from typing import ValuesView
from webbrowser import get
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
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import json
import open3d as o3d
import pymeshlab
from subprocess import run


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
        self.tool_types = [tool['type'] for tool in tool_info]
                        # ['hammer', 'plier', 'wrench', 'screwdriver']
        self.tool_length_range = [tool['length'] for tool in tool_info]
                             #[[0.19, 0.2], [0.15, 0.18], [0.15, 0.18], [0.15, 0.18]]
        self.n_objects = [1, 1, 1, 1]

        self.mode = mode
        self.data_path = data_path
        self.output_path = output_path
        self.kit_size = kit_size
        self.voxel_size = voxel_size
        self.image_size = image_size
        self.homogeneous = False


    def reset(self, iteration=0):
        env = BaseEnv(gui=True)
        p.setGravity(0,0,0)

        # create output folder to store kit info
        kit_folder = self.output_path / f'{iteration:04d}'
        kit_folder.mkdir(parents=True, exist_ok=True)

        # Setup an orthographic camera
        focal_length = 64e4
        z=-200
        view_matrix = p.computeViewMatrix((0, 0, z), (0, 0, 0), (1, 0, 0))
        camera = SimCameraBase(view_matrix, self.image_size,
                           z_near=-z-0.1, z_far=-z+0.1, focal_length=focal_length)

        # Select objects
        obj_shapes = [] # e.g. [[7], [3], [3], [9]]
        obj_lengths = []
        for i in range(len(self.tool_types)):
            # comppute the target length
            scale = np.random.uniform(low=0, high=1)
            length = self.tool_length_range[i][0] + (self.tool_length_range[i][1] - self.tool_length_range[i][0]) * scale
            obj_lengths.append(length)

            if self.mode == 'train':
                obj_shape = np.random.choice(self.train_set, self.n_objects[i])
            else:
                if self.homogeneous:
                    obj_shape = [np.random.choice(self.test_set, self.n_objects[i])] * 1
                else:
                    obj_shape = np.random.choice(self.test_set, self.n_objects[i])
            obj_shapes.append(obj_shape)

        # Kit volume
        kit_vol_shape = np.ceil(self.kit_size / self.voxel_size).astype(np.int) + 2 # e.g. [280, 260, 50]
        kit_vol = np.zeros((kit_vol_shape))
        kit_vol_mask = np.ma.make_mask(kit_vol, shrink=False)
        # print('kit_vol_shape', kit_vol_shape)
        
        # Build Kit
        targ_pos = np.array([[0.03, 0.02, 0.0],     # hammer
                             [-0.03, 0.0, 0.0],     # plier
                             [-0.09, 0.0, 0.0],      # wrench
                             [0.09, 0.0, 0.0]])    # screwdriver
        gt_obj_pos = targ_pos

        obj_info_list = []
        for i, tool in enumerate(self.tool_types):
            for j in range(self.n_objects[i]):
                shape = self.data_path / tool / f'{obj_shapes[i][j]:02d}_coll.obj'
                print('Object:', shape)

                # load mesh
                mesh = o3d.io.read_triangle_mesh(str(shape))
                mesh = np.asarray(mesh.vertices)

                # compute and apply scale
                obj_scale = obj_lengths[i] / (mesh.max(axis=0) - mesh.min(axis=0))[1]
                mesh = mesh * obj_scale
                # print('obj_scale', obj_scale)

                # mesh = trimesh.load(shape, force='mesh')
                urdf_path = MeshRendererEnv.dump_obj_urdf(shape, rgba=np.array([0, 1, 0, 1]), 
                                                          load_collision_mesh=True, scale=obj_scale)
                object_id = p.loadURDF(str(urdf_path))
                urdf_path.unlink()

                # define the object bounds
                # mesh = trimesh.load(shape, force='mesh')
                # obj_bounds = np.zeros((3,2), dtype=np.float32)
                # obj_bounds[:,0] = mesh.vertices.min(axis=0)
                # obj_bounds[:,1] = mesh.vertices.max(axis=0)

                obj_bounds = np.zeros((3,2), dtype=np.float32)
                obj_bounds[:,0] = mesh.min(axis=0)
                obj_bounds[:,1] = mesh.max(axis=0)
                # print('obj_bounds', obj_bounds)


                delta = np.array([0.005, 0.005, 0.001]) # margin for each object direction in mm
                # obj_bounds[:2, 0] -= delta[0]
                # obj_bounds[:2, 1] += delta[1]
                # obj_bounds[2, 1] += delta[2]
                obj_bounds[:3, 0] -= delta[:3]
                obj_bounds[:3, 1] += delta[:3]
                # print('obj_bounds', obj_bounds)

                color_im, depth_im, _ = camera.get_image()
                part_tsdf = TSDFHelper.tsdf_from_camera_data(
                views=[(color_im, depth_im,
                    camera.intrinsics,
                    camera.pose_matrix)],
                    bounds=obj_bounds,
                    voxel_size=self.voxel_size,
                )
                p.removeBody(object_id)

                # print('tsdf_shape', part_tsdf.shape)

                # # Test and visualize the object tsdf 
                # object_tsdf_path = kit_folder/ f"{tool}_{j}.obj"
                # if TSDFHelper.to_mesh(part_tsdf, object_tsdf_path, self.voxel_size, vol_origin=[0, 0, 0]):
                #     print(object_tsdf_path) 
                # else:
                #     print(object_tsdf_path, ": kit generation failed")

                # Dilate the object cavity slightly to accommodate the object 
                mask3d = part_tsdf < 1.0

                pad = 15
                padding_size = np.array([[pad, pad], [pad, pad], [pad, pad]])
                # print(mask3d.shape)
                mask3d = np.pad(mask3d, padding_size, 'constant')
                # print(mask3d.shape)
                # mask3d = mask3d[pad:-pad, pad:-pad, pad:-pad]
                # print(mask3d.shape)

                diamond = ndi.generate_binary_structure(rank=3, connectivity=1)
                mask3d = ndi.binary_dilation(mask3d, diamond, iterations=10)
                mask3d = ndi.binary_erosion(mask3d, diamond, iterations=6)
                mask3d = mask3d[pad:-pad, pad:-pad, pad:-pad]

                occ_grid = np.zeros_like(part_tsdf)
                occ_grid[mask3d] = -1

                part_tsdf = -occ_grid

                print('part_tsdf:', part_tsdf.shape)
                print('kit_vol', kit_vol.shape)

                # Ok. Now wrap this volumes inside the proper kit volume
                kit_xyz = self.get_kit_xyz(targ_pos[i], obj_bounds, kit_vol_shape)
                kit_x = kit_xyz[0]
                kit_y = kit_xyz[1]
                kit_z = min(kit_xyz[2], part_tsdf.shape[2])
                # print(kit_xyz)

                # merge the kit vol
                kit_vol_mask[
                    kit_x: (kit_x + part_tsdf.shape[0]),
                    kit_y: (kit_y + part_tsdf.shape[1]),
                    -kit_z:,
                ] |= (part_tsdf[:, :, :kit_z] == 1.0)

                # record the ground truth of object position relative to the kit mesh frame
                gt_obj_pos[i, 2] = self.kit_size[2] - (obj_bounds[2,1] - obj_bounds[2,0]) / 2  #+ 0.001
                obj_info = {'type':tool, 'id':int(obj_shapes[i][j]), 
                            'pos': gt_obj_pos[i].tolist(), 'scale': obj_scale}
                obj_info_list.append(obj_info)


        # # Make the cavity slightly larger by perform binary closing globally (Operation shifts to local areas.)
        # diamond = ndi.generate_binary_structure(rank=3, connectivity=1)
        # kit_vol_mask = ndi.binary_dilation(kit_vol_mask, diamond, iterations=5)
        # kit_vol_mask = ndi.binary_erosion(kit_vol_mask, diamond, iterations=2, border_value=1)

        def project_cut(a):
            indice = np.where(a == True)
            if len(indice[0]) > 0:
                a[indice[0][-1]:] = True
            return a
        
        kit_vol_mask = np.apply_along_axis(project_cut, 2, kit_vol_mask)
        
        # Convert mask to tsdf volume
        kit_vol[kit_vol_mask] = 1
        kit_vol[kit_vol==0.0] = -1

        # Save the mesh
        kit_temp_path = kit_folder/ "kit_tmp.obj"    
        if TSDFHelper.to_mesh(kit_vol, kit_temp_path, self.voxel_size, 
                                vol_origin=[self.voxel_size * 0.5, 
                                            self.voxel_size * 0.5,
                                            self.voxel_size * ((kit_vol_shape[2]-1)/2)] ): # Set the kit bottom as the zero position 
            print('Save TSDF:', kit_temp_path) 
        else:
            print(shape, ": kit generation failed")

        # Smoothen the mesh
        # mesh = trimesh.load(kit_mesh_path, force='mesh')
        # trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu= 0.0, iterations=10, laplacian_operator=None)
        # kit_mesh_smooth = kit_folder/"kit.obj"
        # mesh.export(kit_mesh_smooth)
        # kit_mesh_path.unlink()
        # kit_path = kit_folder/"kit.obj"
        # mesh = o3d.io.read_triangle_mesh(str(kit_temp_path))
        # mesh = mesh.filter_smooth_taubin(number_of_iterations=10, lambda_filter=0.5, mu=0.0)
        # o3d.io.write_triangle_mesh(str(kit_path), mesh)
        # kit_temp_path.unlink()

        # mesh simplification
        # kit_coll_path = kit_folder / "kit_coll.obj"
        # meshlab_script = self.data_path / "simplify_quadric_decimation.mlx"
        # proc = run(['meshlabserver', '-i', str(kit_path), '-o', str(kit_coll_path), '-s', str(meshlab_script)])
        # print('exit status code', proc.returncode)
        # print('kit coll', kit_coll_path)
        # mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=20000)
        # o3d.io.write_triangle_mesh(str(kit_mesh_smooth), mesh)

        # pymeshlab
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(kit_temp_path))
        
        # Smoothen the mesh
        kit_path = kit_folder/"kit.obj"
        ms.apply_coord_taubin_smoothing(lambda_=0.5, mu=0.0, stepsmoothnum=10)
        ms.save_current_mesh(str(kit_path))

        # Simplify the mesh
        kit_coll_path = kit_folder / "kit_coll.obj"
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=0,
                                                    targetperc=0.01,
                                                    qualitythr=0.3,
                                                    preserveboundary=True,
                                                    boundaryweight=1,
                                                    preservenormal=False,
                                                    preservetopology=True,
                                                    optimalplacement=True,
                                                    planarquadric=True,
                                                    )
        ms.save_current_mesh(str(kit_coll_path))
        
        # TODO generate collison model
        # collision_path = kit_mesh_path.parent / (kit_mesh_path.name[:-4] + '_coll.obj')
        # name_log = kit_mesh_path.parent / (kit_mesh_path.name[:-4] + '_log.txt')
        # p.vhacd(str(kit_mesh_path), str(collision_path), str(name_log), alpha=0.04,resolution=50000 )
        # name_log.unlink()
        # kit_mesh_path.unlink()

        # save the ground truth value
        self.save(kit_folder, obj_info_list)
        
        # delete the temp mesh
        kit_temp_path.unlink()
        
        return

    def get_kit_xyz(self, obj_center, obj_bound, kit_shape):
        kit_xyz = np.zeros((3,))
        kit_xyz[:2] = (obj_center + self.kit_size / 2 + obj_bound[:,0])[:2] # x, y
        kit_xyz[2] = obj_bound[2,1] - obj_bound[2, 0] # z
        kit_xyz = np.ceil(kit_xyz / self.voxel_size).astype(int)
        return kit_xyz
    
    def save(self, path, data):
        data_path = path / f'info.json'
        with open(data_path, 'w') as json_file:
            json.dump(data, json_file)      


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
    print(tool_info)

    # get object paths
    mode = cfg.mode
    data_path = Path(cfg.root_data_path)
    output_path = Path(cfg.output_data_path) / f'kit_{mode}'
    output_path.mkdir(parents=True, exist_ok=True)

    # get parameters
    n_samples = cfg.n_samples
    image_size = np.array(cfg.image_size)
    voxel_size = cfg.voxel_size
    kit_size = np.array(cfg.kit_size)

    # fix the seed
    seed = 0 if mode == 'train' else 1
    np.random.seed(seed)

    # initilize the kit generator 
    kit_gen = KitGenerator(mode, data_path, output_path, tool_info, 
                            kit_size, voxel_size, image_size)

    # generate the kits
    for i in range(n_samples):
        print('Generate Kit', i)
        kit_gen.reset(i)
    return


if __name__ == "__main__":
    main()