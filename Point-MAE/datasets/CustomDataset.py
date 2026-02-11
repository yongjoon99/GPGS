import os
import torch
import numpy as np
import torch.utils.data as data
from .io import IO
from .build import DATASETS
from utils.logger import *
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
import glob
import random
from natsort import natsorted


def extract_sub_pointclouds_box(points, mask=None, min_points=2048, box_size=0.1, num_subclouds=50):

    sub_pointclouds = []

    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)

    for _ in range(num_subclouds):

        if mask is not None:
            masked_indices = np.where(mask)[0]
            seed_idx = np.random.choice(masked_indices)
        else:
            seed_idx = np.random.randint(len(points))
        seed_point = points[seed_idx]


        lower_bound = seed_point - box_size / 2
        upper_bound = seed_point + box_size / 2
        lower_bound = np.maximum(lower_bound, min_bound)
        upper_bound = np.minimum(upper_bound, max_bound)


        in_box_mask = np.all((points >= lower_bound) & (points <= upper_bound), axis=1)
        indices = np.where(in_box_mask)[0]


        current_box_size = box_size
        while len(indices) < min_points:
            current_box_size *= 1.5
            lower_bound = seed_point - current_box_size / 2
            upper_bound = seed_point + current_box_size / 2
            lower_bound = np.maximum(lower_bound, min_bound)
            upper_bound = np.minimum(upper_bound, max_bound)
            in_box_mask = np.all((points >= lower_bound) & (points <= upper_bound), axis=1)
            indices = np.where(in_box_mask)[0]

            if current_box_size > (max_bound - min_bound).max():
                break

        if len(indices) < min_points:
            continue 

        sub_pc = points[indices]
        sub_pointclouds.append(sub_pc)

    return sub_pointclouds



def extract_sub_pointclouds(points, mask = None, min_points=8192, radius=0.05, num_subclouds=100):
 
    nbrs = NearestNeighbors(radius=radius, algorithm='kd_tree').fit(points)
    sub_pointclouds = []

    for _ in range(num_subclouds):

        if mask is not None:
            masked_indices = np.where(mask)[0]
            seed_idx = np.random.choice(masked_indices)
        else:
            seed_idx = np.random.randint(len(points))
        indices = nbrs.radius_neighbors([points[seed_idx]], return_distance=False)[0]

        r = radius
        while len(indices) < min_points :
            r *= 1.5
            indices = nbrs.radius_neighbors([points[seed_idx]], radius=r, return_distance=False)[0]
        if len(indices) < min_points:
            continue  
        sub_pc = points[indices]
        sub_pointclouds.append(sub_pc)


    return sub_pointclouds

@DATASETS.register_module()
class CustomDataset(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.subset = config.subset
        self.npoints = config.N_POINTS

        
        center_coord = np.load(config.center_coord, allow_pickle=True)
        self.center = torch.tensor(center_coord)

        self.sample_points_num = config.npoints
        self.whole = config.get('whole')     
        
        infer = config.infer
        if infer:
            ply_path = os.path.join(self.data_root, 'ply_infer')
        else:
            ply_path = os.path.join(self.data_root, 'ply_mask_300')

        ply_list = natsorted(glob.glob(os.path.join(ply_path, '*.ply')))

        whole_ply_file = os.path.join(self.data_root, f'merge.ply')
        self.sub_points = []
        idx=0
        for ply in ply_list:
            pcd = o3d.io.read_point_cloud(ply)
            points = np.asarray(pcd.points)
            mask = np.ones_like(points)

            remove_outliers = (1-infer)
            outlier_factor = 1.0
            if remove_outliers:
                Q1 = np.percentile(points, 25, axis=0)
                Q3 = np.percentile(points, 75, axis=0)
                IQR = Q3 - Q1
                outlier_mask = (points < (Q1 - outlier_factor * IQR)) | (points > (Q3 + outlier_factor * IQR))

                mask = ~np.any(outlier_mask, axis=1)

            if infer:
                box_size = 50
                num_subclouds = 1
            else:
                box_size = random.randint(1,5)
                num_subclouds = 50

            sub_points = extract_sub_pointclouds_box(points, mask, box_size=box_size, num_subclouds = num_subclouds)
            self.sub_points = self.sub_points + sub_points
            print(idx)
            idx+=1
        self.permutation = np.arange(self.npoints)
        self.fine_distance = np.linalg.norm(self.sub_points[-1] - center_coord, axis=1, keepdims=True).max()

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc, centroid, m
        

    def random_sample(self, pc, num):
        permutation = np.arange(pc.shape[0])
        np.random.shuffle(permutation)
        pc = pc[permutation[-num:]]
        return pc
        
    

    def __getitem__(self, idx):
        data = self.sub_points[idx]

        data = self.random_sample(data, self.sample_points_num)
        data, centroid, max = self.pc_norm(data)
        data = torch.from_numpy(data).float()
        box_center = (self.center - centroid) / max
        return '0', self.fine_distance, data, centroid, max, box_center


    def __len__(self):
        return len(self.sub_points)