import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from plyfile import PlyData, PlyElement
import cv2
import numpy as np
import copy
import open3d as o3d
import random

def save_ply(points, output_ply, color=(255,255,255)):
    # points = np.array(points)
    n_points = points.shape[0]
    header = "\n".join([
        "ply",
        "format ascii 1.0",
        f"element vertex {n_points}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header"
    ]) + "\n"


    default_rgb = color
    with open(output_ply, "w") as f:
        f.write(header)
        for xyz in points:
            line = "{:.8f} {:.8f} {:.8f} {} {} {}\n".format(
                xyz[0], xyz[1], xyz[2], *default_rgb
            )
            f.write(line)


def is_point_in_mask(points, mask, intrinsic, extrinsic=None):

    mask = (mask>0)*1
    N = points.shape[0]
    ones = torch.ones((N,1), device=points.device, dtype=points.dtype)
    points_h = torch.cat([points, ones], dim=1) 

    points_cam = (extrinsic @ points_h.T).T  
    in_front = points_cam[:, 2] > 0   
    
    proj = (intrinsic @ points_cam[:, :3].T).T  
    u = proj[:,0] / proj[:,2]
    v = proj[:,1] / proj[:,2]
    
    H, W = mask.shape
    u_int = torch.round(u).long()
    v_int = torch.round(v).long()
    
    valid_idx = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H) & in_front
    in_mask = torch.zeros(N, dtype=torch.int64, device=points.device)
    valid = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H) & in_front
    if valid_idx.any():
        in_mask[valid_idx] = mask[v_int[valid_idx], u_int[valid_idx]]

    img = torch.zeros((H, W), dtype=torch.uint8)
    img[v_int[valid], u_int[valid]] = 255

    ones = torch.ones(points.shape[0], 1, device=points.device)
    homo_world = torch.cat([points, ones], dim=1)
    camera_coords = (extrinsic @ homo_world.T).T
    fx, fy = intrinsic[0,0], intrinsic[1,1]
    cx, cy = intrinsic[0,2], intrinsic[1,2]

    z = camera_coords[:, 2]
    valid = z > 0
    z_valid = z[valid]
    xyz_valid = camera_coords[valid, :3]

    x = (fx * xyz_valid[:,0]/z_valid + cx).long()
    y = (fy * xyz_valid[:,1]/z_valid + cy).long()

    valid_x = (x >= 0) & (x < W)
    valid_y = (y >= 0) & (y < H)
    valid_bounds = valid_x & valid_y

    x_final = x[valid_bounds]
    y_final = y[valid_bounds]
    z_final = z_valid[valid_bounds]

    sorted_idx = torch.argsort(z_final, descending=True)
    x_sorted = x_final[sorted_idx]
    y_sorted = y_final[sorted_idx]


    valid_mask = torch.zeros((H,W), device=points.device, dtype=torch.bool)



    valid_mask[y_sorted, x_sorted] = True

    return in_mask, valid_mask




def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)

    base_model = builder.model_builder(config.model)

    builder.load_model(base_model, args.ckpts, logger = logger)

    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP
    if args.distributed:
        raise NotImplementedError()

    test(base_model, test_dataloader, args, config, logger=logger)


# visualization
def test(base_model, test_dataloader, args, config, logger = None):

    base_model.eval()  # set model to eval mode
    target = './vis'
    useful_cate = [
        "02691156", #plane
        "04379243",  #table
        "03790512", #motorbike
        "03948459", #pistol
        "03642806", #laptop
        "03467517",     #guitar
        "03261776", #earphone
        "03001627", #chair
        "02958343", #car
        "04090263", #rifle
        "03759954", # microphone
    ]
    with torch.no_grad():

        print(args.scene)

        extrinsic = np.load(f'{config.dataset.test._base_.output_path}/train/ours_300/c2w/{args.ref_name}.npy')
        intrinsic = np.load(f'{config.dataset.test._base_.output_path}/train/ours_300/intri/{args.ref_name}.npy')
        mask = cv2.imread(f'{config.dataset.test._base_.DATA_PATH}/reference/mask/{args.ref_name}.png')

        extrinsic = torch.tensor(extrinsic).cuda()
        intrinsic = torch.tensor(intrinsic).cuda()
        mask = torch.tensor(mask)
        R = extrinsic[:3, :3]
        R_inv = R.T
        
        T = extrinsic[:3, 3]
        T_inv = -R_inv @ T       

        w2c = torch.eye(4)
        w2c[:3, :3] = R_inv
        w2c[:3, 3] = T_inv
        for idx, (taxonomy_ids, min_distance, data, centroid, max, box_center) in enumerate(test_dataloader):

            if taxonomy_ids[0] == "02691156":
                a, b= 90, 135
            elif taxonomy_ids[0] == "04379243":
                a, b = 30, 30
            elif taxonomy_ids[0] == "03642806":
                a, b = 30, -45
            elif taxonomy_ids[0] == "03467517":
                a, b = 0, 90
            elif taxonomy_ids[0] == "03261776":
                a, b = 0, 75
            elif taxonomy_ids[0] == "03001627":
                a, b = 30, -45
            else:
                a, b = 0, 0

            a,b=30,30
            dataset_name = config.dataset.test._base_.NAME
            if dataset_name == 'ShapeNet':
                points = data.cuda()
            elif dataset_name == 'CustomDataset':
                points = data.cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')


            max = max.cuda().type(torch.float32)
            centroid = centroid.cuda().type(torch.float32)

            if idx>0:

                points_in_mask = (points_in_mask - centroid) / max
                new_points_idx = points_in_mask.shape[1]
                if new_points_idx < 512:
                    points[:,:new_points_idx,:] = points_in_mask.type(torch.float32)
                else:
                    points[:,:512,:] = points_in_mask.type(torch.float32)[:,:512,:]

            dense_points, vis_points, centers, pred = base_model(points, box_center, vis=True)
            final_image = []



            dense_points = (dense_points * max) + centroid
            vis_points = (vis_points * max) + centroid
            centers = (centers * max) + centroid

            pred = (pred * max) + centroid



            in_mask, projection = is_point_in_mask(dense_points[0], mask[:,:,0].cuda().type(torch.int32), intrinsic.type(torch.float32), w2c.cuda().type(torch.float32))

            points_in_mask = dense_points[:,in_mask.type(torch.bool),:]
            


            data_path = f'./{config.vis}/test_{idx}'

            if not os.path.exists(data_path):
                os.makedirs(data_path)


            pred = pred.squeeze().detach().cpu().numpy()
            save_ply(pred, os.path.join(data_path, 'pred.ply'))
            pred = misc.get_ptcloud_img(pred,a,b)
            final_image.append(pred[150:650,150:675,:])


            centers_save = centers.squeeze().detach().cpu().numpy()
            centers_save = misc.get_ptcloud_img(centers_save,a,b)
            final_image.append(centers_save[150:650,150:675,:])


            save_points = points.squeeze().detach().cpu().numpy()
            save_ply(save_points, os.path.join(data_path, 'gt.ply'))
            np.savetxt(os.path.join(data_path,'gt.txt'), save_points, delimiter=';')

            save_points = misc.get_ptcloud_img(save_points,a,b)
            final_image.append(save_points[150:650,150:675,:])


            vis_points = vis_points.squeeze().detach().cpu().numpy()
            save_ply(vis_points, os.path.join(data_path, 'vis.ply'))
            np.savetxt(os.path.join(data_path, 'vis.txt'), vis_points, delimiter=';')
            vis_points = misc.get_ptcloud_img(vis_points,a,b)

            final_image.append(vis_points[150:650,150:675,:])

            dense_points = dense_points.squeeze().detach().cpu().numpy()
            save_ply(dense_points, os.path.join(data_path, 'dense_points.ply'), (255,0,0))
            np.savetxt(os.path.join(data_path,'dense_points.txt'), dense_points, delimiter=';')
            dense_points = misc.get_ptcloud_img(dense_points,a,b)
            final_image.append(dense_points[150:650,150:675,:])

            img = np.concatenate(final_image, axis=1)
            img_path = os.path.join(data_path, f'plot.jpg')
            cv2.imwrite(img_path, img)
            print(idx)

            if idx == len(test_dataloader)-1:
 
                total_projection = torch.zeros_like(projection)
                merge_points = (points * max) + centroid
                original_center = centers
                for idx in range(args.infer_iter):
                    rand = random.random()

                    base_model.MAE_encoder.mask_type = config.model.transformer_config.mask_type_iter
                    base_model.MAE_encoder.mask_ratio = config.model.transformer_config.mask_ratio_iter
                    dense_points, vis_points, centers, pred = base_model(points, box_center, vis=True)
                    final_image = []

                    indices = torch.randperm(dense_points.shape[1])[:points.shape[1]]
                    next_points = copy.deepcopy(dense_points[:, indices, :])
                    next_points[:,:64,:] = centers.unsqueeze(0)

                    max = max.cuda()
                    centroid = centroid.cuda()
                    dense_points = (dense_points * max) + centroid
                    vis_points = (vis_points * max) + centroid
                    centers = (centers * max) + centroid
                    pred = (pred * max) + centroid
                    in_mask, projection = is_point_in_mask(dense_points[0], mask[:,:,0].cuda().type(torch.int32), intrinsic.type(torch.float32), w2c.cuda().type(torch.float32))

                    merge_points = torch.cat([merge_points, dense_points],dim=1)

                    print(idx)

                    total_projection = total_projection | projection
                    ratio = total_projection[mask[:,:,0]].sum() / (mask[:,:,0]>0).sum()
                    print(ratio)
                    if ratio > 0.7:

                        pcd = o3d.io.read_point_cloud(f"{config.dataset.test._base_.DATA_PATH}/ply_mask_300/{args.ref_name}.ply")
                        points = np.asarray(pcd.points)
                        merge_points = merge_points.squeeze().detach().cpu().numpy()
                        save_points = np.concatenate([points,merge_points])
                        save_ply(save_points, os.path.join(config.vis, 'merge.ply'), (255,0,0))
                        print('enough_projection')
                        return 0

                    points = next_points

                pcd = o3d.io.read_point_cloud(f"{config.dataset.test._base_.DATA_PATH}/ply_mask_300/{args.ref_name}.ply")
                points = np.asarray(pcd.points)
                merge_points = merge_points.squeeze().detach().cpu().numpy()
                save_points = np.concatenate([points,merge_points])
                save_ply(save_points, os.path.join(config.vis, 'merge.ply'), (255,0,0))
                print('all_process_complete')
                
                break


        return
