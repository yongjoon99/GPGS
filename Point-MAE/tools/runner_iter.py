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

def save_ply(points, output_ply, color=(255,255,255)):
    # points = np.array(points)
    n_points = points.shape[0]
    header = f"""ply
    format ascii 1.0
    element vertex {n_points}
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    """
    default_rgb = color

    # PLY 파일로 저장
    with open(output_ply, "w") as f:
        f.write(header)
        for xyz in points:
            line = "{:.8f} {:.8f} {:.8f} {} {} {}\n".format(
                xyz[0], xyz[1], xyz[2], *default_rgb
            )
            f.write(line)


def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)

    base_model = builder.model_builder(config.model)
    # base_model.load_model_from_ckpt(args.ckpts)
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
        for idx, (taxonomy_ids, model_ids, data, centroid, max, box_center) in enumerate(test_dataloader):

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


            for idx in range(30):
                dense_points, vis_points, centers, pred = base_model(points, box_center, vis=True)
                final_image = []



                indices = torch.randperm(dense_points.shape[1])[:points.shape[1]]
                next_points = copy.deepcopy(dense_points[:, indices, :])

                max = max.cuda()
                centroid = centroid.cuda()
                dense_points = (dense_points * max) + centroid
                vis_points = (vis_points * max) + centroid
                centers = (centers * max) + centroid
                pred = (pred * max) + centroid


                data_path = f'./{config.vis}/test_{idx}'
                if not os.path.exists(data_path):
                    os.makedirs(data_path)


                pred = pred.squeeze().detach().cpu().numpy()
                save_ply(pred, os.path.join(data_path, 'pred.ply'))
                pred = misc.get_ptcloud_img(pred,a,b)
                final_image.append(pred[150:650,150:675,:])


                centers = centers.squeeze().detach().cpu().numpy()
                centers = misc.get_ptcloud_img(centers,a,b)
                final_image.append(centers[150:650,150:675,:])


                points = points.squeeze().detach().cpu().numpy()
                save_ply(points, os.path.join(data_path, 'gt.ply'))
                np.savetxt(os.path.join(data_path,'gt.txt'), points, delimiter=';')


                points = misc.get_ptcloud_img(points,a,b)
                final_image.append(points[150:650,150:675,:])


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
                points = next_points

                if idx > 200:
                    break
            break

        return
