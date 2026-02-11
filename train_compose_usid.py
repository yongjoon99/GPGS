#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '6'
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.sh_utils import SH2RGB, RGB2SH
from utils.project import create_point_cloud, get_intrinsics2, create_point_cloud_torch
from decompose_ply import decompose
from plyfile import PlyData, PlyElement
import torch.nn.functional as F
import open3d as o3d
import glob
import imageio
import copy
import numpy as np
import cv2
from scipy.spatial import ConvexHull, Delaunay
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False






def points_inside_convex_hull(point_cloud, mask, remove_outliers=True, outlier_factor=1.0):
    """
    Given a point cloud and a mask indicating a subset of points, this function computes the convex hull of the
    subset of points and then identifies all points from the original point cloud that are inside this convex hull.

    Parameters:
    - point_cloud (torch.Tensor): A tensor of shape (N, 3) representing the point cloud.
    - mask (torch.Tensor): A tensor of shape (N,) indicating the subset of points to be used for constructing the convex hull.
    - remove_outliers (bool): Whether to remove outliers from the masked points before computing the convex hull. Default is True.
    - outlier_factor (float): The factor used to determine outliers based on the IQR method. Larger values will classify more points as outliers.

    Returns:
    - inside_hull_tensor_mask (torch.Tensor): A mask of shape (N,) with values set to True for the points inside the convex hull
                                              and False otherwise.
    """

    # Extract the masked points from the point cloud
    masked_points = point_cloud[mask].cpu().numpy()

    # Remove outliers if the option is selected
    if remove_outliers:
        Q1 = np.percentile(masked_points, 25, axis=0)
        Q3 = np.percentile(masked_points, 75, axis=0)
        IQR = Q3 - Q1
        outlier_mask = (masked_points < (Q1 - outlier_factor * IQR)) | (masked_points > (Q3 + outlier_factor * IQR))
        filtered_masked_points = masked_points[~np.any(outlier_mask, axis=1)]
    else:
        filtered_masked_points = masked_points

    # Compute the Delaunay triangulation of the filtered masked points
    delaunay = Delaunay(filtered_masked_points)

    # Determine which points from the original point cloud are inside the convex hull
    points_inside_hull_mask = delaunay.find_simplex(point_cloud.cpu().numpy()) >= 0

    # Convert the numpy mask back to a torch tensor and return
    inside_hull_tensor_mask = torch.tensor(points_inside_hull_mask, device='cuda')

    return inside_hull_tensor_mask




def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, load_iteration, outlier):

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    if load_iteration != 0:
        scene = Scene(dataset, gaussians, load_iteration, norest = False)
    else:
        scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    kernel_size = dataset.kernel_size

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    trainCameras = scene.getTrainCameras().copy()
    if dataset.disable_filter3D:
        gaussians.reset_3D_filter()
    else:
        gaussians.compute_3D_filter(cameras=trainCameras)

    viewpoint_stack = scene.getTrainCameras().copy()
    ema_loss_for_log, ema_depth_loss_for_log, ema_mask_loss_for_log, ema_normal_loss_for_log = 0.0, 0.0, 0.0, 0.0

    require_depth = not dataset.use_coord_map
    require_coord = dataset.use_coord_map

    opt.iterations = 310    
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1


    ref = dataset.gt_idx - 1



    f_dc_bank = copy.deepcopy(gaussians._features_dc)
    f_rest_bank = copy.deepcopy(gaussians._features_rest)



    with torch.no_grad():


        reference_cam = copy.deepcopy(viewpoint_stack[args.gt_idx-1])
        with torch.no_grad():
            proj_gaussians = copy.deepcopy(gaussians)
        viewpoint_cam = reference_cam
        scene_name = dataset.model_path.split('/')[-1].split('_')[0]
        # /home/cv/Desktop/yongjoon/Infusion/depth_inpainting/output/aircap_301/575_mask.ply
        ref_name = reference_cam.image_name.split('.')[0]
        gt_image = reference_cam.original_image.cuda()
        image_name = reference_cam.image_name

        scale = 1

        mask = cv2.imread(f'{dataset.source_path}/reference/mask/{ref_name}.png')
        mask = cv2.resize(mask, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


        # depth_path_large = f'/home/cv/Desktop/yongjoon/RaDe-GS/output_usid_test/{scene_name}/train/ours_30000/depth_inv_npy/{ref_name}.npy'

        depth_path_large = f'{dataset.source_path}/completed_depth_{ref_name}.npy'
        depth = np.load(depth_path_large, allow_pickle=True)

        depth = 1 / depth




        depth = torch.tensor(depth).cuda().unsqueeze(0).unsqueeze(0)
        depth = F.interpolate(depth, scale_factor=scale, mode='bilinear', align_corners=False)

        depth = depth[0,0]

        c2w = reference_cam.world_view_transform.T.inverse()
        W, H = reference_cam.image_width * scale , reference_cam.image_height * scale
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W) / 2],
            [0, H / 2, 0, (H) / 2],
            [0, 0, 0, 1]]).float().cuda().T
        projection_matrix = c2w.T @ reference_cam.full_proj_transform
        intrins = (projection_matrix @ ndc2pix)[:3,:3].T
        
        grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')
        points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
        rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
        rays_o = c2w[:3,3]
        xyz = depth.reshape(-1, 1) * rays_d + rays_o
        

        mask = torch.tensor(mask).cuda()[:,:,0] > 200

        mask_dilate = cv2.imread(f'{dataset.source_path}/reference/mask_dilate/{ref_name}.png')
        mask_dilate = cv2.resize(mask_dilate, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        mask_dilate = torch.tensor(mask_dilate).cuda()[:,:,0] > 200

        resized_img = imageio.imread(f'{dataset.source_path}/reference/{ref_name}.jpg')
        resized_img = torch.tensor(resized_img).cuda().permute(2,0,1).unsqueeze(0)/255
        resized_img = F.interpolate(resized_img, scale_factor=scale, mode='bilinear', align_corners=False)[0]


        mask_flat = mask.reshape(-1)
        masked_xyz = xyz[mask_flat == 1].type(torch.float32)
        masked_rgb = resized_img.reshape(3, -1).T[mask_flat == 1].type(torch.float32)
        resized_points = torch.cat([masked_xyz, masked_rgb],axis=1)




        shape=gaussians._xyz.shape
        # points = torch.tensor(points).cuda()
        # colors = torch.tensor(colors).cuda()
        # colors = RGB2SH(colors)
       

        # projected_points_large = torch.cat([points,colors],axis=1).type(torch.float32)

        gaussians.add_points(resized_points)
        gaussians.compute_3D_filter(cameras=trainCameras)
        scene.save(30001)


        point_cloud_path = os.path.join(dataset.model_path, "point_cloud/iteration_{}".format(30001))

        savepath = point_cloud_path
        np.savetxt(savepath + '/points.txt', np.array(shape), fmt='% 06d')



def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--load_iteration", type=int, default = 300)
    parser.add_argument("--outlier", type=float, default = 1.0)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # if not args.disable_viewer:
    #     network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.load_iteration, args.outlier)

    # All done
    print("\nTraining complete.")
