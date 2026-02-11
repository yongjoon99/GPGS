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
from decompose_ply import decompose

import copy
import numpy as np
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
    gaussians = GaussianModel(dataset.sh_degree)
    if load_iteration != 0:
        scene = Scene(dataset, gaussians, load_iteration, norest = True)
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

    viewpoint_stack = None
    ema_loss_for_log, ema_depth_loss_for_log, ema_mask_loss_for_log, ema_normal_loss_for_log = 0.0, 0.0, 0.0, 0.0

    require_depth = not dataset.use_coord_map
    require_coord = dataset.use_coord_map

    opt.iterations = 301 
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1


    f_dc_bank = copy.deepcopy(gaussians._features_dc)

    with torch.no_grad():
        gaussians._features_rest = gaussians._features_rest*0


    for iteration in range(first_iter, opt.iterations + 1):


        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        mask = viewpoint_cam.mask.cuda()

        viewpoint_cam.original_image = torch.stack([mask,mask,mask])

        reg_kick_on = iteration >= opt.regularization_from_iter
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, kernel_size, require_coord = require_coord and reg_kick_on, require_depth = require_depth and reg_kick_on)
        image: torch.Tensor
        image, viewspace_point_tensor, visibility_filter, radii = (
                                                                    render_pkg["render"], 
                                                                    render_pkg["viewspace_points"], 
                                                                    render_pkg["visibility_filter"], 
                                                                    render_pkg["radii"])
        gt_image = viewpoint_cam.original_image.cuda()

        loss = l1_loss(image, gt_image)

        loss.backward()


        with torch.no_grad():
            # Remove gradient excluding features_dc components
            if gaussians._xyz.grad is not None :
                    gaussians._xyz.grad[:] = 0.0
            if gaussians._features_rest.grad is not None:
                    gaussians._features_rest.grad[:] = 0.0
            if gaussians._scaling.grad is not None:
                    gaussians._scaling.grad[:] = 0.0
            if gaussians._rotation.grad is not None:
                    gaussians._rotation.grad[:] = 0.0
            if gaussians._opacity.grad is not None:
                    gaussians._opacity.grad[:] = 0.0

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log


            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()


            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)



            if iteration==300:

                sub = gaussians._features_dc - f_dc_bank
                obj = (sub[:,0,0] > 0) + (sub[:,0,1] > 0) + (sub[:,0,2] > 0) > 0
                del_index = torch.where(obj==True)[0]

                mask3d_convex = points_inside_convex_hull(gaussians._xyz.detach(), obj, outlier_factor=outlier)

                gaussians._opacity[mask3d_convex] = -10000
                gaussians._opacity[obj] = -10000

                mask3d_convex = obj + mask3d_convex
                
                center = gaussians._xyz[mask3d_convex].mean(axis=0).detach().cpu().numpy()

                np.save(dataset.model_path + '/center_coord', center)
                

                obj_idx = mask3d_convex.detach().cpu().numpy()
                np.save(dataset.model_path + '/obj_idx', obj_idx)
                size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                gaussians.densify_and_prune(opt.densify_grad_threshold, 0.05, scene.cameras_extent, size_threshold)
                if dataset.disable_filter3D:
                    gaussians.reset_3D_filter()
                else:
                    gaussians.compute_3D_filter(cameras=trainCameras)
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                decompose(dataset,obj_idx, iteration, load_iteration)


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
    parser.add_argument("--load_iteration", type=int, default = 30000)
    parser.add_argument("--outlier", type=float, default = 0.5)
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
