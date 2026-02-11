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
import copy
import os
import cv2
import torch
import imageio
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.project import  get_intrinsics2, project_point_cloud_with_rgb_mask
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np
from plyfile import PlyData, PlyElement
import glob
import torch.nn.functional as F
from skimage import color


def transfer(inpainted_region, original_region, dilate_mask, near_mask):


    original_region = color.rgb2lab(original_region)
    inpainted_region = color.rgb2lab(inpainted_region)

    near_pixel = original_region[near_mask==1]
    near_mean = near_pixel.mean(axis=0)
    near_std = near_pixel.std(axis=0)

    mask_pixel = inpainted_region[dilate_mask==1]

    region_mean = mask_pixel.mean(axis=0)
    region_std = mask_pixel.std(axis=0)


    adjusted = near_std * (inpainted_region - region_mean) / region_std + near_mean


    adjusted[:,:,0] = np.clip(adjusted[:,:,0], 1, 99)
    adjusted[:,:,1] = np.clip(adjusted[:,:,1], -120, 120)
    adjusted[:,:,2] = np.clip(adjusted[:,:,2], -120, 120)

    adjusted[:,:,1:3] = inpainted_region[:,:,1:3]
    adjusted = color.lab2rgb(adjusted)

    
    adjusted = np.clip(adjusted,0.0,1.0)



    return adjusted



def training(dataset, opt, pipe, testing_iterations, saving_iterations,
            checkpoint_iterations, checkpoint, debug_from, load_iteration,
            ref_idx, dilate, threshold):


    first_iter = 0

    gaussians = GaussianModel(dataset.sh_degree)
    if load_iteration != 0:
        scene = Scene(dataset, gaussians, load_iteration)
    else:
        scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")






    viewpoint_stack = scene.getTrainCameras().copy()


    stop_grad_path = args.model_path + f'/point_cloud/iteration_{args.load_iteration}/points.txt'

    if os.path.exists(stop_grad_path):
        with open(stop_grad_path, 'r') as file:
            total_points = float(file.readline().strip())
            args.stop_grad = int(total_points)
        
    else:
        pass

    with torch.no_grad():

        reference_cam = copy.deepcopy(viewpoint_stack[ref_idx])
        with torch.no_grad():
            proj_gaussians = copy.deepcopy(gaussians)

        scene_name = dataset.model_path.split('/')[-1].split('_')[0]
        ref_name = reference_cam.image_name.split('.')[0]
        gt_image = reference_cam.original_image.cuda()
        
        ply_path_large = f'Point-MAE/inference_results/{scene_name}_finetune/merge.ply'

        plydata = PlyData.read(ply_path_large)
        x = plydata['vertex']['x']
        y = plydata['vertex']['y']
        z = plydata['vertex']['z']
        r = plydata['vertex']['red']
        g = plydata['vertex']['green']
        b = plydata['vertex']['blue']

        points = torch.tensor(np.stack([x,y,z],axis=1)).cuda().type(torch.float32)[:]
        colors = torch.tensor(np.stack([r,g,b],axis=1)).cuda().type(torch.float32)[:]/255

        c2w=torch.eye(4)
        c2w[:3, :3]=torch.from_numpy(reference_cam.R)
        c2w[:3, 3] = -torch.from_numpy(reference_cam.R) @ torch.from_numpy(reference_cam.T)

        intrinsics = get_intrinsics2(reference_cam.image_height, reference_cam.image_width,reference_cam.FoVx,reference_cam.FoVy)
        intrinsics = torch.tensor(intrinsics).cuda()

        R = c2w[:3, :3]
        R_inv = R.T
        

        T = c2w[:3, 3]
        T_inv = -R_inv @ T
        

        w2c = torch.eye(4)
        w2c[:3, :3] = R_inv
        w2c[:3, 3] = T_inv
        mask = cv2.imread(f'{dataset.source_path}/reference/mask/{ref_name}.png')

        projected_points = torch.cat([points,colors],axis=1)
        projected_image, depth, v_mask, proj_mask = project_point_cloud_with_rgb_mask(projected_points, intrinsics, w2c.cuda(),(reference_cam.image_height, reference_cam.image_width), gt_image.permute(1,2,0), mask, 1)

        np.save(dataset.source_path + f'/completed_depth_{ref_name}', depth.detach().cpu().numpy())

        scale = 4
        mask = cv2.resize(mask, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)



        kernel_size = 11
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)


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
        

        mask = torch.tensor(mask).cuda()[:,:,0] > 0

        mask_dilate = cv2.imread(f'{dataset.source_path}/reference/mask_dilate/{ref_name}.png')
        mask_dilate = cv2.resize(mask_dilate, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        mask_dilate = torch.tensor(mask_dilate).cuda()[:,:,0] > 0

        resized_img = imageio.imread(glob.glob(f'{dataset.source_path}/reference/{ref_name}.*')[0])
        resized_img = torch.tensor(resized_img).cuda().permute(2,0,1).unsqueeze(0)/255
        resized_img = F.interpolate(resized_img, scale_factor=scale, mode='bilinear', align_corners=False)[0]

        if mask is not None:
            mask_flat = mask.reshape(-1)
            masked_xyz = xyz[mask_flat == 1].type(torch.float32)
            masked_rgb = resized_img.reshape(3, -1).T[mask_flat == 1].type(torch.float32)
            resized_points = torch.cat([masked_xyz, masked_rgb],axis=1)

            mask_dilate_flat = mask_dilate.reshape(-1)
            masked_dilate_xyz = xyz[mask_dilate_flat == 1].type(torch.float32)
            masked_dilate_rgb = resized_img.reshape(3, -1).T[mask_dilate_flat == 1].type(torch.float32)
            resized_dilated_points = torch.cat([masked_dilate_xyz, masked_dilate_rgb],axis=1)


    original_path =  f'{dataset.model_path}/train/ours_300/renders'
    original_list = sorted(glob.glob(os.path.join(original_path, '*.png')))


    os.makedirs(os.path.join(dataset.source_path, 'projection_large'), exist_ok=True)
    os.makedirs(os.path.join(dataset.source_path, 'projection_mask_large'), exist_ok=True)
    os.makedirs(os.path.join(dataset.source_path, 'projection_small'), exist_ok=True)
    os.makedirs(os.path.join(dataset.source_path, 'projection_mask_small'), exist_ok=True)
    os.makedirs(os.path.join(dataset.source_path, 'near_mask'), exist_ok=True)
    os.makedirs(os.path.join(dataset.source_path, 'transfered_images'), exist_ok=True)
    

    with torch.no_grad():
        for iteration in range(first_iter, len(viewpoint_stack)):

            viewpoint_cam = viewpoint_stack[iteration]
            
            mask = viewpoint_cam.mask.unsqueeze(dim=2)

            kernel_size = 0
            render_pkg_o = render(viewpoint_cam, proj_gaussians, pipe, background, kernel_size = kernel_size)
            render_depth_o = render_pkg_o["expected_depth"]
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, kernel_size = kernel_size)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            render_depth = render_pkg["expected_depth"]
            render_depth = 1 / (render_depth + 1e-6)

            intrinsics = get_intrinsics2(viewpoint_cam.image_height, viewpoint_cam.image_width,viewpoint_cam.FoVx,viewpoint_cam.FoVy)
            intrinsics = torch.tensor(intrinsics).cuda()
            c2w=torch.eye(4)
            c2w[:3, :3]=torch.from_numpy(viewpoint_cam.R)
            c2w[:3, 3] = -torch.from_numpy(viewpoint_cam.R) @ torch.from_numpy(viewpoint_cam.T)
            
            R = c2w[:3, :3]
            R_inv = R.T
            
            T = c2w[:3, 3]
            T_inv = -R_inv @ T
            
            w2c = torch.eye(4)
            w2c[:3, :3] = R_inv
            w2c[:3, 3] = T_inv
            image = (image*255).permute(1,2,0)
            img_vis = image.cpu().detach().numpy().astype('uint8')
            original_image = imageio.imread(original_list[iteration])
            image = torch.tensor(original_image).cuda()

            inpaint=1
            projected_image_small, proj_depth_small, v_mask_small, proj_mask_small = project_point_cloud_with_rgb_mask(resized_points, intrinsics, w2c.cuda(),(viewpoint_cam.image_height, viewpoint_cam.image_width),image, mask, inpaint, depth=False)
            projected_image_large, proj_depth_large, v_mask_large, proj_mask_large = project_point_cloud_with_rgb_mask(resized_dilated_points, intrinsics, w2c.cuda(),(viewpoint_cam.image_height, viewpoint_cam.image_width),image, mask, inpaint, depth=False)

            v_mask_large = v_mask_large.unsqueeze(dim=2) * 1.0
            v_mask_small = v_mask_small.unsqueeze(dim=2) * 1.0

            inv_proj = 1 / proj_depth_small
            vis_proj = (inv_proj*255).squeeze(0).detach().cpu().numpy().astype('uint8')

            depth_mask = (render_depth >10000)
            render_depth[depth_mask] = 1e-6
            vis_depth = (render_depth*255).squeeze().detach().cpu().numpy().astype('uint8')

            inv_proj_depth = 1/proj_depth_large
            proj_depth_large = proj_depth_large * v_mask_large.squeeze()
            inv_proj_depth = inv_proj_depth * v_mask_large.squeeze()
            sub = (1/render_depth) - proj_depth_large
            valid = sub > -threshold
            f_mask = valid.squeeze() * v_mask_large.squeeze()

            kernel = np.ones((5, 5), np.uint8)
            
            f_mask = cv2.dilate(f_mask.squeeze().cpu().detach().numpy(), kernel, iterations=dilate)
            f_mask = cv2.erode(f_mask, kernel, iterations=dilate)
            f_mask = torch.tensor(f_mask).unsqueeze(0).cuda()

            f_mask_small = f_mask * v_mask_small.squeeze()

            f_mask_small = cv2.dilate(f_mask_small.squeeze().cpu().detach().numpy(), kernel, iterations=dilate)
            f_mask_small = torch.tensor(f_mask_small).unsqueeze(0).cuda()

            f_mask = f_mask.permute(1,2,0)
            f_mask_small = f_mask_small.permute(1,2,0)
            save_image = image*(1-f_mask) + projected_image_large * f_mask  
            save_image_small = image*(1-f_mask_small*f_mask) + projected_image_large * f_mask_small  




            save_image = save_image.detach().cpu().numpy().astype('uint8')
            save_image_small = save_image_small.detach().cpu().numpy().astype('uint8')
            projected_image_large = projected_image_large.detach().cpu().numpy()
            projected_image_small = projected_image_small.detach().cpu().numpy()
            f_mask = (f_mask*255).squeeze().detach().cpu().numpy().astype('uint8')
            v_mask_large = (v_mask_large*255).squeeze().detach().cpu().numpy().astype('uint8')
            f_mask_small = (f_mask_small*255).squeeze().detach().cpu().numpy().astype('uint8')
            v_mask_small = (v_mask_small*255).squeeze().detach().cpu().numpy().astype('uint8')



            near_mask = f_mask - f_mask_small


            render_unseen = image.detach().cpu().numpy()
            if near_mask.sum()==0:
                tranfered_image = save_image
            else:
                adjusted = transfer(save_image/255, render_unseen/255, f_mask/255, near_mask/255)*255
                tranfered_image = (render_unseen * np.expand_dims(1 - v_mask_large/255, 2) + adjusted * np.expand_dims(v_mask_large/255, 2)).astype('uint8')
            viewpoint_cam.image_name = viewpoint_cam.image_name.split('.')[0] + '.png'
            imageio.imwrite(dataset.source_path + f'/projection_large/{viewpoint_cam.image_name}',save_image)
            imageio.imwrite(dataset.source_path + f'/projection_mask_large/{viewpoint_cam.image_name}',f_mask)
            imageio.imwrite(dataset.source_path + f'/projection_small/{viewpoint_cam.image_name}',save_image_small)
            imageio.imwrite(dataset.source_path + f'/projection_mask_small/{viewpoint_cam.image_name}',f_mask_small)
            imageio.imwrite(dataset.source_path + f'/near_mask/{viewpoint_cam.image_name}',near_mask)
            imageio.imwrite(dataset.source_path + f'/transfered_images/{viewpoint_cam.image_name}',tranfered_image)

            print(iteration)

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
    parser.add_argument("--threshold", type=float, default = 0.2)
    parser.add_argument("--ref_idx", type=int, default = 62)
    parser.add_argument("--dilate", type=int, default = 0)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # if not args.disable_viewer:
    #     network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
              args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.load_iteration,
                args.ref_idx,args.dilate, args.threshold)

    # All done
    print("\nTraining complete.")
