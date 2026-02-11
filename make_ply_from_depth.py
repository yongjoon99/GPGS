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
import torch
import imageio
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.project import create_point_cloud, get_intrinsics2
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np
import cv2
import glob




def write_ply_mask(points,colors,path_ply,mask=None):
    if mask is not None:
        num = np.sum(mask)
    else:
        num = points.shape[0]
    ply_header = '''ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''.format(int(num))

    with open(path_ply, 'w') as f:
        f.write(ply_header)
        for i in range(points.shape[0]):
            if mask.reshape(-1)[i]:
                f.write('{} {} {} {} {} {}\n'.format(points[i,0], points[i,1], points[i,2],
                                                                int(colors[i, 0]*255), int(colors[i, 1]*255), int(colors[i, 2]*255)))
def write_ply(points,colors,path_ply,mask=None):
    if mask is not None:
        num = np.sum(mask)
    else:
        num = points.shape[0]
    ply_header = '''ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''.format(num)

    with open(path_ply, 'w') as f:
        f.write(ply_header)
        for i in range(points.shape[0]):
            f.write('{} {} {} {} {} {}\n'.format(points[i,0], points[i,1], points[i,2],
                                                                int(colors[i, 0]*255), int(colors[i, 1]*255), int(colors[i, 2]*255)))         



def training(dataset, opt, pipe, testing_iterations,
              saving_iterations, checkpoint_iterations, checkpoint, debug_from, load_iteration, ref_idx, sample_interval):


    gaussians = GaussianModel(dataset.sh_degree)
    if load_iteration != 0:
        scene = Scene(dataset, gaussians, load_iteration)
    else:
        scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)



    viewpoint_stack = scene.getTrainCameras().copy()





    with torch.no_grad():

        for i in list(range(0,len(viewpoint_stack), sample_interval)) + [ref_idx]:

            
            viewpoint_cam = viewpoint_stack[i]
            
            gt_image = viewpoint_cam.original_image.cuda()
            scene_name = dataset.source_path.split('/')[-1]
            image_name = viewpoint_cam.image_name

            inv_depth = np.load(f'{dataset.model_path}/train/ours_300/depth_npy/{image_name}.npy')
            
            if len(inv_depth.shape)>2:
                inv_depth = inv_depth[:,:,0]

            depth = 1/inv_depth
            intrinsics = get_intrinsics2(viewpoint_cam.image_height, viewpoint_cam.image_width,viewpoint_cam.FoVx,viewpoint_cam.FoVy)
            
            if i==ref_idx: 
                mask = imageio.imread(glob.glob(f"{dataset.source_path}/reference/mask/{image_name}.*")[0])
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=3)
            else:
                mask = imageio.imread(glob.glob(f"{dataset.source_path}/mask/{image_name}.*")[0])

                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=3)

            colors = imageio.imread(f'{dataset.model_path}/train/ours_300/renders/{image_name}.png') / 255
            colors = colors.reshape((gt_image.shape[1]*gt_image.shape[2]),3).astype(float)

            mask = (mask<1)




            c2w=torch.eye(4)
            c2w[:3, :3]=torch.from_numpy(viewpoint_cam.R)
            c2w[:3, 3] = -torch.from_numpy(viewpoint_cam.R) @ torch.from_numpy(viewpoint_cam.T)

            depth = depth.squeeze()        

            mask_points = create_point_cloud(depth, intrinsics, c2w.cpu().numpy())



            if len(mask.shape)>2:
                mask = mask[:,:,0]

            path_to_save = os.path.join(dataset.source_path, 'ply_300')
            os.makedirs(path_to_save, exist_ok=True)

            ply_path =os.path.join(
            path_to_save, f"{image_name}.ply"
            )
            write_ply(mask_points, colors, ply_path)

            path_to_save = os.path.join(dataset.source_path,'ply_mask_300')
            os.makedirs(path_to_save, exist_ok=True)

            ply_path =os.path.join(
            path_to_save, f"{image_name}.ply"
            )
            write_ply_mask(mask_points, colors, ply_path, mask)


            if i==ref_idx:

                mask = ~mask*1
                mask = mask.astype('uint8')
                kernel = np.ones((5, 5), np.uint8)
                mask_5 = cv2.dilate(mask, kernel, iterations=10) * (1-mask)
                mask_4 = cv2.dilate(mask, kernel, iterations=30) * (1-mask)
                mask_3 = cv2.dilate(mask, kernel, iterations=50) * (1-mask)
                mask_2 = cv2.dilate(mask, kernel, iterations=70) * (1-mask)
                mask_1 = cv2.dilate(mask, kernel, iterations=90) * (1-mask)

                path_to_save = os.path.join(dataset.source_path,'ply_infer')
                os.makedirs(path_to_save, exist_ok=True)
                ply_path =os.path.join(
                path_to_save, f"{image_name}_1.ply"
                )
                write_ply_mask(mask_points, colors, ply_path, mask_1)

                ply_path =os.path.join(
                path_to_save, f"{image_name}_2.ply"
                )
                write_ply_mask(mask_points, colors, ply_path, mask_2)

                ply_path =os.path.join(
                path_to_save, f"{image_name}_3.ply"
                )
                write_ply_mask(mask_points, colors, ply_path, mask_3)

                ply_path =os.path.join(
                path_to_save, f"{image_name}_4.ply"
                )
                write_ply_mask(mask_points, colors, ply_path, mask_4)

                ply_path =os.path.join(
                path_to_save, f"{image_name}_5.ply"
                )
                write_ply_mask(mask_points, colors, ply_path, mask_5)
            print(i)




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
    parser.add_argument("--ref_idx", type=int, default = 184)
    parser.add_argument("--sample_interval", type=int, default = 5)
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
              args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.load_iteration, args.ref_idx, args.sample_interval)

    # All done
    print("\nTraining complete.")
