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
from scene import Scene

from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.render_utils import generate_path
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, get_combined_args_debug
from gaussian_renderer import GaussianModel
import imageio

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, kernel_size):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_path")


    scene_name = model_path.split('/')[-1]
    makedirs(render_path, exist_ok=True)

    frames = []
    d_frames = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background, kernel_size=kernel_size)["render"]
        gt = view.original_image[0:3, :, :]
        rendering = torch.clamp(rendering, 0.0, 1.0)
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

        frames.append((rendering.permute(1,2,0).detach().cpu().numpy()*255).astype('uint8'))

    c_video_path = os.path.join(model_path, name, f'ours_{iteration}/{scene_name}_path_color.mp4')

    imageio.mimwrite(c_video_path, frames, fps=30, quality=5)


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        n_fames = 120
        cam_traj = generate_path(scene.getTrainCameras(), n_frames=n_fames)

        render_set(dataset.model_path, "train", scene.loaded_iter, cam_traj, gaussians, pipeline, background, dataset.kernel_size)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    # args = get_combined_args_debug(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)