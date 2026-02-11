import torch
from diffusers import EulerDiscreteScheduler
from diffusers.schedulers import DDIMScheduler, DDPMScheduler

from utils.autoencoder_utils import AutoencoderKL
from utils.marigold_di_utils import AGDDv2
from utils.marigold_ww_utils import MarigoldPipeline
# copy from gaussian-opacity-fields
# copy from 2DGS
import math
import torch
import numpy as np

def depths_to_points(view, depthmap):
    # c2w = (view.world_view_transform.T).inverse()
    # we train in camera coordinate
    c2w = torch.eye(4).float().cuda()
    W, H = view.image_width, view.image_height
    fx = W / (2 * math.tan(view.FoVx / 2.))
    fy = H / (2 * math.tan(view.FoVy / 2.))
    intrins = torch.tensor(
        [[fx, 0., W/2.],
        [0., fy, H/2.],
        [0., 0., 1.0]]
    ).float().cuda()
    grid_x, grid_y = torch.meshgrid(torch.arange(W)+0.5, torch.arange(H)+0.5, indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3).float().cuda()
    rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
    rays_o = c2w[:3,3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points


def depth_to_normal(view, depth):
    """
        view: view camera
        depth: depthmap 
    """
    points = depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output, points


def dilate_mask(mask, iterations=1, kernel_size=3):
    # Dilate the mask
    dilated_mask = mask.clone()
    for _ in range(iterations):
        dilated_mask = torch.nn.functional.max_pool2d(dilated_mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    return dilated_mask[0]

def estimate_depth_marigold(rgb: torch.Tensor):
    """
    Args:
        rgb (torch.Tensor): (3, H, W)

    Returns:
        torch.Tensor: (1, H, W)
    """
    pipe = AGDDv2.from_pretrained("prs-eth/marigold-depth-v1-0", prediction_type="depth").to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    depth = pipe(rgb, is_latent_optimizing=False)[0]
    return depth

# ==============================
# AGDD: Adaptive Guided Depth Diffusion
# ==============================
def normalize_depth_ignore_zeros(depth_tensor, min_val=None, max_val=None):
    """
    將 depth tensor 正規化到 0~1 範圍，忽略值為 0 的區域
    
    參數:
    depth_tensor (torch.Tensor): 輸入的 depth tensor
    min_val (float): 自定義的最小值，預設為 None (使用非零區域的最小值)
    max_val (float): 自定義的最大值，預設為 None (使用非零區域的最大值)
    
    回傳:
    torch.Tensor: 正規化後的 tensor，原本為 0 的區域保持為 0
    """
    # 將輸入轉換為 float 類型
    depth_tensor = depth_tensor.float()
    
    # 創建非零區域的 mask
    valid_mask = (depth_tensor != 0)
    
    # 只從非零區域取得最大最小值
    if min_val is None:
        min_val = torch.min(depth_tensor[valid_mask])
    if max_val is None:
        max_val = torch.max(depth_tensor[valid_mask])
    
    # 避免除以零
    if max_val == min_val:
        return torch.zeros_like(depth_tensor)
    
    # 創建輸出 tensor
    normalized = torch.zeros_like(depth_tensor)
    
    # 只正規化非零區域
    normalized[valid_mask] = (depth_tensor[valid_mask] - min_val) / (max_val - min_val)
    
    return normalized

def unnormalize_depth_ignore_zeros(depth_tensor, ref_depth_tensor):
    valid_mask = (ref_depth_tensor != 0)
    min_val = torch.min(ref_depth_tensor[valid_mask])
    max_val = torch.max(ref_depth_tensor[valid_mask])
    # unnormalize the align_depth from original gt depth, ignore the zero values
    unnormalized = depth_tensor * (max_val - min_val) + min_val
    return unnormalized


def align_depth_agdd_v2(depth, rgb, mask, opt, seed=7777, tb_writer=None):
    """_summary_

    Args:
        depth (_type_): _description_
        rgb (_type_): _description_
        mask (_type_): _description_
        opt (_type_): _description_
        seed (int, optional): _description_. Defaults to 7777.

    Returns:
        align_depth: (1, H, W)
    """
    
    # IMPORTANT NOTE: bfloat16 tends to predict unsmooth align depth, so we use float16 instead
    pipe = AGDDv2.from_pretrained(
        "prs-eth/marigold-v1-0", variant="fp16", torch_dtype=torch.float16
    ).to("cuda")
    vae = AutoencoderKL.from_pretrained("prs-eth/marigold-v1-0", subfolder="vae").to(dtype=torch.float16).to("cuda")
    pipe.register_modules(vae=vae)
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    # pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    depth[mask[None] == 1] = 0
    gt_depth = normalize_depth_ignore_zeros(depth)

    rgb = rgb.to(torch.float16)
    gt_depth = gt_depth.to(torch.float16)
    mask = (mask == 1).to(torch.float16)[None]
    
    generator = torch.Generator()
    generator.manual_seed(seed)
    inpaint_depth = pipe(image=rgb, incomplete_depth=gt_depth, unseen_mask=mask, num_inference_steps=50, generator=generator, is_latent_optimizing=True, opt=opt, tb_writer=tb_writer)
    align_depth = inpaint_depth.to(torch.float32)
    
    # unnormalize the align_depth from original gt depth, ignore the zero values
    unnormalize_align_depth = unnormalize_depth_ignore_zeros(align_depth, depth)
    
    # get the error of align_depth and gt_depth
    print(depth)
    print(unnormalize_align_depth)
    print(f"""
        \033[93m#########final alignment error#########\033[0m
        {torch.abs(unnormalize_align_depth[mask == 0] - depth[mask == 0]).mean().item()}
        \033[93m#########final alignment error#########\033[0m
    """)
    
    return unnormalize_align_depth


# ==============================
# Wonder World Guided Depth Diffusion
# https://github.com/KovenYu/WonderWorld.git
# ==============================

def align_depth_marigold_ww(depth, rgb, mask, opt, seed=7777):
    """Wonder World Guided Depth Diffusion

    Args:
        depth (torch.Tensor): (1, H, W)
        rgb (torch.Tensor): (3, H, W)
        mask (torch.Tensor): (1, H, W)
        opt (OptimizationParams): arguments for optimization
        seed (int, optional): random seed. Defaults to 7777.
        
    Returns:
        torch.Tensor: (1, H, W)
    """
    pipe = MarigoldPipeline.from_pretrained(
        "prs-eth/marigold-v1-0", variant="fp16", torch_dtype=torch.bfloat16 
    ).to("cuda")
    # vae = AutoencoderKL.from_pretrained("prs-eth/marigold-v1-0", subfolder="vae").to(dtype=torch.float16).to("cuda")
    # pipe.register_modules(vae=vae)
    # pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    
    
    depth[mask[None] == 1] = 0
    gt_depth = normalize_depth_ignore_zeros(depth)

    rgb = rgb.to(torch.bfloat16)
    gt_depth = gt_depth.to(torch.bfloat16)
    mask_align = (mask == 0)
    
    align_depth = pipe(
        rgb,
        denoising_steps=30,     # optional
        ensemble_size=1,       # optional
        processing_res=0,     # optional
        match_input_res=True,   # optional
        batch_size=0,           # optional
        color_map=None,   # optional
        show_progress_bar=True, # optional
        depth_conditioning=True,
        target_depth=gt_depth,
        mask_align=mask_align,
        mask_farther=None,
        guidance_steps=8,
        # guidance_steps=20,
        logger=None,
    )[None].to(torch.float32)
    
    unnormalize_align_depth = unnormalize_depth_ignore_zeros(align_depth, depth)
    return unnormalize_align_depth