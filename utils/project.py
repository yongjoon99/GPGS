import numpy as np
import torch
import cv2


def get_intrinsics2(H, W, fovx, fovy):
    fx = 0.5 * W / np.tan(0.5 * fovx)
    fy = 0.5 * H / np.tan(0.5 * fovy)
    cx = 0.5 * W
    cy = 0.5 * H
    return np.array([[fx,  0, cx],
                     [ 0, fy, cy],
                     [ 0,  0,  1]])

def create_point_cloud(depth_map, camera_matrix, extrinsic_matrix):

    """Create point cloud from depth map and camera parameters."""
    
    height = depth_map.shape[0]
    width = depth_map.shape[1]

    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    x, y = np.meshgrid(x, y)

    normalized_x = (x - camera_matrix[0, 2]) / camera_matrix[0, 0]
    normalized_y = (y - camera_matrix[1, 2]) / camera_matrix[1, 1]
    normalized_z = np.ones_like(x)


    depth_map_reshaped = np.repeat(depth_map[:, :, np.newaxis], 3, axis=2)
    homogeneous_camera_coords = depth_map_reshaped * np.dstack((normalized_x, 
                                                                normalized_y, 
                                                                normalized_z))


    ones = np.ones((height, width, 1))
    homogeneous_camera_coords = np.dstack((homogeneous_camera_coords, ones))

    homogeneous_world_coords = np.dot(homogeneous_camera_coords, 
                                      extrinsic_matrix.T)


    point_cloud = (homogeneous_world_coords[:, :, :3] / 
                                            homogeneous_world_coords[:, :, 3:])

    point_cloud = point_cloud.reshape(-1, 3)

    return point_cloud


def interpolate_masked_depth(depth_map, mask, radius=1):
    """
    인접한 유효 픽셀의 평균으로 마스크 영역을 주변부에서부터 반복적으로 채웁니다.
    depth_map: [H, W] torch tensor (실수형)
    mask: [H, W] torch bool tensor (True: 채워야 할 영역)
    radius: int, 이웃 반경(1=8방향, 2=24방향 등)
    반환값: 채운 torch tensor
    """
    filled_depth = depth_map.clone()
    mask_fill = mask.clone()

    # radius 내 모든 (dy, dx) 이웃 좌표 생성 (0,0 제외)
    neighbors_offsets = [
        (dy, dx) for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if not (dy == 0 and dx == 0)
    ]

    while mask_fill.any():
        neighbors_sum = torch.zeros_like(filled_depth)
        neighbors_count = torch.zeros_like(filled_depth)

        for dy, dx in neighbors_offsets:
            shifted_mask = mask_fill.roll(shifts=(dy, dx), dims=(0, 1))
            shifted_depth = filled_depth.roll(shifts=(dy, dx), dims=(0, 1))
            valid = ~shifted_mask
            neighbors_sum += shifted_depth * valid.float()
            neighbors_count += valid.float()

        valid_pixels = mask_fill & (neighbors_count > 0)
        filled_depth[valid_pixels] = neighbors_sum[valid_pixels] / neighbors_count[valid_pixels]
        mask_fill[valid_pixels] = False

    return filled_depth

def project_point_cloud_with_rgb_mask(point_cloud, new_camera_matrix, new_extrinsic, image_shape, image, mask, interpolate=True, depth=True):


    device = point_cloud.device
    H, W = image_shape
    
    point_cloud = point_cloud.type(torch.float32)

    xyz = point_cloud[:, :3]
    rgb = point_cloud[:, 3:] * 255 
    
    ones = torch.ones(xyz.shape[0], 1, device=device)
    homo_world = torch.cat([xyz, ones], dim=1)
    camera_coords = (new_extrinsic @ homo_world.T).T

    fx, fy = new_camera_matrix[0,0], new_camera_matrix[1,1]
    cx, cy = new_camera_matrix[0,2], new_camera_matrix[1,2]
    

    z = camera_coords[:, 2]
    valid = z > 0
    z_valid = z[valid]
    xyz_valid = camera_coords[valid, :3]
    rgb_valid = rgb[valid]

    x = (fx * xyz_valid[:,0]/z_valid + cx).long()
    y = (fy * xyz_valid[:,1]/z_valid + cy).long()


    valid_x = (x >= 0) & (x < W)
    valid_y = (y >= 0) & (y < H)
    valid_bounds = valid_x & valid_y
    
    x_final = x[valid_bounds]
    y_final = y[valid_bounds]
    z_final = z_valid[valid_bounds]
    rgb_final = rgb_valid[valid_bounds]

    
    sorted_idx = torch.argsort(z_final, descending=True)
    x_sorted = x_final[sorted_idx]
    y_sorted = y_final[sorted_idx]
    rgb_sorted = rgb_final[sorted_idx]


    color_image = torch.zeros((H,W,3), device=device, dtype=torch.uint8)
    depth_map = torch.full((H,W), 1e10, device=device)
    valid_mask = torch.zeros((H,W), device=device, dtype=torch.bool)


    color_image[y_sorted, x_sorted] = rgb_sorted.byte()
    depth_map[y_sorted, x_sorted] = z_final[sorted_idx]
    valid_mask[y_sorted, x_sorted] = True

    p_mask = (depth_map == 1e10).cpu().numpy().astype(np.uint8)

    valid_mask = 1-p_mask
    color_image = image * torch.tensor(1-valid_mask).unsqueeze(2).cuda() + color_image * torch.tensor(valid_mask).unsqueeze(2).cuda()
    color_np = color_image.cpu().detach().numpy().astype('uint8')
    mask = (1-valid_mask)

    if interpolate:
        depth_np = depth_map.detach().cpu().numpy()
        depth_map = interpolate_masked_depth(depth_map, (depth_map == 1e10), radius=2)
        inpainted = cv2.inpaint(color_np, mask, 3, cv2.INPAINT_TELEA)
        color_image = torch.from_numpy(inpainted).to(device)
        

    valid_mask = torch.from_numpy(valid_mask).to(device)
    color_image = color_image * valid_mask.unsqueeze(2)

    return color_image, depth_map, valid_mask, mask

