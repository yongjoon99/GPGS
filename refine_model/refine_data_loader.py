import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
class MaskedPatchDataset(Dataset):
    def __init__(self, image_list, mask_list, gt_list, patch_size=8, transform=None):

        self.ps = patch_size
        self.transform = transform

        self.images = []
        self.masks = []
        self.gts = []
        self.coords = []
        self.idxs=[]
        for i in range(len(image_list)):
            proj = torch.tensor(imageio.imread(image_list[i])).permute(2,0,1).cuda() / 255
            mask = torch.tensor(imageio.imread(mask_list[i]))
            if len(mask.shape)>2:
                mask = mask[:,:,0].cuda()
            else:
                mask = mask.cuda()
            mask = (mask>0)*1.0
            gt = torch.tensor(imageio.imread(gt_list[i])).permute(2,0,1).cuda() / 255

            patches, coords, gt_patches, patch_masks, idxs = extract_aligned_patches(proj, mask, gt, patch_size=patch_size, idx=i)
            
            if patches == None:
                continue
            self.images = self.images + patches
            self.masks = self.masks + patch_masks
            self.gts = self.gts + gt_patches
            self.coords = self.coords + coords
            self.idxs = self.idxs + idxs

        print(len(self.images), 'patches')


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        patch = self.images[idx]
        patch_mask = self.masks[idx]
        gt_patch = self.gts[idx]
        img_idx = self.idxs[idx]
        coord = self.coords[idx]
        return patch, gt_patch, patch_mask, img_idx, coord




def extract_aligned_patches(image, mask, gt, idx, patch_size=16):

    H, W = mask.shape

    y_idx, x_idx = torch.where(mask)
    if y_idx.numel() == 0:
        return None,None,None,None,None
    
    def expand_edge(min_val, max_val, multiple, limit):
        length = max_val - min_val + 1
        new_length = ((length + multiple -1) // multiple) * multiple
        new_min = max(0, min_val - (new_length - length)//2)
        new_max = min(new_min + new_length -1, limit-1)
        return new_min, new_max
    
    y_min, y_max = expand_edge(y_idx.min().item(), y_idx.max().item(), patch_size, H)
    x_min, x_max = expand_edge(x_idx.min().item(), x_idx.max().item(), patch_size, W)
    
    grid_h = (y_max - y_min + 1) // patch_size
    grid_w = (x_max - x_min + 1) // patch_size

 
    x_coords = torch.arange(W)
    y_coords = torch.arange(H)
    grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
    grid_x = grid_x / W
    grid_y = grid_y / H
    coordinate_tensor = torch.stack((grid_x, grid_y), axis=0).cuda()
    

    patches = []
    coords = []
    gt_patches = []
    patch_masks = []
    idxs = []
    for i in range(grid_h):
        for j in range(grid_w):
            top = y_min + i*patch_size
            left = x_min + j*patch_size
            patch_mask = mask[top:top+patch_size, left:left+patch_size]
            
            if patch_mask.sum() > (patch_size**2 - 1):
                patch = image[..., top:top+patch_size, left:left+patch_size] if image.dim()==3 \
                       else image[top:top+patch_size, left:left+patch_size]
                
                gt_patch = gt[..., top:top+patch_size, left:left+patch_size] if gt.dim()==3 \
                       else gt[top:top+patch_size, left:left+patch_size]
                coord = coordinate_tensor[...,top:top+patch_size, left:left+patch_size]
                if torch.abs(gt_patch - patch).mean() < 0.3:
                    patches.append(patch)
                    gt_patches.append(gt_patch)
                    patch_masks.append(patch_mask)
                    coords.append(coord)
                    idxs.append(idx)
                

    if len(patches) > 0:
        return patches, coords, gt_patches, patch_masks, idxs
    else:
        return None, None, None, None, None


def extract_aligned_patches_test(image, mask, idx, patch_size=16, overlap=2):

    device = image.device
    H, W = mask.shape
    

    y_idx, x_idx = torch.where(mask)
    if y_idx.numel() == 0:
        return None, None, None, None, None, None

    y_min, y_max = y_idx.min().item(), y_idx.max().item()
    x_min, x_max = x_idx.min().item(), x_idx.max().item()
    

    def expand_edge(min_val, max_val, multiple, limit):
        length = max_val - min_val + 1
        new_length = ((length + multiple -1) // multiple) * multiple
        new_min = max(0, min_val - (new_length - length)//2)
        new_max = min(new_min + new_length -1, limit-1)
        return new_min, new_max
    y_min, y_max = expand_edge(y_min, y_max, patch_size, H)
    x_min, x_max = expand_edge(x_min, x_max, patch_size, W)
    

    stride = patch_size - overlap

    x_coords = torch.arange(W)
    y_coords = torch.arange(H)
    grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
    grid_x = grid_x / W
    grid_y = grid_y / H
    coordinate_tensor = torch.stack((grid_x, grid_y), axis=0).cuda()
    


    patches = []
    coords = []
    patch_masks = []
    idxs = []
    coords_patchs=[]

    y = y_min
    while y <= y_max - patch_size + 1:
        x = x_min
        while x <= x_max - patch_size + 1:
            patch_mask = mask[y:y+patch_size, x:x+patch_size]
            
            if patch_mask.sum() > (1): 
                patch = image[..., y:y+patch_size, x:x+patch_size] if image.dim()==3 \
                      else image[y:y+patch_size, x:x+patch_size]
                coords_patch = coordinate_tensor[..., y:y+patch_size, x:x+patch_size]
                patches.append(patch)
                patch_masks.append(patch_mask)
                coords.append([y, x])
                idxs.append(idx)
                coords_patchs.append(coords_patch)
            x += stride 
        y += stride  
    
    if len(patches) > 0:
        return patches, torch.tensor(coords, device=device), patch_masks, idxs, coords_patchs
    else:
        return None, None, None, None, None
