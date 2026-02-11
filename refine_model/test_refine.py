import torch
import numpy as np
import torch.optim as optim
from refine_data_loader import MaskedPatchDataset, extract_aligned_patches_test

import os
import glob
from PIL import Image
from simple_model import SimpleNet
from tqdm import tqdm
import argparse
import shutil


def create_gaussian_weight(patch_size):

    center = patch_size / 2
    y, x = np.ogrid[-center:patch_size-center, -center:patch_size-center]
    sigma = patch_size / 4
    weight = np.exp(-(x*x + y*y) / (2*sigma*sigma))
    return np.float32(weight)

def blend_patches_with_weights(image, output_patches, coords, patch_size=16):

    H, W, C = image.shape
    result = np.zeros_like(image, dtype=np.float32)
    weight_sum = np.zeros((H, W, 3), dtype=np.float32)
    

    weight_map = create_gaussian_weight(patch_size)
    weight_map = np.repeat(weight_map[:, :, np.newaxis], 3, axis=2) 
    

    for patch, (top, left) in zip(output_patches, coords):
        patch_np = patch.transpose(1, 2, 0) 
        
        weighted_patch = patch_np * weight_map
        result[top:top+patch_size, left:left+patch_size] += weighted_patch
        
        weight_sum[top:top+patch_size, left:left+patch_size] += weight_map

    mask = weight_sum > 0

    result = np.where(mask, result / weight_sum, image)
    
    return result







def main(args):

    scene = args.source.split('/')[-1]
    proj_path = f'{args.source}/transfered_images'
    image_path = f'{args.source}/images'
    refined_image_path = f'{args.source}/refined_images'
    mask_path = f'{args.source}/projection_mask_small'
    gt_path =  f'{args.output}/train/ours_300/renders'
    output_path = f'output/{scene}'

    os.makedirs(output_path,exist_ok=True)

    
    proj_list = sorted(glob.glob(os.path.join(proj_path, '*.png')) + glob.glob(os.path.join(proj_path, '*.jpg')))
    mask_list = sorted(glob.glob(os.path.join(mask_path, '*.png')) + glob.glob(os.path.join(mask_path, '*.jpg')))
    small_mask_list = sorted(glob.glob(os.path.join(mask_path, '*.png')) + glob.glob(os.path.join(mask_path, '*.jpg')))
    gt_list = sorted(glob.glob(os.path.join(gt_path, '*.png')) + glob.glob(os.path.join(gt_path, '*.jpg')))
    image_list = sorted(glob.glob(os.path.join(image_path, '*.png')) + glob.glob(os.path.join(image_path, '*.jpg')))

    model = SimpleNet(15, 3, len(proj_list)).cuda()

    checkpoint_dir = os.path.join('checkpoints', f"{scene}")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(os.path.join(f'{checkpoint_dir}/checkpoint_epoch_{args.epoch}.pth'), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    patch_size = 16

    for i in range(len(proj_list)):
        image = np.array(Image.open(proj_list[i]).convert("RGB"))
        mask = np.array(Image.open(mask_list[i]).convert("L")) > 127
        gt = np.array(Image.open(gt_list[i]).convert("RGB"))
        small_mask = np.array(Image.open(small_mask_list[i]).convert("L")) > 127
        image_tesnsor = torch.tensor(image).permute(2,0,1).cuda()/255
        mask = torch.tensor(mask).cuda()
        name = proj_list[i].split('/')[-1].split('.')[0] + '.' + image_list[0].split('/')[-1].split('.')[1]
        save_path = os.path.join(output_path, name)

        patches, coords, masks, idxs, coord_patches = extract_aligned_patches_test(image_tesnsor, mask, i, patch_size=patch_size, overlap=args.overlap)
        if patches==None:
            Image.fromarray(image.astype(np.uint8)).save(save_path)
            continue
        patches = torch.stack(patches)
        masks = torch.stack(masks)
        idxs = torch.tensor(idxs).cuda()

        coord_patches = torch.stack(coord_patches)

        with torch.no_grad():
            input_patches = torch.cat([patches,coord_patches], dim=1)
            output_residual_patches = model(input_patches, idxs)
            output_patches = output_residual_patches + patches


        output_np = output_patches.cpu().numpy() * 255
        output_np = np.clip(output_np, 0, 255)

        coords_np = coords.cpu().numpy()
        result = blend_patches_with_weights(image.astype(np.float32), output_np, coords_np, patch_size)
        
        small_mask = np.expand_dims(small_mask,axis=2)
        result = result*(small_mask) + gt * (1-small_mask)

        Image.fromarray(result.astype(np.uint8)).save(save_path)
        print(i)

    if args.prepare_data:

        shutil.copytree(image_path, refined_image_path)
        output_list = sorted(glob.glob(os.path.join(output_path, '*.png')) + glob.glob(os.path.join(output_path, '*.jpg')))
        for i in range(len(output_list)):

            name = output_list[i].split('/')[-1]
            shutil.copyfile(output_list[i], refined_image_path + '/' + name)
            print(i)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--source', type=str, default='../data/usid/cookie')
    parser.add_argument('--output', type=str, default='../output/usid/cookie')
    parser.add_argument('--overlap', type=int, default=6)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--prepare_data', type=int, default=1)



    args = parser.parse_args()
    
    main(args)
    