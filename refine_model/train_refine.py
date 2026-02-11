import os
import torch
import numpy as np
import torch.optim as optim
from simple_model import SimpleNet
from refine_data_loader import MaskedPatchDataset
from torch.utils.data import Dataset, DataLoader
import glob
from torch.utils.tensorboard import SummaryWriter 
from tqdm import tqdm
import argparse
from FDL.FDL_pytorch import FDL_loss


def main(args):
    scene = args.source.split('/')[-1]
    log_dir = os.path.join('logs', f'{scene}_logs')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir) 
    proj_path = f'{args.source}/transfered_images'
    mask_path = f'{args.source}/near_mask'   
    gt_path =  f'{args.output}/train/ours_300/renders'
    proj_list = sorted(glob.glob(os.path.join(proj_path, '*.png')))
    mask_list = sorted(glob.glob(os.path.join(mask_path, '*.png')))
    gt_list = sorted(glob.glob(os.path.join(gt_path, '*.png')))

    model = SimpleNet(15,3, len(proj_list)).cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    patch_size = 16
    dataset = MaskedPatchDataset(proj_list, mask_list, gt_list, patch_size=patch_size)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=True)
    checkpoint_dir = os.path.join('checkpoints', f"{scene}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    loss_fn = torch.nn.L1Loss()
    epochs=args.epoch
    progress_bar = tqdm((range(epochs)), desc="Training progress")

    fdl_loss = FDL_loss(l_len=2, phase_weight=0.01).cuda()

    for epoch in progress_bar:
        model.train()   
        total_loss = 0.0
        total_freq = 0.0
        total_color = 0.0
        total_mean = 0.0
        total_std = 0.0
        num_batches = 0
        for patches in dataloader:
            
            optimizer.zero_grad()
            proj_patches = patches[0]
            gt_patches = patches[1]
            mask_patches = patches[2].unsqueeze(dim=1)
            idx = patches[3].cuda()
            coord = patches[4].cuda()

            input_patches = torch.cat([proj_patches, coord], dim=1)
            pred_residual = model(input_patches, idx)
            pred_patches = pred_residual + proj_patches

            freq_loss = fdl_loss(pred_patches, gt_patches)


            color_loss = loss_fn(pred_patches, gt_patches)

            loss = freq_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_freq += freq_loss.item()
            total_color += color_loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        progress_bar.set_postfix(loss=f"{avg_loss:.4f}")

        avg_loss = total_loss / num_batches
        avg_freq = total_freq / num_batches
        avg_color = total_color / num_batches
        avg_mean = total_mean / num_batches
        avg_std = total_std / num_batches

        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Loss/freq', avg_freq, epoch)
        writer.add_scalar('Loss/color', avg_color, epoch)
        writer.add_scalar('Loss/mean', avg_mean, epoch)
        writer.add_scalar('Loss/std', avg_std, epoch)
        
        if epoch % 20 == 0:
            writer.add_images('Input', proj_patches[:4], epoch)
            writer.add_images('Prediction', pred_patches[:4], epoch)
            writer.add_images('Ground Truth', gt_patches[:4], epoch)
            writer.add_images('mask', mask_patches[:4], epoch)
        if epoch % 100 ==0 or epoch == epochs-1:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }
            torch.save(
                checkpoint, 
                os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
            )
            print(f'checkpoint_{epoch}_saved')
        if epoch % 10 == 0:
            progress_bar.write(f"Epoch [{epoch}/{epochs}] Loss: {loss:.4f}")
    writer.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--source', type=str, default='../data/usid/cookie')
    parser.add_argument('--output', type=str, default='../output/usid/cookie')
    parser.add_argument('--epoch', type=int, default=201)

    args = parser.parse_args()
    
    main(args)
    