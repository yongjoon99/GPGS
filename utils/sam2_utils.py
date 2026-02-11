import os
import shutil
from argparse import ArgumentParser
from glob import glob

import cv2
import numpy as np
from natsort import natsorted
from PIL import Image
from sam2.sam2_video_predictor import SAM2VideoPredictor
from tqdm import tqdm

predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large")
    
    
def export_unseen_mask(source_path, output_path):
    """
    Export unseen mask by using unseen contour as bbox prompt to SAM2.
    Save the mask as png image with the same name as the image in the source_path.
    
    source_path: path to the source directory
    output_path: path to the output directory
    """
    removal_image_dir = os.path.join(output_path, "renders")
    unseen_contour_dir = os.path.join(output_path, "unseen_contour")
    unseen_mask_dir = os.path.join(source_path, "unseen_masks")
    name_dir = os.path.join(source_path, "images")
    os.makedirs(unseen_mask_dir, exist_ok=True)
    # 1. 'init_state' only support .jpg, so save all removal images in .png to another dir but as .jpg.
    removal_image_dir_jpg = os.path.join(os.path.dirname(removal_image_dir), "renders_jpg")
    os.makedirs(removal_image_dir_jpg, exist_ok=True)
    for file in os.listdir(removal_image_dir):
        if file.endswith(".png"):
            shutil.copy(os.path.join(removal_image_dir, file), os.path.join(removal_image_dir_jpg, file.replace(".png", ".jpg")))
    
    # 2. init_state
    state = predictor.init_state(video_path=removal_image_dir_jpg)
    predictor.reset_state(state)
    
    # 3. add new prompts and instantly get the output on the same frame
    unseen_contour_paths = natsorted(glob(os.path.join(unseen_contour_dir, "*.png")))
    for frame_idx, unseen_contour_path in tqdm(enumerate(unseen_contour_paths), desc="Add unseen contour as a bbox prompt"):
        unseen_contour = Image.open(unseen_contour_path)
        unseen_contour_np = np.array(unseen_contour) # shape (H, W)
        
        # TODO: Opt. Some mask operations
        # unseen_contour_np = cv2.morphologyEx(unseen_contour_np, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        # unseen_contour_np = cv2.morphologyEx(unseen_contour_np, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        
        unseen_contour_np = np.where(unseen_contour_np > 0, 1, 0)
        
        # get the bbox of the unseen contour
        y_indices, x_indices = np.nonzero(unseen_contour_np)
        if len(y_indices) == 0 or len(x_indices) == 0:
            print(f"No unseen contour found at frame {frame_idx}")
            continue
        
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        bbox = np.array([[x_min, y_min], [x_max, y_max]])
        # add new prompts and instantly get the output on the same frame
        frame_idx, object_ids, masks = predictor.add_new_points_or_box(
            state, 
            frame_idx=frame_idx, 
            obj_id=1,
            box=bbox
        )
        
    # 4. propagate the prompts to get masklets throughout the video
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }   
        
    # 5. visualize the masks with red & save to tmp
    vis_frame_stride = 30
    img_paths = natsorted(glob(os.path.join(removal_image_dir_jpg, "*.jpg")))
    for out_frame_idx in range(0, len(os.listdir(removal_image_dir_jpg)), vis_frame_stride):
        img = Image.open(img_paths[out_frame_idx])
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            img = np.array(img)
            img[out_mask[0]] = (255, 0, 0)
            img = Image.fromarray(img)
            os.makedirs("tmp/unseen_masks", exist_ok=True)
            img.save(os.path.join("tmp/unseen_masks", f"{out_frame_idx}_{out_obj_id}.jpg"))

    # 6. save the output unseenmasks to unseen_masks dir, and in the same name as the image in the source_path
    name_paths = natsorted(glob(name_dir + '/*'))
    for out_frame_idx in range(0, len(os.listdir(removal_image_dir_jpg))):
        file_name = os.path.basename(name_paths[out_frame_idx])
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            out_mask = np.where(out_mask > 0, 255, 0)
            out_mask = out_mask.astype(np.uint8)[0]
            out_mask = Image.fromarray(out_mask)
            out_mask.save(os.path.join(unseen_mask_dir, file_name))
            
    # cleanup
    shutil.rmtree(removal_image_dir_jpg)
            
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, required=True, choices=["360-USID", "Other-360", "my_data"])
    parser.add_argument("--scene", "-s", type=str, required=True)
    args = parser.parse_args()
    
    dataset_name = args.dataset
    scene_name = args.scene
    
    export_unseen_mask(
        source_path=f"data/{dataset_name}/{scene_name}",
        output_path=f"output/{dataset_name}/{scene_name}/train/ours_30000_object_removal"
    )