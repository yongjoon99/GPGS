# GPGS: Consistent 3D Object Removal via Geometry-Aware 3D Inpainting andProjected Image Refinement in 3D Gaussian Splatting

Yongjoon Lee, Donghyeon Cho

![Teaser](figures/pipeline.png)

# 1. Installation
## Clone this repository.
```
git clone https://github.com/yongjoon99/GPGS.git --recursive
```

## Install dependencies.
1. create an environment
```
conda create -n GPGS python=3.9
conda activate GPGS
```

2. install pytorch and other dependencies.
```
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install submodules/diff-gaussian-rasterization --no-build-isolation
pip install submodules/simple-knn  --no-build-isolation
pip install -r requirements.txt
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl

cd Point-MAE/extensions/chamfer_dist
python setup.py install --user
```

Download the Point-MAE pretrained model from [Point-MAE](https://github.com/Pang-Yatian/Point-MAE) and move pretrain.pth file to Point-MAE.

# 2. Data
## Dataset download
We used 360-USID and COR-NeRF datasets.

Download 360-USID dataset from [AuraFusion](https://kkennethwu.github.io/aurafusion360/)

Download COR-NERF dataset from [COR-DATA](https://drive.google.com/drive/folders/1v9ew8lvJEfFc4hp6ZO0Rtr5B-6freS8E?usp=sharing)

We also provide the sample data that include reference image and unseen mask with correct data structure in [Sample Data](https://drive.google.com/drive/folders/1cyAu5CuLf3ChvN9MNhaIpk97gCe0qlrC?usp=drive_link)

## Unseen mask and reference image generation

We manually created the unseen mask using [SAM](https://github.com/facebookresearch/segment-anything), and generated the reference image using [LeftRefill](https://github.com/ewrfcas/LeftRefill).

You can use any segmentation and inpainting model to make unseen mask and reference image.

The reference image and the unseen mask must have the same name as the original image.

## Data Structure

The data should be structured as follows.

Do not separate the test images from the usid data; place all images in the images directory.

We also provide sample data 

```
data
│
|____COR
│  |____{scene name} 
|     |______images
│     |______masks     
|     |______referene
│     |   |____mask
|     |   |  |___{unseen_mask_name}.png
│     |   |____dilated_mask
|     |   |  |___{unseen_mask_name}.png
│     |   |____{reference_image_name}.png
|     |______sparse
│     
|____usid
│  |____{scene name} 
|     |______images
│     |______masks     
|     |______referene
│     |   |____mask
|     |   |  |___{unseen_mask_name}.png
│     |   |____dilated_mask
|     |   |  |___{unseen_mask_name}.png
│     |   |____{reference_image_name}.png
|     |______sparse

```

# 3. Training


Modify the following argument in scripts/train_and_decompose.sh to match your data and run the script.

<details>
<summary><span style="font-weight: bold;">Arguments Details</span></summary>

  #### --source_path / -s
  Path to the source directory.
  #### --model_path / -m 
  Path where the trained model should be stored.
  #### --images / -i
  Alternative subdirectory for COLMAP images (```images``` by default).
  #### --gt_idx, --train_gt
  We created parameters based on the COR-NeRF dataset, where the GT for the preceding index has training data in the subsequent index.

  When --train_gt == 1, the image corresponding to [:gt_idx] in the entire image list is used for training.

  When --train_gt == 0, the image corresponding to [gt_idx:] in the entire image list is used for training.

  Therefore, for usid data where the GT is in the latter index, the argument must be used in reverse. 

</details>
<br>

```
bash scripts/train_and_decompose.sh
```

After creating the unseen mask and reference image, modify the following argument in scripts/project_and_refinement.sh to match your data and run the script.

<details>
<summary><span style="font-weight: bold;">Arguments Details</span></summary>

  #### --ref_idx
  Index determining which image in the train set is the reference image.
  #### --ref_name / -m 
  Name of the reference image. ex\) <ref_name>.png
  #### --images / -i
  Alternative subdirectory for COLMAP images (```images``` by default).
  #### --gt_idx, --train_gt
  We created parameters based on the COR-NeRF dataset, where the GT for the preceding index has training data in the subsequent index.

  When --train_gt == 1, the image corresponding to [:gt_idx] in the entire image list is used for training.

  When --train_gt == 0, the image corresponding to [gt_idx:] in the entire image list is used for training.

  Therefore, for usid data where the GT is in the latter index, the argument must be used in reverse. 

</details>
<br>

```
bash scripts/project_and_refinement.sh
```

To run GPGS using the sample data containing unseen masks and reference images, execute the following command


```
bash scripts/train_sample.sh
```





# Acknowledgements
This work is built upon the following repositories:
* [RaDe-GS](https://github.com/HKUST-SAIL/RaDe-GS): We used their depth-involved rasterization for Gaussian Splatting.
* [Point-MAE](https://github.com/Pang-Yatian/Point-MAE): Our Geometry-aware 3D inpainting is based on their implementation.
* [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting): The core optimization and rasterization engine.
