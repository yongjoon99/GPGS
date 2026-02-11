#!/bin/bash



export CUDA_VISIBLE_DEVICES=7



SCENE='skateboard'
gt_idx=187
ref_idx=186
ref_name=00186
DATASET=usid
python train.py -s data/${DATASET}/${SCENE} -m output/${DATASET}/${SCENE} --train_gt 1 --gt_idx $gt_idx
python render.py -s data/${DATASET}/${SCENE} -m output/${DATASET}/${SCENE} --train_gt 1 --gt_idx $gt_idx
python train_decompose.py -s data/${DATASET}/${SCENE} -m output/${DATASET}/${SCENE} --train_gt 1 --gt_idx $gt_idx
python render_depth.py -s data/${DATASET}/${SCENE} -m output/${DATASET}/${SCENE} --train_gt 1 --gt_idx $gt_idx --iteration 300 --images images
python make_ply_from_depth.py -s data/${DATASET}/${SCENE} -m output/${DATASET}/${SCENE} --train_gt 1 --gt_idx $gt_idx --ref_idx $ref_idx

cd Point-MAE


python main.py --config cfgs/custom.yaml \
    --start_ckpts pretrain.pth \
    --exp_name finetune_${SCENE} \
    --scene $SCENE --dataset $DATASET --epoch 50

python main_vis.py --config cfgs/custom.yaml \
    --ckpts experiments/custom/cfgs/finetune_${SCENE}/ckpt-epoch-050.pth \
    --exp_name finetune_${SCENE} --dataset $DATASET --mask_ratio 0.4 --infer 1 \
    --scene $SCENE --vis inference_results/${SCENE}_finetune --infer_iter 200 --ref_name $ref_name

cd ..

python proj_transfer.py -s data/${DATASET}/${SCENE} -m output/${DATASET}/${SCENE} --train_gt 1 --gt_idx $gt_idx --ref_idx $ref_idx --threshold 0.5

cd refine_model

python train_refine.py --source ../data/${DATASET}/${SCENE} --output ../output/${DATASET}/${SCENE} --epoch 201
python test_refine.py --source ../data/${DATASET}/${SCENE} --output ../output/${DATASET}/${SCENE} --epoch 200

cd ..

python train_compose.py -s data/${DATASET}/${SCENE} -m output/${DATASET}/${SCENE} --train_gt 1 --gt_idx $gt_idx --ref_idx $ref_idx --load_iteration 300 \
    --iteration 3000 --images refined_images

python render.py -s data/${DATASET}/${SCENE} -m output/${DATASET}/${SCENE} --train_gt 1 --gt_idx $gt_idx --iteration 3000
python render_path.py -s data/${DATASET}/${SCENE} -m output/${DATASET}/${SCENE} --train_gt 1 --gt_idx $gt_idx --iteration 3000
