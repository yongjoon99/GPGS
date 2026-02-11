#!/bin/bash

export CUDA_VISIBLE_DEVICES=7

SCENE='sign'
gt_idx=158
DATASET=COR
python train.py -s data/${DATASET}/${SCENE} -m output/${DATASET}/${SCENE} --train_gt 0 --gt_idx $gt_idx
python render.py -s data/${DATASET}/${SCENE} -m output/${DATASET}/${SCENE} --train_gt 0 --gt_idx $gt_idx
python train_decompose.py -s data/${DATASET}/${SCENE} -m output/${DATASET}/${SCENE} --train_gt 0 --gt_idx $gt_idx --outlier 0.5
python render_depth.py -s data/${DATASET}/${SCENE} -m output/${DATASET}/${SCENE} --train_gt 0 --gt_idx $gt_idx --iteration 300