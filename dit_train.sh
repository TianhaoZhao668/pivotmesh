#!/bin/bash

torchrun dit_train.py --model DiT-MIN/3 \
    --global-batch-size 64 \
    --ckpt-every 10000 \
    --obj-name ShapeNetCorev2_obj_800faces_mask \
    --data-path /root/shapenet_data/ShapeNetCore_decimates_800face \
    --from-pretrained /root/github_code/pivotmesh/results/ShapeNetCorev2_obj_800faces_mask-DiT-MIN-3/checkpoints/0650000.pt