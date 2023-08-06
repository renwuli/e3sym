#!/bin/bash

if [ ! -e ".tmp" ];
then
    mkdir -p .tmp
fi

CUDA_VISIBLE_DEVICES=0 python test.py --config configs/test.yaml
