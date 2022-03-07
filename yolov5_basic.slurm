#!/bin/bash
#SBATCH --job-name=yolob
#SBATCH --nodes=1
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --time=200:00:00
#SBATCH --output=result/yolov5_basic-%j
#SBATCH --exclude=node2

source activate yoloenv



python train.py --img 640 --batch 16 --epochs 3 --data VOC.yaml --weights yolov5s.pt --img 640 
