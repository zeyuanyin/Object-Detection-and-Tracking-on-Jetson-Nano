#!/bin/bash

# Scripts to reproduce the result in the project.

### Train baseline algorithms.

`srun --ntasks=1 --cpus-per-task=12 -p gpu -q gpu-8 --gres=gpu:1 python3 train_ssd.py --net vgg16-ssd --data=data/fruit --pretrained-ssd='models/vgg16-ssd-mp-0_7726.pth' --model-dir=models/fruit_vgg  --batch-size 6 --lr 0.0003 --epochs=100`
srun --ntasks=1 --cpus-per-task=12 -p gpu -q gpu-8 --gres=gpu:1 python3 train_ssd.py --net mb1-ssd --data=data/fruit --pretrained-ssd='models/mobilenet-v1-ssd-mp-0_675.pth' --model-dir=models/fruit_mob1 --epochs=100  --batch-size 6 --lr 0.0003
srun --ntasks=1 --cpus-per-task=12 -p gpu -q gpu-8 --gres=gpu:1 python3 train_ssd.py --net mb2-ssd-lite --data=data/fruit --pretrained-ssd='models/mb2-ssd-lite-mp-0_686.pth' --model-dir=models/fruit_mb2 --epochs=100 --batch-size 6 --lr 0.0003
srun --ntasks=1 --cpus-per-task=12 -p gpu -q gpu-8 --gres=gpu:1 python3 train_ssd.py --net sq-ssd-lite- --data=data/fruit --model-dir=models/fruit_sq --epochs=100 --batch-size 6 --lr 0.0003

### Evaluate the learned model 

srun --ntasks=1 --cpus-per-task=12 -p gpu -q gpu-8 --gres=gpu:1 python3 eval_ssd.py --model=models/fruit_mb2/mb2-ssd-lite-Epoch-99-Loss-3.320802785504249.pth --label_file=models/fruit/labels.txt --net="mb2-ssd-lite" --dataset_type="open_images" --dataset=data/fruit/
srun --ntasks=1 --cpus-per-task=12 -p gpu -q gpu-8 --gres=gpu:1 python3 eval_ssd.py --model=models/fruit_mob1/mb1-ssd-Epoch-95-Loss-3.305213132212239.pth --label_file=models/fruit/labels.txt --net="mb1-ssd" --dataset_type="open_images" --dataset=data/fruit/
srun --ntasks=1 --cpus-per-task=12 -p gpu -q gpu-8 --gres=gpu:1 python3 eval_ssd.py --model=models/fruit_vgg/vgg16-ssd-Epoch-99-Loss-3.460213225887668.pth --label_file=models/fruit/labels.txt --net="vgg16-ssd" --dataset_type="open_images" --dataset=data/fruit/


### Quant the learned model 
srun --ntasks=1 --cpus-per-task=12 -p gpu -q gpu-8 --gres=gpu:1 python3 quant-mb1.py
srun --ntasks=1 --cpus-per-task=12 -p gpu -q gpu-8 --gres=gpu:1 python3 quant-mb2.py
srun --ntasks=1 --cpus-per-task=12 -p gpu -q gpu-8 --gres=gpu:1 python3 quant-vgg.py
srun --ntasks=1 --cpus-per-task=12 -p gpu -q gpu-8 --gres=gpu:1 python3 quant-sq.py

### Prune the model 
srun --ntasks=1 --cpus-per-task=12 -p gpu -q gpu-8 --gres=gpu:1 python3 prune-mb1.py
srun --ntasks=1 --cpus-per-task=12 -p gpu -q gpu-8 --gres=gpu:1 python3 prune-mb2.py
srun --ntasks=1 --cpus-per-task=12 -p gpu -q gpu-8 --gres=gpu:1 python3 prune-vgg.py
srun --ntasks=1 --cpus-per-task=12 -p gpu -q gpu-8 --gres=gpu:1 python3 prune-sq.py
