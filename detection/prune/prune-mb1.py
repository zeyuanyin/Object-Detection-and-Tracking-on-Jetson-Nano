import torch
from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.datasets.voc_dataset import VOCDataset
from vision.datasets.open_images import OpenImagesDataset
from vision.utils import box_utils, measurements
from vision.utils.misc import str2bool, Timer
import argparse
import pathlib
import numpy as np
import logging
import sys
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor

import os
import torch.quantization
import time 
from torch.nn.utils import prune
import torch.nn as nn 
from simplify import simplify


DEVICE = torch.device("cuda:0")
# load the dataset and run with 

# load the model 
class Args():
    net="mb1-ssd"
    model="./models/fruit_mob1/mb1-ssd-Epoch-95-Loss-3.305213132212239.pth"
    dataset_type="open_images"
    dataset="data/fruit"
    checkpoint_folder="./prune/"
    nms_method="hard"
    iou_threshold=0.5
    eval_dir="models/eval_results"
    use_2007_metric=True
    mb2_width_mult=1.0
args=Args()

if args.dataset_type == "voc":
    dataset = VOCDataset(args.dataset, is_test=True)
elif args.dataset_type == 'open_images':
    dataset = OpenImagesDataset(args.dataset, dataset_type="test")
else:
    print(f"The dataset type  {args.dataset_type } is not supported.")


def prune_model_l1_structured(model, layer_type, proportion):
    for module in model.modules():
        if isinstance(module, layer_type):
            prune.ln_structured(module, 'weight', proportion, n=1, dim=0)
            prune.remove(module, 'weight')
    return model
from eval_ssd import MeanAPEvaluator

net = create_mobilenetv1_ssd(len(dataset.class_names), is_test=True)
net.load(args.model)
net.eval()
prop = 0.01
prune_model_l1_structured(net.base_net, nn.Conv2d, prop)
eval = MeanAPEvaluator(dataset, net, arch=args.net, eval_dir=args.eval_dir, 
                    nms_method=args.nms_method, iou_threshold=args.iou_threshold,
                    use_2007_metric=args.use_2007_metric, device=DEVICE)
mean_ap, class_ap = eval.compute()
print(prop, mean_ap)
eval.log_results(mean_ap, class_ap)

model_path = os.path.join(args.checkpoint_folder, f"{args.net}-l1structured.pth")
net.save(model_path)