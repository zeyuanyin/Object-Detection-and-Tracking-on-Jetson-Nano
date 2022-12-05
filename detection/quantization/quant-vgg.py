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


class Args():
    net="vgg16-ssd"
    model="./models/fruit_vgg/vgg16-ssd-Epoch-99-Loss-3.182431227161038.pth"
    dataset_type="open_images"
    dataset="data/fruit"
    
    nms_method="hard"
    iou_threshold=0.5
    eval_dir="models/eval_results"
    use_2007_metric=True
    mb2_width_mult=1.0
    checkpoint_folder="./quant/"
args=Args()


DEVICE = torch.device("cuda:0")
# load the dataset
if args.dataset_type == "voc":
    dataset = VOCDataset(args.dataset, is_test=True)
elif args.dataset_type == 'open_images':
    dataset = OpenImagesDataset(args.dataset, dataset_type="test")
else:
    print(f"The dataset type  {args.dataset_type } is not supported.")


# create the network
if args.net == 'vgg16-ssd':
    net = create_vgg_ssd(len(dataset.class_names), is_test=True)
elif args.net == 'mb1-ssd':
    net = create_mobilenetv1_ssd(len(dataset.class_names), is_test=True)
elif args.net == 'mb1-ssd-lite':
    net = create_mobilenetv1_ssd_lite(len(dataset.class_names), is_test=True)
elif args.net == 'sq-ssd-lite':
    net = create_squeezenet_ssd_lite(len(dataset.class_names), is_test=True)
elif args.net == 'mb2-ssd-lite':
    net = create_mobilenetv2_ssd_lite(len(dataset.class_names), width_mult=args.mb2_width_mult, is_test=True)
else:
    logging.fatal(f"Invalid network architecture type '{arch}' - it should be one of:  vgg16-ssd, mb1-ssd, mb1-ssd-lite, mb2-ssd-lite, sq-ssd-lite")
    sys.exit(1)

# load the model
print(f"Loading model {args.model}")
net.load(args.model)

from eval_ssd import MeanAPEvaluator

# eval = MeanAPEvaluator(dataset, net, arch=args.net, eval_dir=args.eval_dir, 
#                            nms_method=args.nms_method, iou_threshold=args.iou_threshold,
#                            use_2007_metric=args.use_2007_metric, device=DEVICE)
# mean_ap, class_ap = eval.compute()
# print('base', mean_ap)
# eval.log_results(mean_ap, class_ap)


net.half()
quantized_base_net = torch.quantization.quantize_dynamic(
    net.base_net, {torch.nn.Conv2d,torch.nn.Linear}, dtype=torch.qint8
)
net.base_net = quantized_base_net

net.base_net.half()
for module in [net.base_net,net.extras,net.regression_headers,net.classification_headers]:
    # for m in module.modules():
    #     if type(m) == torch.nn.Conv2d:
    #         m.qconfig = torch.quantization.default_qconfig
    #         torch.quantization.prepare(m, inplace=True)
    quantized_base_net = torch.quantization.quantize_dynamic(module, {torch.nn.Conv2d,torch.nn.Linear}, dtype=torch.qint8)
    module = quantized_base_net

model_path = os.path.join(args.checkpoint_folder, f"{args.net}-quant-all.pth")
net.save(model_path)

eval = MeanAPEvaluator(dataset, net, arch=args.net, eval_dir=args.eval_dir, 
                           nms_method=args.nms_method, iou_threshold=args.iou_threshold,
                           use_2007_metric=args.use_2007_metric, device=DEVICE)
mean_ap, class_ap = eval.compute()
print('quant res', mean_ap)
eval.log_results(mean_ap, class_ap)
