# Object-Detection-and-Tracking-on-Jetson-Nano
> Group 06's Project for ML701@MBZUAI

---

## Image Classification

1) Follow the [repo](https://github.com/weiaicunzai/pytorch-cifar100) to get the base models.
2) Use the quantization/pruning methods to get a light models.
3) Evaluate the light models and record the performance.

### Train baseline algorithms.
```
python train.py -net vgg16 -gpu
python train.py -net mobilenet -gpu
python train.py -net squeezenet -gpu
```

### Evaluate or prune the learned model 
Both code of evaluation and prune are at `classification/prune+test.py` file. 

When only evaluation, comment the code (line 50-60) of pruning.

Example：
```
python test.py -net vgg16 -weights ./checkpoint/vgg16/Sunday_04_December_2022_03h_07m_25s/vgg16-200-regular.pth
python test.py -net squeezenet -weights ./checkpoint/squeezenet/Sunday_04_December_2022_15h_25m_46s/squeezenet-200-regular.pth
python test.py -net mobilenet -weights ./checkpoint/mobilenet/Sunday_04_December_2022_01h_15m_53s/mobilenet-200-regular.pth
```



### Quantize the learned model 
For quantization, run jupyter notebook script `classification/quant_model.ipynb`.

---

## Object Detection

1) Follow the [repo](https://github.com/dusty-nv/pytorch-ssd) to get the base models.
2) Use the quantization/pruning methods to get a light models.
3) Evaluate the light models and record the performance.
4) Export the pytorch models into the onnx files and run it at Jetson Nano.



We highly recommend using CSCC/CIAI cluster to speed up your training, by adding:
```
srun --ntasks=1 --cpus-per-task=12 -p gpu -q gpu-8 --gres=gpu:1 
```
before the code. 

### Train baseline algorithms.
```
python train_ssd.py --net vgg16-ssd --data=data/fruit --pretrained-ssd='models/vgg16-ssd-mp-0_7726.pth' --model-dir=models/fruit_vgg  --batch-size 6 --lr 0.0003 --epochs=100
python train_ssd.py --net mb1-ssd --data=data/fruit --pretrained-ssd='models/mobilenet-v1-ssd-mp-0_675.pth' --model-dir=models/fruit_mob1 --epochs=100  --batch-size 6 --lr 0.0003
python train_ssd.py --net mb2-ssd-lite --data=data/fruit --pretrained-ssd='models/mb2-ssd-lite-mp-0_686.pth' --model-dir=models/fruit_mb2 --epochs=100 --batch-size 6 --lr 0.0003
python train_ssd.py --net sq-ssd-lite- --data=data/fruit --model-dir=models/fruit_sq --epochs=100 --batch-size 6 --lr 0.0003
```

### Evaluate the learned model 
```
python eval_ssd.py --model=models/fruit_mb2/mb2-ssd-lite-Epoch-99-Loss-3.320802785504249.pth --label_file=models/fruit/labels.txt --net="mb2-ssd-lite" --dataset_type="open_images" --dataset=data/fruit/
python eval_ssd.py --model=models/fruit_mob1/mb1-ssd-Epoch-95-Loss-3.305213132212239.pth --label_file=models/fruit/labels.txt --net="mb1-ssd" --dataset_type="open_images" --dataset=data/fruit/
python eval_ssd.py --model=models/fruit_vgg/vgg16-ssd-Epoch-99-Loss-3.460213225887668.pth --label_file=models/fruit/labels.txt --net="vgg16-ssd" --dataset_type="open_images" --dataset=data/fruit/
```

### Quant the learned model 
```
python quant-mb1.py
python quant-mb2.py
python quant-vgg.py
python quant-sq.py
```
### Prune the model 
```
python prune-mb1.py
python prune-mb2.py
python prune-vgg.py
python prune-sq.py
```

---

## Object Tracking

1) Export the pytorch models into the onnx files.
2) Evaluate the onnx models on the video data at Jetson Nano.

---

## Other

The base models and pruned/quantized models and the corresponding onnx files are uploaded into the [OneDirve](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zeyuan_yin_mbzuai_ac_ae/EhS7id5SfKdDnI7Ygcd_pHYBEt9hHw-c97kj_hIVhG6tSw?e=pNb4Df). (People in MBZUAI with the link can view)
