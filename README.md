# Object-Detection-and-Tracking-on-Jetson-Nano
> Group 06's Project for ML701@MBZUAI

## Image Classification

1) Follow the [repo](https://github.com/weiaicunzai/pytorch-cifar100) to get the base models.
2) Use the quantization/pruning methods to get a light models.
3) Evaluate the light models and record the performance.

#### Train baseline algorithms.
```
python train.py -net vgg16 -gpu
python train.py -net mobilenet -gpu
python train.py -net squeezenet -gpu
```

#### Evaluate or prune the learned model 
Both code of evaluation and prune are at `classification/prune+test.py` file. 

When only evaluation, comment the code (line 50-60) of pruning.

Example：
```
python test.py -net vgg16 -weights ./checkpoint/vgg16/Sunday_04_December_2022_03h_07m_25s/vgg16-200-regular.pth
python test.py -net squeezenet -weights ./checkpoint/squeezenet/Sunday_04_December_2022_15h_25m_46s/squeezenet-200-regular.pth
python test.py -net mobilenet -weights ./checkpoint/mobilenet/Sunday_04_December_2022_01h_15m_53s/mobilenet-200-regular.pth
```



#### Quantize the learned model 
For quantization, run jupyter notebook script `classification/quant_model.ipynb`.



## Object Detection

1) Follow the [repo](https://github.com/dusty-nv/pytorch-ssd) to get the base models.
2) Use the quantization/pruning methods to get a light models.
3) Evaluate the light models and record the performance.
4) Export the pytorch models into the onnx files and run it at Jetson Nano.


## Object Tracking

1) Export the pytorch models into the onnx files.
2) Evaluate the onnx models on the video data at Jetson Nano.


### Other

The base models and pruned/quantized models and the corresponding onnx files are uploaded into the [OneDirve](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zeyuan_yin_mbzuai_ac_ae/EhS7id5SfKdDnI7Ygcd_pHYBEt9hHw-c97kj_hIVhG6tSw?e=pNb4Df). (People in MBZUAI with the link can view)
