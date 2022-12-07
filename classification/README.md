### For classification, we use this [repo](https://github.com/weiaicunzai/pytorch-cifar100) as base model. First download it.

# Scripts to reproduce the result in the project.

### Train baseline algorithms.
`python train.py -net vgg16 -gpu`

`python train.py -net mobilenet -gpu`

`python train.py -net squeezenet -gpu`

### Evaluate or prune the learned model 
Both code of evaluation and prune are at `prune+test.py` file. 

When only evaluation, comment the code (line 50-60) of pruning.

Example：

`python test.py -net vgg16 -weights ./checkpoint/vgg16/Sunday_04_December_2022_03h_07m_25s/vgg16-200-regular.pth`

`python test.py -net squeezenet -weights ./checkpoint/squeezenet/Sunday_04_December_2022_15h_25m_46s/squeezenet-200-regular.pth`

`python test.py -net mobilenet -weights ./checkpoint/mobilenet/Sunday_04_December_2022_01h_15m_53s/mobilenet-200-regular.pth`



### Quant the learned model 
For quantization, run jupyter notebook script `quant_model.ipynb`.
