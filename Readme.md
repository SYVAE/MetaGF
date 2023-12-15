# Meta Gradient Fusion: Training Dynamic-Depth Neural Networks Harmoniously

### Important Modification:
if you want to apply meta-gf to your multi-task learning program, please see the guideline in QuicklyApplication_MetaGF_ForMultitask.zip

### Install:

Pytorch>=1.9.0

### Dataset: 

CIFAR10 CIFAR100 ImageNet



### Usage:

Run the script in the folder named "Script"(select the corresponding networks)

After training finished, run the script in the "Script/TestScript"(select the corresponding networks)

Please downloading the Cifar datasets and put it to the "./data/cifar"

data
└── cifar
    ├── cifar-100-python
    │   ├── meta
    │   ├── test
    │   └── train
    ├── cifar-100-python.tar.gz
    ├── cifar-10-batches-py
    │   ├── batches.meta
    │   ├── data_batch_1
    │   ├── data_batch_2
    │   ├── data_batch_3
    │   ├── data_batch_4
    │   ├── data_batch_5
    │   ├── readme.html
    │   └── test_batch
    └── cifar-10-python.tar.gz

### Notes:

- We adopt channel-wise weighting for vgg because the layer of vgg is too less
- We adopt layer-wise weighting policy for resnet and msdnet
- We adopt EMA updating policy for the meta-weights training
- We train the ImageNet in distributed mode

This version of code may have bugs.  We will continue updating it. 

### New results:
The accuracy of Cagrad on ImageNet:
58.37/64.21/66.88/68.22/69.42


### Acknowledgements

We thanks for the public codes provided by the following works:

Li, H., Zhang, H., Qi, X., Yang, R., Huang, G.: Improved techniques for training
adaptive deep networks. In: Proceedings of the IEEE/CVF International Confer-
ence on Computer Vision. pp. 1891–1900 (2019)

Kaya, Y., Hong, S., Dumitras, T.: Shallow-deep networks: Understanding and mit-
igating network overthinking. In: International Conference on Machine Learning.
pp. 3301–3310. PMLR



