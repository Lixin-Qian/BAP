# Blind Adversarial Pruning

This repository contains the source code for Blind Adversarial Pruning (BAP), a Python library to reappearance our BAP results.

## Related articles 

This related articles of this code are listed follows,

[BAP in arXiv](https://arxiv.org/abs/2004.05913) 
[1] Haidong Xie, Lixin Qian, Xueshuang Xiang*, Naijin Liu, Blind Adversarial Pruning: Towards the Comprehensive Robust Models with Gradually Pruning Against Blind Adversarial Attacks, IEEE International Conference on Multimedia and Expo (ICME) 2021, in press, Shenzhen, China. 

[BAT in arXiv](https://arxiv.org/abs/2004.05914) 
[2] Haidong Xie, Xueshuang Xiang*, Naijin Liu, Bin Dong, Blind Adversarial Training: Towards the Comprehensively Robust Models against Blind Adversarial Attack, arXiv 2004.05914. 

If you use this code for academic research, you are highly encouraged to cite this paper.
```
@article{xie2020blind,
      title={Blind Adversarial Pruning: Balance Accuracy, Efficiency and Robustness}, 
      author={Haidong Xie and Lixin Qian and Xueshuang Xiang and Naijin Liu},
      year={2020},
      journal={arXiv preprint arXiv:2004.05913},
}
```
## Setting up

### Dependency libraries:

This code is writing in Python, 

dependence on tensorflow: https://github.com/tensorflow/tensorflow ，

cleverhans: https://github.com/tensorflow/cleverhans.

and tensorflow/model-optimization:  https://github.com/tensorflow/model-optimization

### Currently supported setups

Imagine that our program does not rely on the underlying Library of a specific version, but considering the hidden modification and matching problems that version change may bring, list the version number used in the calculation. 

Python v2.7.3

Tensorflow-gpu v1.15.0

Cleverhans v3.0.1

Tensorflow-model-optimization v0.2.1

## Tutorials

To help you get started with this code, we perform the following tutorial code for example:

1. Train BAP model:
```
python cal_model.py 
```
2. Evaluate the trained model
```
python cal_acc.py 
```
The parameters within BAP have been preset in the code and can be modified as needed. 

## Copyright

Copyright 2018 - Qian Xuesen Laboratory, China Academy of Space Technology.
