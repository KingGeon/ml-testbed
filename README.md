<div align="center">

# Mongoose AI Testbed

[![python](https://img.shields.io/badge/-Python_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)

</div>

<br>

## 📌  Introduction

원하는 experiment 환경을 넣어 모델 실험을 수행 및 로깅

+ wandb 가입 요망
+ 코드 신규작성 / 수정 시, linting 에 유의

```
python src/train.py experiment=cifar/resnet
```

## Datasets
+ MNIST
+ CIFAR10

## Models
### Image Classification
+ MLP
+ CNN
+ GoogleNet (Inception Module)
+ ResNet, Previous Activated ResNet
