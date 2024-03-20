---
title: Pytorch | Tutorial-03 数据转换
mathjax: true
date: 2024-03-20 14:34:01
category:
  - Tutorial
tags:
  - DeepLearning
  - Pytorch
---

> 这是对 Pytorch 官网的 Tutorial 教程的中文翻译。

数据并不总是以训练机器学习算法所需的最终处理形式出现，我们使用转换来对数据执行一些操作并使其适合训练。

所有 TorchVision 数据集都有两个参数：用于修改特征的  `transform`  和用于修改标签的  `target_transform`。接受包含转换逻辑的可调用对象。 torchvision.transforms 模块提供了几种开箱即用的常用转换。

FashionMNIST 数据集的特征采用 PIL 图像格式，标签为整数。对于训练，我们需要将**特征**作为**归一化张量**，将**标签**作为**独热编码张量**。为了进行这些转换，我们使用  `ToTensor`  和  `Lambda` 。

```python
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)
```

输出：

```
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to data/FashionMNIST/raw/train-images-idx3-ubyte.gz

  0%|          | 0/26421880 [00:00<?, ?it/s]
  0%|          | 65536/26421880 [00:00<01:12, 362470.31it/s]
  1%|          | 229376/26421880 [00:00<00:38, 681259.72it/s]
  4%|3         | 950272/26421880 [00:00<00:11, 2185553.59it/s]
 15%|#4        | 3833856/26421880 [00:00<00:02, 7599317.20it/s]
 34%|###4      | 9109504/26421880 [00:00<00:00, 18310296.11it/s]
 46%|####5     | 12091392/26421880 [00:00<00:00, 17936658.84it/s]
 68%|######7   | 17924096/26421880 [00:01<00:00, 22974578.28it/s]
 89%|########9 | 23592960/26421880 [00:01<00:00, 25758355.11it/s]
100%|##########| 26421880/26421880 [00:01<00:00, 18198564.66it/s]
Extracting data/FashionMNIST/raw/train-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw/train-labels-idx1-ubyte.gz

  0%|          | 0/29515 [00:00<?, ?it/s]
100%|##########| 29515/29515 [00:00<00:00, 325487.35it/s]
Extracting data/FashionMNIST/raw/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz

  0%|          | 0/4422102 [00:00<?, ?it/s]
  1%|1         | 65536/4422102 [00:00<00:12, 362947.95it/s]
  5%|5         | 229376/4422102 [00:00<00:06, 682324.89it/s]
 21%|##1       | 950272/4422102 [00:00<00:01, 2189897.25it/s]
 87%|########6 | 3833856/4422102 [00:00<00:00, 7611069.08it/s]
100%|##########| 4422102/4422102 [00:00<00:00, 6093636.48it/s]
Extracting data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz

  0%|          | 0/5148 [00:00<?, ?it/s]
100%|##########| 5148/5148 [00:00<00:00, 39985698.13it/s]
Extracting data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw
```

## ToTensor()

ToTensor 将 PIL 图像或 NumPy `ndarray` 转换为 `FloatTensor` 。并将图像像素值缩放到 `[0., 1.]` 范围内。

## Lambda 转换

Lambda 转换应用任何用户定义的 lambda 函数。在这里，我们定义一个函数将整数转换为 one-hot 编码张量。它首先创建一个大小为 10 的零张量（数据集中的标签数量）并调用 `scatter_` ，它在标签 `y` 给出的索引上分配 `value=1` 。

```python
target_transform = Lambda(lambda y: torch.zeros(
    10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
```
