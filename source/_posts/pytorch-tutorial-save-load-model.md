---
title: Pytorch | Tutorial-07 保存和加载模型
mathjax: true
date: 2024-03-20 19:52:16
category:
  - Tutorial
tags:
  - DeepLearning
  - Pytorch
---

> 这是对 Pytorch 官网的 Tutorial 教程的中文翻译。

在本节中，我们将了解如何通过保存、加载和运行模型预测来持久保存模型状态。

```python
import torch
import torchvision.models as models
```

## 保存和加载模型权重

PyTorch 模型将学习到的参数存储在内部状态字典中，称为 `state_dict` 。这些可以通过 `torch.save` 方法保存：

```python
model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'model_weights.pth')
```

```
Downloading: "https://download.pytorch.org/models/vgg16-397923af.pth" to /var/lib/jenkins/.cache/torch/hub/checkpoints/vgg16-397923af.pth

  0%|          | 0.00/528M [00:00<?, ?B/s]
  2%|2         | 12.7M/528M [00:00<00:04, 133MB/s]
  5%|5         | 26.8M/528M [00:00<00:03, 142MB/s]
  8%|7         | 40.9M/528M [00:00<00:03, 145MB/s]
 10%|#         | 55.0M/528M [00:00<00:03, 146MB/s]
 13%|#3        | 69.1M/528M [00:00<00:03, 147MB/s]
 16%|#5        | 83.3M/528M [00:00<00:03, 147MB/s]
 18%|#8        | 97.4M/528M [00:00<00:03, 148MB/s]
 21%|##1       | 112M/528M [00:00<00:02, 148MB/s]
 24%|##3       | 126M/528M [00:00<00:02, 148MB/s]
 26%|##6       | 140M/528M [00:01<00:02, 148MB/s]
 29%|##9       | 154M/528M [00:01<00:02, 148MB/s]
 32%|###1      | 168M/528M [00:01<00:02, 148MB/s]
 35%|###4      | 182M/528M [00:01<00:02, 148MB/s]
 37%|###7      | 196M/528M [00:01<00:02, 148MB/s]
 40%|###9      | 211M/528M [00:01<00:02, 148MB/s]
 43%|####2     | 225M/528M [00:01<00:02, 148MB/s]
 45%|####5     | 239M/528M [00:01<00:02, 148MB/s]
 48%|####7     | 253M/528M [00:01<00:01, 148MB/s]
 51%|#####     | 267M/528M [00:01<00:01, 148MB/s]
 53%|#####3    | 281M/528M [00:02<00:01, 148MB/s]
 56%|#####5    | 295M/528M [00:02<00:01, 148MB/s]
 59%|#####8    | 309M/528M [00:02<00:01, 148MB/s]
 61%|######1   | 324M/528M [00:02<00:01, 148MB/s]
 64%|######3   | 338M/528M [00:02<00:01, 148MB/s]
 67%|######6   | 352M/528M [00:02<00:01, 148MB/s]
 69%|######9   | 366M/528M [00:02<00:01, 148MB/s]
 72%|#######2  | 380M/528M [00:02<00:01, 148MB/s]
 75%|#######4  | 394M/528M [00:02<00:00, 148MB/s]
 77%|#######7  | 408M/528M [00:02<00:00, 148MB/s]
 80%|########  | 423M/528M [00:03<00:00, 148MB/s]
 83%|########2 | 437M/528M [00:03<00:00, 148MB/s]
 85%|########5 | 451M/528M [00:03<00:00, 148MB/s]
 88%|########8 | 465M/528M [00:03<00:00, 148MB/s]
 91%|######### | 479M/528M [00:03<00:00, 142MB/s]
 93%|#########3| 493M/528M [00:03<00:00, 143MB/s]
 96%|#########6| 507M/528M [00:03<00:00, 144MB/s]
 99%|#########8| 521M/528M [00:03<00:00, 146MB/s]
100%|##########| 528M/528M [00:03<00:00, 147MB/s]
```

要加载模型权重，您需要先创建同一模型的实例，然后使用 `load_state_dict()` 方法加载参数。

```python
model = models.vgg16() # we do not specify ``weights``, i.e. create untrained model
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()
```

```
VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace=True)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace=True)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace=True)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)
```

> 请务必在推理之前调用 `model.eval()` 方法，将 dropout 和批量归一化层设置为评估模式。如果不这样做将会产生不一致的推理结果。

## 保存和加载带有形状的模型

加载模型权重时，我们需要首先实例化模型类，因为该类定义了网络的结构。我们可能希望将此类的结构与模型一起保存，在这种情况下，我们可以将 `model` （而不是 `model.state_dict()` ）传递给保存函数：

```python
torch.save(model, 'model.pth')
```

然后我们可以像这样加载模型：

```python
model = torch.load('model.pth')
```

> 此方法在序列化模型时使用 Python pickle 模块，因此它依赖于加载模型时实际可用的类的定义。
