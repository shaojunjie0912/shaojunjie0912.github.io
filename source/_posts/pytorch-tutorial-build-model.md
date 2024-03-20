---
title: Pytorch | Tutorial-04 构建神经网络模型
mathjax: true
date: 2024-03-20 14:34:08
category:
  - Tutorial
tags:
  - DeepLearning
  - Pytorch
---

> 这是对 Pytorch 官网的 Tutorial 教程的中文翻译。

神经网络由对数据执行操作的层/模块组成。 torch.nn 命名空间提供了构建您自己的神经网络所需的所有构建块。 PyTorch 中的每个模块都是 nn.Module 的子类。神经网络本身就是一个模块，由其他模块（层）组成。这种嵌套结构允许轻松构建和管理复杂的架构。

在以下部分中，我们将构建一个神经网络来对 FashionMNIST 数据集中的图像进行分类。

```python
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
```

## 选择 GPU/CPU 训练

我们希望能够在 GPU 或 MPS 等硬件加速器（如果可用）上训练我们的模型。让我们检查一下 torch.cuda 或 torch.backends.mps 是否可用，否则我们使用 CPU。

```python
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using {device} device")
```

输出：

```
Using cuda device
```

## 定义神经网络类

我们通过继承 `nn.Module` 类来定义神经网络，并在 `__init__` 中初始化神经网络层。每个 `nn.Module` 子类都在 `forward` 方法中实现对输入数据的操作。

```python
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
```

创建 `NeuralNetwork` 实例，将其移动到 `device` 并打印其结构。

```python
model = NeuralNetwork().to(device)
print(model)
```

输出：

```
NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)
```

为了使用该模型，我们将输入数据传递给它。这将执行模型的 `forward` 以及一些后台操作。_不要直接调用 `model.forward()`_

将数据输入模型后，模型会返回一个二维张量，其中 dim=0 对应于 10 个类别的原始预测值（如 0，1，2，3...），dim=1 对应于每个预测值的得分（如 -10.9，5.7，6.9，2.4，66.6...）。我们通过将返回的二维张量传递给 `nn.Softmax` 模块的实例来获取预测概率。

```python
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")
```

输出：

```
Predicted class: tensor([7], device='cuda:0')
```

---

## 模型层

让我们分解 FashionMNIST 模型中的各个层。为了说明这一点，我们将用包含 3 张大小为 28x28 图像的小批量样本，看看当我们将其传递到网络时会发生什么。

```python
input_image = torch.rand(3,28,28)
print(input_image.size())
```

输出：

```
torch.Size([3, 28, 28])
```

### nn.Flatten

我们初始化 nn.Flatten 层，将每个二维 28x28 图像转换为包含 784 个像素值的连续数组（不改变小批量样本的维度，即 3）。

```python
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())
```

输出：

```
torch.Size([3, 784])
```

### nn.Linear

线性层使用其存储的权重和偏差对输入进行线性变换。

```python
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())
```

输出：

```
torch.Size([3, 20])
```

### nn.ReLU

非线性激活函数在模型输入和输出之间建立复杂的映射。在线性变换后应用非线性激活函数以引入非线性，帮助神经网络学习各种情况。

在此模型中，我们在线性层之间使用 nn.ReLU，也可以使用其他激活函数为模型引入非线性。

```python
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")
```

输出：

```
Before ReLU: tensor([[ 0.4158, -0.0130, -0.1144,  0.3960,  0.1476, -0.0690, -0.0269,  0.2690,
          0.1353,  0.1975,  0.4484,  0.0753,  0.4455,  0.5321, -0.1692,  0.4504,
          0.2476, -0.1787, -0.2754,  0.2462],
        [ 0.2326,  0.0623, -0.2984,  0.2878,  0.2767, -0.5434, -0.5051,  0.4339,
          0.0302,  0.1634,  0.5649, -0.0055,  0.2025,  0.4473, -0.2333,  0.6611,
          0.1883, -0.1250,  0.0820,  0.2778],
        [ 0.3325,  0.2654,  0.1091,  0.0651,  0.3425, -0.3880, -0.0152,  0.2298,
          0.3872,  0.0342,  0.8503,  0.0937,  0.1796,  0.5007, -0.1897,  0.4030,
          0.1189, -0.3237,  0.2048,  0.4343]], grad_fn=<AddmmBackward0>)


After ReLU: tensor([[0.4158, 0.0000, 0.0000, 0.3960, 0.1476, 0.0000, 0.0000, 0.2690, 0.1353,
         0.1975, 0.4484, 0.0753, 0.4455, 0.5321, 0.0000, 0.4504, 0.2476, 0.0000,
         0.0000, 0.2462],
        [0.2326, 0.0623, 0.0000, 0.2878, 0.2767, 0.0000, 0.0000, 0.4339, 0.0302,
         0.1634, 0.5649, 0.0000, 0.2025, 0.4473, 0.0000, 0.6611, 0.1883, 0.0000,
         0.0820, 0.2778],
        [0.3325, 0.2654, 0.1091, 0.0651, 0.3425, 0.0000, 0.0000, 0.2298, 0.3872,
         0.0342, 0.8503, 0.0937, 0.1796, 0.5007, 0.0000, 0.4030, 0.1189, 0.0000,
         0.2048, 0.4343]], grad_fn=<ReluBackward0>)
```

### nn.Sequential

nn.Sequential 是模块的有序容器。数据按照与定义相同的顺序在所有模块间传递。可以使用顺序容器来组成一个快速网络，例如 `seq_modules` 。

```python
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)
```

### nn.Softmax

神经网络最后一个线性层返回的 _logits_ （原始值）被传递到 nn.Softmax 模块。 logits 缩放到 0 和 1 之间，表示模型对每个类别的预测概率。 `dim` 参数指示维度，沿该维度值的总和必须为 1。

```python
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)
```

## 模型参数

神经网络内的许多层都是*参数化*的，即具有在训练期间优化的相关权重和偏差。`nn.Module` 的子类自动跟踪模型对象内定义的所有字段，并使所有参数可使用模型的 `parameters()` 或 `named_parameters()` 方法访问。

在此示例中，我们迭代每个参数，并打印其大小及其值的预览。

```python
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
```

输出：

```
Model structure: NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)


Layer: linear_relu_stack.0.weight | Size: torch.Size([512, 784]) | Values : tensor([[ 0.0273,  0.0296, -0.0084,  ..., -0.0142,  0.0093,  0.0135],
        [-0.0188, -0.0354,  0.0187,  ..., -0.0106, -0.0001,  0.0115]],
       device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.0.bias | Size: torch.Size([512]) | Values : tensor([-0.0155, -0.0327], device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.2.weight | Size: torch.Size([512, 512]) | Values : tensor([[ 0.0116,  0.0293, -0.0280,  ...,  0.0334, -0.0078,  0.0298],
        [ 0.0095,  0.0038,  0.0009,  ..., -0.0365, -0.0011, -0.0221]],
       device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.2.bias | Size: torch.Size([512]) | Values : tensor([ 0.0148, -0.0256], device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.4.weight | Size: torch.Size([10, 512]) | Values : tensor([[-0.0147, -0.0229,  0.0180,  ..., -0.0013,  0.0177,  0.0070],
        [-0.0202, -0.0417, -0.0279,  ..., -0.0441,  0.0185, -0.0268]],
       device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.4.bias | Size: torch.Size([10]) | Values : tensor([ 0.0070, -0.0411], device='cuda:0', grad_fn=<SliceBackward0>)
```
