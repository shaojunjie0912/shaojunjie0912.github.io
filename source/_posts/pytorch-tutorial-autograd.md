---
title: Pytorch | Tutorial-05 autograd 自动微分
mathjax: true
date: 2024-03-20 14:34:15
category:
  - Tutorial
tags:
  - DeepLearning
  - Pytorch
---

> 这是对 Pytorch 官网的 Tutorial 教程的中文翻译。

在训练神经网络时，最常用的算法是反向传播。在该算法中，根据损失函数相对于给定参数的梯度来调整参数（模型权重）。

为了计算这些梯度，PyTorch 有一个名为 `torch.autograd` 的内置微分引擎。它能自动计算任何计算图的梯度。

考虑最简单的单层神经网络，具有输入 `x` 、参数 `w` 和 `b` 以及一些损失函数。它可以通过以下方式在 PyTorch 中定义：

```python
import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
```

## 张量、函数和计算图

上述代码定义了以下计算图：

![计算图](https://raw.githubusercontent.com/shaojunjie0912/Picgo-Images/main/img/20240320103252.png)

在这个网络中， `w` 和 `b` 是我们需要优化的参数。因此，我们需要能够计算损失函数相对于这些变量的梯度。为此，我们设置这些张量的 `requires_grad` 属性。

> 可以在创建张量时设置 `requires_grad` 的值，或者之后使用 `x.requires_grad_(True)` 方法设置。

构造张量的计算图的函数实际上是类 `Function` 的对象。该对象可以完成前向传播和反向传播的函数计算。反向传播函数的声明在张量的 `grad_fn` 属性中。您可以在文档中找到 `Function` 的更多信息。

```python
print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")
```

输出：

```
Gradient function for z = <AddBackward0 object at 0x7f47dfe73310>
Gradient function for loss = <BinaryCrossEntropyWithLogitsBackward0 object at 0x7f47dfe71060>
```

## 计算梯度

为了优化神经网络中参数的权重，我们需要计算损失函数相对于参数的导数，即，我们需要 `x` 和 `y` 为固定值下的 $\frac{\partial{loss}}{\partial{w}}$ 和 $\frac{\partial{loss}}{\partial{b}}$。为了计算这些导数，我们调用 `loss.backward()` ，然后通过 `w.grad` 和 `b.grad` 获取导数值：

```python
loss.backward()
print(w.grad)
print(b.grad)
```

输出：

```
tensor([[0.3313, 0.0626, 0.2530],
        [0.3313, 0.0626, 0.2530],
        [0.3313, 0.0626, 0.2530],
        [0.3313, 0.0626, 0.2530],
        [0.3313, 0.0626, 0.2530]])
tensor([0.3313, 0.0626, 0.2530])
```

> - 我们只能获取计算图中叶节点的 `grad` 属性，其中 `requires_grad` 属性设置为 `True` 。对于我们图中的所有其他节点，梯度将不可用。
> - 出于性能原因，我们只能在给定图上使用 `backward` 执行一次梯度计算。如果我们需要在同一个图上执行多个 `backward` 调用，则需要将 `retain_graph=True` 传递给 `backward` 调用。

## 禁用梯度追踪

默认情况下，所有具有 `requires_grad=True` 的张量都会跟踪其计算历史并支持梯度计算。然而，在某些情况下，我们不需要这样做，例如，当我们训练了模型并且只想将其应用于某些输入数据时，即我们只想将网络用于前向计算。

第一种方法：用 `torch.no_grad()` 块包围代码来停止跟踪计算：

```python
z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)
```

输出：

```
True
False
```

第二种方法：在张量上使用 `detach()` 方法：

```python
z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)
```

```
False
```

您可能想要禁用梯度跟踪的原因有：

- 将神经网络中的某些参数标记为**冻结参数**
- 仅进行前向计算时**加快计算速度**，因为对不跟踪梯度的张量进行计算会更有效

## 有关计算图的更多信息

从概念上讲，autograd 将所有数据（张量）和执行操作（以及生成的新张量）的记录保存在由 Function 对象组成的有向无环图 (DAG) 中。在 DAG 中，叶子是输入张量，根是输出张量。通过从根到叶追踪该计算图，您可以使用链式法则自动计算梯度。

在前向传播计算中，autograd 同时做两件事：

- 运行请求的操作来计算输出张量
- 在 DAG 中维护操作的梯度函数

当在 DAG 根上调用 `.backward()` 时，后向传播计算开始。 autograd 会：

- 计算每个 `.grad_fn` 的梯度，
- 将它们累积到相应张量的 `.grad` 属性中
- 使用链式法则，一直传播到叶张量。

> DAG 在 PyTorch 中是动态的。需要注意的是：计算图是从头开始重新创建的；每次 `.backward()` 调用后，autograd 开始填充新计算图。这正是允许您在模型中使用控制语句的原因；如果需要，您可以在每次迭代时更改形状、大小和操作。

## 可选阅读：张量梯度和雅可比积

大部分情况下，我们有一个标量损失函数，并且需要计算某些参数的梯度。但在某些情况下，输出函数是任意张量。在这种情况下，PyTorch 允许您计算所谓的雅可比积，而不是实际的梯度。

对于向量函数 $\vec{y}=f(\vec{x})$ ，其中 $\vec{x}=\left\langle x_{1}, \ldots, x_{n}\right\rangle$ 和 $\vec{y}=\left\langle y_{1}, \ldots, y_{m}\right\rangle$ ， $\vec{y}$ 相对于 $\vec{x}$ 的梯度为由雅可比矩阵给出：

$$
J=\left(\begin{array}{ccc}\frac{\partial y_{1}}{\partial x_{1}} & \cdots & \frac{\partial y_{1}}{\partial x_{n}} \\\vdots & \ddots & \vdots \\\frac{\partial y_{m}}{\partial x_{1}} & \cdots & \frac{\partial y_{m}}{\partial x_{n}}\end{array}\right)
$$

PyTorch 允许您计算给定输入向量 $v=\left(v_{1} \ldots v_{m}\right)$ 的雅可比积 $v^{T} \cdot J$，而不是计算雅可比矩阵本身。这是通过使用 $v$ 作为参数调用 `backward` 来实现的。考虑到我们要计算乘积，因此 $v$  的大小应该与原始张量的大小相同：

```python
inp = torch.eye(4, 5, requires_grad=True)
out = (inp+1).pow(2).t()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"First call\n{inp.grad}")
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nSecond call\n{inp.grad}")
inp.grad.zero_()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nCall after zeroing gradients\n{inp.grad}")
```

输出：

```
First call
tensor([[4., 2., 2., 2., 2.],
        [2., 4., 2., 2., 2.],
        [2., 2., 4., 2., 2.],
        [2., 2., 2., 4., 2.]])

Second call
tensor([[8., 4., 4., 4., 4.],
        [4., 8., 4., 4., 4.],
        [4., 4., 8., 4., 4.],
        [4., 4., 4., 8., 4.]])

Call after zeroing gradients
tensor([[4., 2., 2., 2., 2.],
        [2., 4., 2., 2., 2.],
        [2., 2., 4., 2., 2.],
        [2., 2., 2., 4., 2.]])
```

请注意，当我们使用相同的参数第二次调用 `backward` 时，梯度的值是不同的。发生这种情况是因为在进行 `backward` 传播时，PyTorch 会累积梯度值，即：将计算出的梯度值添加到计算图所有叶节点的 `grad` 属性中。如果要计算正确的梯度，则需要先将 `grad` 属性清零。在训练中，优化器可以帮助我们做到这一点。

> 以前我们调用不带参数的 `backward()` 函数。这本质上相当于调用 `backward(torch.tensor(1.0))` ，这是对于标量函数的情况（例如神经网络训练期间的损失）计算梯度的有效方法。
