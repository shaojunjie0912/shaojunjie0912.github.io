---
title: Pytorch | Tutorial-06 优化模型参数
mathjax: true
date: 2024-03-20 19:35:50
category:
  - Tutorial
tags:
  - DeepLearning
  - Pytorch
---

> 这是对 Pytorch 官网的 Tutorial 教程的中文翻译。

现在我们有了模型和数据，是时候通过优化模型参数来训练、验证和测试我们的模型了。训练模型是一个迭代过程，在每次迭代中，模型都会对输出进行预测，计算其预测的误差（损失），保存误差相对于其参数的导数，并使用梯度下降优化这些参数。有关此过程的更详细的演示，请观看 3Blue1Brown 的有关[反向传播](https://www.bilibili.com/video/BV16x411V7Qg/?spm_id_from=333.788.recommend_more_video.0&vd_source=4f4a4c16e4377d637b62ca3ef8d16c69)的视频。

## 准备工作

我们使用之前加载数据集以及构建模型的代码。

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

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

model = NeuralNetwork()
```

## 超参数

超参数是可调整的参数，可让您控制模型优化过程。不同的超参数值会影响模型训练和收敛速度（阅读有关超参数调整的[更多信息](https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html)）

我们定义以下训练超参数：

- epoch：数据集迭代次数
- batch size：参数更新之前通过网络传播的数据样本数量
- 学习率：每个批次/时期更新模型参数的量。较小的值会导致学习速度较慢，而较大的值可能会导致训练期间出现不可预测的行为

```python
learning_rate = 1e-3
batch_size = 64
epochs = 5
```

## 优化循环

一旦我们设置了超参数，我们就可以使用优化循环来训练和优化我们的模型。优化循环的每次迭代称为一个 epoch。

每个 epoch 由两个主要部分组成：

- 训练循环：迭代训练数据集并尝试收敛到最佳参数
- 验证/测试循环：迭代测试数据集以检查模型性能是否有所改善

让我们简单熟悉一下训练循环中使用的一些概念。

### 损失函数

当提供一些训练数据时，我们未经训练的网络可能不会给出正确的答案。损失函数衡量的是预测结果与真实结果的偏差程度，我们在训练时就是想要最小化损失函数。为了计算损失，我们使用给定数据样本的输入进行预测，并将其与真实数据标签值进行比较。

常见的损失函数包括用于回归任务的 `nn.MSELoss`（均方误差）和用于分类的 `nn.NLLLoss`（负对数似然）。 `nn.CrossEntropyLoss`（交叉熵损失） 结合了 `nn.LogSoftmax` 和 `nn.NLLLoss` 。

我们将模型的输出 logits 传递给 `nn.CrossEntropyLoss` ，这将标准化 logits 并计算预测误差。

```python
# 初始化损失函数
loss_fn = nn.CrossEntropyLoss()
```

### 优化器

优化是调整模型参数以减少每个训练步骤中模型误差的过程。优化算法定义了如何执行此过程（在本例中我们使用随机梯度下降）。所有优化逻辑都封装在 `optimizer` 对象中。这里，我们使用SGD优化器；此外，PyTorch 中还有许多不同的优化器，例如 ADAM 和 RMSProp，它们可以更好地处理不同类型的模型和数据。

我们传入需要训练的模型参数和学习率超参数来初始化优化器。

```python
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
```

在训练循环内，优化分三个步骤进行：

- 调用 `optimizer.zero_grad()` 重置模型参数的梯度。默认情况下渐变相加；为了防止重复计算，我们在每次迭代时明确地将它们归零。
- 通过调用 `loss.backward()` 反向传播预测损失。 PyTorch 存储损失关于每个参数的梯度。
- 一旦有了梯度，我们就调用 `optimizer.step()` 通过反向传播中收集的梯度来调整参数。

## 完整的实现

我们定义了循环优化代码的 `train_loop` 和根据测试数据评估模型性能的 `test_loop` 。

```python
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # 将 model 设置为训练模式
    # 对于批量归一化和 dropout 层很重要
    # 这里不必要，只是为了实践考虑
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # 计算预测值和损失
        pred = model(X)
        loss = loss_fn(pred, y)

        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
	# 将 model 设置为评估模式
    # 对于批量归一化和 dropout 层很重要
    # 这里不必要，只是为了实践考虑
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # 用 torch.no_grad() 评估模型，确保评估模式下不会计算梯度
    # 也用于减少在 requires_grad=True 时不必要的梯度计算和张量的内存使用
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```

我们初始化损失函数和优化器，并将其传递给 `train_loop` 和 `test_loop` 。您可以随意增加 epoch 来跟踪模型性能的改进。

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
```

输出：

```
Epoch 1
-------------------------------
loss: 2.298730  [   64/60000]
loss: 2.289123  [ 6464/60000]
loss: 2.273286  [12864/60000]
loss: 2.269406  [19264/60000]
loss: 2.249603  [25664/60000]
loss: 2.229407  [32064/60000]
loss: 2.227368  [38464/60000]
loss: 2.204261  [44864/60000]
loss: 2.206193  [51264/60000]
loss: 2.166651  [57664/60000]
Test Error:
 Accuracy: 50.9%, Avg loss: 2.166725

Epoch 2
-------------------------------
loss: 2.176750  [   64/60000]
loss: 2.169595  [ 6464/60000]
loss: 2.117500  [12864/60000]
loss: 2.129272  [19264/60000]
loss: 2.079674  [25664/60000]
loss: 2.032928  [32064/60000]
loss: 2.050115  [38464/60000]
loss: 1.985236  [44864/60000]
loss: 1.987887  [51264/60000]
loss: 1.907162  [57664/60000]
Test Error:
 Accuracy: 55.9%, Avg loss: 1.915486

Epoch 3
-------------------------------
loss: 1.951612  [   64/60000]
loss: 1.928685  [ 6464/60000]
loss: 1.815709  [12864/60000]
loss: 1.841552  [19264/60000]
loss: 1.732467  [25664/60000]
loss: 1.692914  [32064/60000]
loss: 1.701714  [38464/60000]
loss: 1.610632  [44864/60000]
loss: 1.632870  [51264/60000]
loss: 1.514263  [57664/60000]
Test Error:
 Accuracy: 58.8%, Avg loss: 1.541525

Epoch 4
-------------------------------
loss: 1.616448  [   64/60000]
loss: 1.582892  [ 6464/60000]
loss: 1.427595  [12864/60000]
loss: 1.487950  [19264/60000]
loss: 1.359332  [25664/60000]
loss: 1.364817  [32064/60000]
loss: 1.371491  [38464/60000]
loss: 1.298706  [44864/60000]
loss: 1.336201  [51264/60000]
loss: 1.232145  [57664/60000]
Test Error:
 Accuracy: 62.2%, Avg loss: 1.260237

Epoch 5
-------------------------------
loss: 1.345538  [   64/60000]
loss: 1.327798  [ 6464/60000]
loss: 1.153802  [12864/60000]
loss: 1.254829  [19264/60000]
loss: 1.117322  [25664/60000]
loss: 1.153248  [32064/60000]
loss: 1.171765  [38464/60000]
loss: 1.110263  [44864/60000]
loss: 1.154467  [51264/60000]
loss: 1.070921  [57664/60000]
Test Error:
 Accuracy: 64.1%, Avg loss: 1.089831

Epoch 6
-------------------------------
loss: 1.166889  [   64/60000]
loss: 1.170514  [ 6464/60000]
loss: 0.979435  [12864/60000]
loss: 1.113774  [19264/60000]
loss: 0.973411  [25664/60000]
loss: 1.015192  [32064/60000]
loss: 1.051113  [38464/60000]
loss: 0.993591  [44864/60000]
loss: 1.039709  [51264/60000]
loss: 0.971077  [57664/60000]
Test Error:
 Accuracy: 65.8%, Avg loss: 0.982440

Epoch 7
-------------------------------
loss: 1.045165  [   64/60000]
loss: 1.070583  [ 6464/60000]
loss: 0.862304  [12864/60000]
loss: 1.022265  [19264/60000]
loss: 0.885213  [25664/60000]
loss: 0.919528  [32064/60000]
loss: 0.972762  [38464/60000]
loss: 0.918728  [44864/60000]
loss: 0.961629  [51264/60000]
loss: 0.904379  [57664/60000]
Test Error:
 Accuracy: 66.9%, Avg loss: 0.910167

Epoch 8
-------------------------------
loss: 0.956964  [   64/60000]
loss: 1.002171  [ 6464/60000]
loss: 0.779057  [12864/60000]
loss: 0.958409  [19264/60000]
loss: 0.827240  [25664/60000]
loss: 0.850262  [32064/60000]
loss: 0.917320  [38464/60000]
loss: 0.868384  [44864/60000]
loss: 0.905506  [51264/60000]
loss: 0.856353  [57664/60000]
Test Error:
 Accuracy: 68.3%, Avg loss: 0.858248

Epoch 9
-------------------------------
loss: 0.889765  [   64/60000]
loss: 0.951220  [ 6464/60000]
loss: 0.717035  [12864/60000]
loss: 0.911042  [19264/60000]
loss: 0.786085  [25664/60000]
loss: 0.798370  [32064/60000]
loss: 0.874939  [38464/60000]
loss: 0.832796  [44864/60000]
loss: 0.863254  [51264/60000]
loss: 0.819742  [57664/60000]
Test Error:
 Accuracy: 69.5%, Avg loss: 0.818780

Epoch 10
-------------------------------
loss: 0.836395  [   64/60000]
loss: 0.910220  [ 6464/60000]
loss: 0.668506  [12864/60000]
loss: 0.874338  [19264/60000]
loss: 0.754805  [25664/60000]
loss: 0.758453  [32064/60000]
loss: 0.840451  [38464/60000]
loss: 0.806153  [44864/60000]
loss: 0.830360  [51264/60000]
loss: 0.790281  [57664/60000]
Test Error:
 Accuracy: 71.0%, Avg loss: 0.787271

Done!
```
