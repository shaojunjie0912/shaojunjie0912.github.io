---
title: Pytorch | Tutorial-02 数据集和数据加载器
mathjax: true
date: 2024-03-20 14:33:10
category:
  - Tutorial
tags:
  - DeepLearning
  - Pytorch
---

> 这是对 Pytorch 官网的 Tutorial 教程的中文翻译。

用于处理数据样本的代码可能会变得混乱且难以维护。理想情况下，我们希望数据集代码与模型训练代码分离，以获得更好的代码可读性和模块化性。 PyTorch 提供了两个数据基元： `torch.utils.data.DataLoader` 和 `torch.utils.data.Dataset` ，允许您使用预加载的数据集以及您自己的数据。

## Dataset

> An abstract class representing a `Dataset`. All datasets that represent a map from keys to data samples should subclass it.

`torch.utils.data.Dataset`：存储样本及其相应的标签

## DataLoader

> Data loader combines a dataset and a sampler, and provides an iterable over the given dataset.

> DataLoader 组合数据集和采样器，并在给定数据集上提供一个可迭代对象

`torch.utils.data.DataLoader`：将 `Dataset` 包装为可迭代对象（**不是迭代器**），以便轻松访问样本。

## 加载数据集

以下是从 TorchVision 加载 Fashion-MNIST 数据集的示例。 Fashion-MNIST 由 60,000 个训练示例和 10,000 个测试示例组成。每个示例包含一个 28×28 灰度图像和来自 10 个类别之一的关联标签。

我们使用以下参数加载 FashionMNIST 数据集：

- `root` 存储训练/测试数据的路径；

- `train` 指定训练或测试数据集；

- `download=True`：如果 `root` 路径没有数据，则自动下载；

- `transform` 和 `target_transform`：指定特征和标签转换；

```python
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


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
```

输出：

```
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to data/FashionMNIST/raw/train-images-idx3-ubyte.gz

  0%|          | 0/26421880 [00:00<?, ?it/s]
  0%|          | 65536/26421880 [00:00<01:13, 360295.74it/s]
  1%|          | 229376/26421880 [00:00<00:38, 677593.53it/s]
  3%|3         | 884736/26421880 [00:00<00:12, 2010113.37it/s]
 11%|#1        | 2949120/26421880 [00:00<00:04, 5687069.56it/s]
 33%|###3      | 8749056/26421880 [00:00<00:01, 15106693.68it/s]
 54%|#####4    | 14385152/26421880 [00:01<00:00, 20420624.75it/s]
 77%|#######6  | 20283392/26421880 [00:01<00:00, 24317188.58it/s]
 98%|#########8| 25919488/26421880 [00:01<00:00, 26417716.71it/s]
100%|##########| 26421880/26421880 [00:01<00:00, 18077316.33it/s]
Extracting data/FashionMNIST/raw/train-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw/train-labels-idx1-ubyte.gz

  0%|          | 0/29515 [00:00<?, ?it/s]
100%|##########| 29515/29515 [00:00<00:00, 329447.61it/s]
Extracting data/FashionMNIST/raw/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz

  0%|          | 0/4422102 [00:00<?, ?it/s]
  1%|1         | 65536/4422102 [00:00<00:12, 363003.55it/s]
  5%|5         | 229376/4422102 [00:00<00:06, 682229.73it/s]
 21%|##1       | 950272/4422102 [00:00<00:01, 2189295.40it/s]
 86%|########5 | 3801088/4422102 [00:00<00:00, 8352312.98it/s]
100%|##########| 4422102/4422102 [00:00<00:00, 6091038.99it/s]
Extracting data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz

  0%|          | 0/5148 [00:00<?, ?it/s]
100%|##########| 5148/5148 [00:00<00:00, 30072809.18it/s]
Extracting data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw
```

## 迭代和可视化数据集

我们可以像列表一样手动索引 `Datasets` ： `training_data[index]` 。我们使用 `matplotlib` 来可视化训练数据中的一些样本。

```python
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
```

![image.png](https://raw.githubusercontent.com/shaojunjie0912/Picgo-Images/main/img/20240320153003.png)

---

## 创建自定义数据集

自定义 Dataset 类必须实现三个函数：`__init__`、`__len__` 和 `__getitem__`。

下例 FashionMNIST 图像存储在目录 `img_dir` 中，它们的标签单独存储在 CSV 文件 `annotations_file` 中。

```python
import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
```

### `__init__`

`__init__` 函数在实例化 Dataset 对象时运行一次，初始化*包含图片的目录*、*标签文件*和*转换选项*。

labels.csv 文件：

```
tshirt1.jpg, 0
tshirt2.jpg, 0
......
ankleboot999.jpg, 9
```

### `__len__`

`__len__` 函数返回数据集中的样本数。

### `__getitem__`

`__getitem__` 函数加载并返回给定索引 `idx` 处的数据集的样本。基于索引，它识别图片目录，使用 `read_image` 将其转换为张量，从 `self.img_labels` 中的 csv 文件中检索相应的标签，对其调用转换函数（如果适用），并返回张量图像和相应的标签。

---

## 使用 DataLoader 准备数据以进行训练

`Dataset` 检索数据集的特征并一次标记一个样本。在训练模型时，我们通常希望以“小批量”的方式传递样本，在每个时期重新整理数据以减少模型过度拟合，并使用 Python 的 `multiprocessing` 来加速数据检索。

`DataLoader` 是一个可迭代对象，它通过一个简单的 API 为我们抽象了这种复杂性。

```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
```

## 遍历 DataLoader

我们已将该数据集加载到 `DataLoader` 中，并且可以根据需要迭代数据集。下面的每次迭代都一批 `train_features` 和 `train_labels` （分别包含 `batch_size=64` 个特征和标签）。因为我们指定了 `shuffle=True` ，所以在迭代所有批次后，数据会被打乱。

```python
# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
```

![image.png](https://raw.githubusercontent.com/shaojunjie0912/Picgo-Images/main/img/20240320153151.png)

输出：

```
Feature batch shape: torch.Size([64, 1, 28, 28])
Labels batch shape: torch.Size([64])
Label: 5
```
