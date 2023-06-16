import os, gzip, torch
import torch.nn as nn
import numpy as np
import imageio
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 导入mnist数据集
def dataloader(dataset, input_size, batch_size, split='train'):
    transform = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor(), transforms.Normalize(mean=(0.5), std=(0.5))])
    data_loader = DataLoader(
        datasets.MNIST('data/mnist', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True)
    return data_loader

def load_mnist(dataset):
    # 拼接数据目录路径
    data_dir = os.path.join("../data", dataset)

    def extract_data(filename, num_data, head_size, data_size):
        # 使用gzip打开文件
        with gzip.open(filename) as bytestream:
            # 读取文件头部信息
            bytestream.read(head_size)
            # 读取数据内容
            buf = bytestream.read(data_size * num_data)
            # 将数据转换为NumPy数组，并设定数据类型为无符号8位整数，再转换为浮点数类型
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    # 从训练集中提取图像数据
    data = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    # 将图像数据重新调整形状为(60000, 28, 28, 1)
    trX = data.reshape((60000, 28, 28, 1))

    # 从训练集中提取标签数据
    data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    # 将标签数据重新调整形状为(60000,)
    trY = data.reshape((60000))

    # 从测试集中提取图像数据
    data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    # 将图像数据重新调整形状为(10000, 28, 28, 1)
    teX = data.reshape((10000, 28, 28, 1))

    # 从测试集中提取标签数据
    data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    # 将标签数据重新调整形状为(10000,)
    teY = data.reshape((10000))

    # 将训练集和测试集的标签数据类型转换为整数类型
    trY = np.asarray(trY).astype(np.int)
    teY = np.asarray(teY)

    # 将训练集和测试集的图像数据在第0维度上连接
    X = np.concatenate((trX, teX), axis=0)
    # 将训练集和测试集的标签数据在第0维度上连接，并将数据类型转换为整数类型
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    # 设置随机数种子
    seed = 233
    np.random.seed(seed)
    # 在训练集和测试集上进行随机打乱
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    # 创建一个形状为(len(y), 10)的零矩阵，数据类型为浮点数
    y_vec = np.zeros((len(y), 10), dtype=np.float)
    # 将标签数据转换为one-hot编码形式
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1

    # 将图像数据转置为(样本数, 通道数, 图像高度, 图像宽度)的形状，并进行归一化处理
    X = X.transpose(0, 3, 1, 2) / 255.

    # 将NumPy数组转换为PyTorch张量，并设定数据类型为浮点数
    X = torch.from_numpy(X).type(torch.FloatTensor)
    y_vec = torch.from_numpy(y_vec).type(torch.FloatTensor)

    return X, y_vec

# 保存图片
def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    image = (image * 255).astype(np.uint8)
    image = np.expand_dims(image, axis=2)
    print(image.shape)
    # 假设 image 是一个三维数组，形状为 (height, width, channels)
    image_tensor = torch.from_numpy(image.transpose((2, 0, 1))).unsqueeze(0)
    combined_tensor = torch.cat((image_tensor, image_tensor), dim=0)
    print(type(combined_tensor))
    print(combined_tensor.shape)

    return imageio.imsave(path, image)

# 将图片合成为大图像，在生成gif中使用
def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
    return img


# 打印损失函数曲线
def loss_plot(hist, path = 'Train_hist.png', model_name = ''):
    x = range(len(hist['D_loss']))

    y1 = hist['D_loss']
    y2 = hist['G_loss']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(path, model_name + '_loss.png')

    plt.savefig(path)

    plt.close()


# 初始化神经网络
def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()