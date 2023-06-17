import torch.nn as nn
from PIL import Image
import torch
import random
import os


# 网络权重初始化
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


# Save the images
def save_images(tensor, filepath):
    # 将图像Tensor转换为PIL Image对象
    image = tensor.detach().squeeze().cpu().numpy()  # 去除维度为1的维度，并转换为NumPy数组
    image = (image * 255).astype('uint8')  # 将像素值范围从[0,1]映射到[0,255]并转换为整数类型
    image = Image.fromarray(image, mode='L')  # 创建灰度图像的PIL Image对象
    image.save(filepath)


# ACGAN generator network architecture
class generator(nn.Module):
    def __init__(self, input_dim=100, output_dim=1, input_size=32, class_num=10):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.class_num = class_num

        # Full connection layer
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim + self.class_num, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(),
        )

        # Deconvolution layer
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )

        # Weights initialization
        initialize_weights(self)

    # Forward Propagation
    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = self.fc(x)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.deconv(x)

        return x


# Auxiliary Classifier Generative Adversarial Network class
class ACGAN:
    def __init__(self):
        # Parameters
        self.save_dir = 'models'
        self.result_dir = 'results'
        self.model_name = 'ACGAN'
        self.input_size = 28
        self.z_dim = 62
        self.class_num = 10
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks initialization
        self.G = generator(input_dim=self.z_dim, output_dim=1, input_size=self.input_size)
        if torch.cuda.is_available():
            self.G.cuda()
        self.load()  # Load model
        self.G.eval()  # Evaluation mode

    def generate(self, num):
        # Directory to save results
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        sample_y_ = torch.zeros(1, self.class_num)
        sample_y_[0][num] = 1.
        sample_z_ = torch.rand((1, self.z_dim))

        noise = torch.rand(1, 28, 28) * 0.01   # 均匀噪声

        if torch.cuda.is_available():
            sample_z_, sample_y_, noise = sample_z_.cuda(), sample_y_.cuda(), noise.cuda()

        samples = (self.G(sample_z_, sample_y_) + 1) / 2 + noise
        samples = samples / samples.max()       # 归一化
        
        return samples

    def load(self):
        save_dir = os.path.join(self.save_dir)

        self.G.load_state_dict(
            torch.load(os.path.join(save_dir, self.model_name + '_G.pkl'), map_location=self.device))

class AiGcMn:
    def __init__(self):
        self.gan = ACGAN()  # Create ACGAN model instance

    def generate(self, num_list):
        flag = False
        for i in num_list:
            if not flag:
                flag = True
                res = self.gan.generate(i)
            else:
                res = torch.cat((res, self.gan.generate(i)), dim=0)
        return res


def main():
    gan = ACGAN()

    random_range = [random.randint(0, 9) for _ in range(10)]
    print(random_range)
    flag = False
    # Here is random input
    for i in random_range:
        if not flag:
            flag = True
            res = gan.generate(i)  # Generate corresponding image to number i
        else:
            res = torch.cat((res, gan.generate(i)), dim=0)  # Connect the generated images

    # Save images
    for i in range(res.shape[0]):
        # Reverse image to white on black
        # inverted_images = 1 - res[i]
        save_images(res[i], os.path.join("./results", f'result_{i}.png'))

    return res


# Test here
if __name__ == '__main__':
    main()
