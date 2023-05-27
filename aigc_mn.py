from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torch
import random
import os


# Load mnist dataset
def dataloader(dataset, input_size, batch_size, split='train'):
    transform = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor(),
                                    transforms.Normalize(mean=0.5, std=0.5)])
    if dataset == 'mnist':
        data_loader = DataLoader(
            datasets.MNIST('data/mnist', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)

    return data_loader


# Network weights initialization
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
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
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


# ACGAN discriminator network architecture
class discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, input_dim=1, output_dim=1, input_size=32, class_num=10):
        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.class_num = class_num

        # Convolution layer
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )

        # Full connection layer 1
        self.fc1 = nn.Sequential(
            nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
        )

        # Discriminator classifier
        self.dc = nn.Sequential(
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid(),
        )

        # Conditioning classifier
        self.cl = nn.Sequential(
            nn.Linear(1024, self.class_num),
        )

        # Weights initialization
        initialize_weights(self)

    # Forward Propagation
    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        x = self.fc1(x)
        d = self.dc(x)
        c = self.cl(x)

        return d, c


# Auxiliary Classifier Generative Adversarial Network class
class ACGAN:
    def __init__(self):
        # Parameters
        self.epoch = 50
        self.sample_num = 100
        self.batch_size = 64
        self.save_dir = 'models'
        self.result_dir = 'results'
        self.dataset = 'mnist'
        self.log_dir = 'logs'
        self.model_name = 'ACGAN'
        self.input_size = 28
        self.z_dim = 62
        self.class_num = 10
        self.sample_num = self.class_num ** 2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load dataset
        self.data_loader = dataloader(self.dataset, self.input_size, self.batch_size)
        data = self.data_loader.__iter__().__next__()[0]

        # Networks initialization
        self.G = generator(input_dim=self.z_dim, output_dim=data.shape[1], input_size=self.input_size)
        self.D = discriminator(input_dim=data.shape[1], output_dim=1, input_size=self.input_size)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=0.0002, betas=(0.2, 0.999))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=0.0002, betas=(0.2, 0.999))
        if torch.cuda.is_available():
            self.G.cuda()
            self.D.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
            self.CE_loss = nn.CrossEntropyLoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()
            self.CE_loss = nn.CrossEntropyLoss()

        # Fixed noise & Condition
        self.sample_z_ = torch.zeros((self.sample_num, self.z_dim))
        for i in range(self.class_num):
            self.sample_z_[i * self.class_num] = torch.rand(1, self.z_dim)
            for j in range(1, self.class_num):
                self.sample_z_[i * self.class_num + j] = self.sample_z_[i * self.class_num]

        temp = torch.zeros((self.class_num, 1))
        for i in range(self.class_num):
            temp[i, 0] = i

        temp_y = torch.zeros((self.sample_num, 1))
        for i in range(self.class_num):
            temp_y[i * self.class_num: (i + 1) * self.class_num] = temp

        self.sample_y_ = torch.zeros((self.sample_num, self.class_num)).scatter_(1, temp_y.type(torch.LongTensor), 1)

        if torch.cuda.is_available():
            self.sample_z_, self.sample_y_ = self.sample_z_.cuda(), self.sample_y_.cuda()

    def generate(self, num):
        self.load()  # Load model
        self.G.eval()  # Evaluation mode

        # Directory to save results
        if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.model_name):
            os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.model_name)

        sample_y_ = torch.zeros(1, self.class_num)
        sample_z_ = torch.rand((1, self.z_dim))

        sample_y_[0][num] = 1.
        if torch.cuda.is_available():
            sample_z_, sample_y_ = sample_z_.cuda(), sample_y_.cuda()
        samples = self.G(sample_z_, sample_y_)

        samples = 1 - (samples + 1) / 2

        save_images(samples, self.result_dir + '/' + self.dataset + '/' +
                    self.model_name + '/' + self.model_name + '_num%03d' % num + '.png')
        return samples

    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        self.G.load_state_dict(
            torch.load(os.path.join(save_dir, self.model_name + '_G.pkl'), map_location=self.device))
        self.D.load_state_dict(
            torch.load(os.path.join(save_dir, self.model_name + '_D.pkl'), map_location=self.device))


def main():
    gan = ACGAN()

    random_range = [random.randint(0, 9) for _ in range(20)]
    flag = False
    # Here is random input
    for i in random_range:
        if not flag:
            flag = True
            res = gan.generate(i)  # Generate corresponding image to number i
        else:
            res = torch.cat((res, gan.generate(i)), dim=0)  # Connect the generated images

    print(res.shape)

    # Save images
    for i in range(res.shape[0]):
        # Reverse image to white on black
        inverted_images = 1 - res[i]
        save_images(inverted_images, os.path.join("./results/mnist/ACGAN", f'result_{i}.png'))

    return res


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


# Test here
if __name__ == '__main__':
    main()

# batch_size = 10  # 批量大小
# input_tensor = torch.randint(0, 10, (batch_size,)).long()
# aigcmn = AiGcMn()
# generated_images = aigcmn.generate(input_tensor)
# print(generated_images.shape)
#
# # 保存生成的图像
# for i in range(batch_size):
#     img = generated_images[i].squeeze().detach().numpy()
#     img = 1 - img  # 取反，黑底白字
#     img = (img * 255).astype(np.uint8)  # 将图像数据转换为 uint8 类型
#     image_path = os.path.join("./results/mnist/ACGAN", f"image_{i}.png")
#     imageio.imwrite(image_path, img)
# print("images saved.")
