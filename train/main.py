import torch
from ACGAN import ACGAN


def main():
    torch.backends.cudnn.benchmark = True
    gan = ACGAN(epoch=25, gpu=1)
    gan.train()


if __name__ == '__main__':
    main()
