from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def dataloader(dataset, input_size, batch_size, split='train'):
    transform = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor(), transforms.Normalize(mean=(0.5), std=(0.5))])
    data_loader = DataLoader(
        datasets.MNIST('data/mnist', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True)
    return data_loader