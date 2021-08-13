from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import params


def get_mnist(train=True):
    """ Get MNIST data loader """

    # Image pre-processing
    transform = transforms.Compose([transforms.Resize(params.image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,)),
                                    transforms.Lambda(lambda x: x.repeat(3, 1, 1))])

    # MNIST dataset
    mnist_dataset = datasets.MNIST(root=params.data_root,
                                   train=train,
                                   transform=transform,
                                   download=True)
    
    # MNIST data loader
    mnist_loader = DataLoader(dataset=mnist_dataset,
                              batch_size=params.batch_size,
                              shuffle=True)

    return mnist_loader