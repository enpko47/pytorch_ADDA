from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import params


def get_usps(train=True):
    """ Get USPS data loader """

    # Image pre-processing
    transform = transforms.Compose([transforms.Resize(params.image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.2469,), (0.2989,)),
                                    transforms.Lambda(lambda x: x.repeat(3, 1, 1))])

    # USPS dataset
    usps_dataset = datasets.USPS(root=params.data_root,
                                   train=train,
                                   transform=transform,
                                   download=True)
    
    # USPS data loader
    usps_loader = DataLoader(dataset=usps_dataset,
                              batch_size=params.batch_size,
                              shuffle=True)

    return usps_loader