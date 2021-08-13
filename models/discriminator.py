import torch.nn as nn


class Discriminator(nn.Module):
    """ Discriminator for ADDA """

    def __init__(self):
        """ Initialize Discriminator  """

        super(Discriminator, self).__init__()
        
        self.layer = nn.Sequential(
            nn.Linear(in_features=500, out_features=500),
            nn.ReLU(),
            nn.Linear(in_features=500, out_features=500),
            nn.ReLU(),
            nn.Linear(in_features=500, out_features=1)
        )

    def forward(self, x):
        """ Forward the discriminator """

        x = self.layer(x)
        return x