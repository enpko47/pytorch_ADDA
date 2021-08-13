import torch.nn as nn


class Encoder(nn.Module):
    """ LeNet encoder for ADDA """

    def __init__(self):
        """ Initialize LeNet encoder """

        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            # 1st convolutional layer
            # Input  : [ 3 X 28 X 28]
            # Output : [20 X 12 X 12]
            nn.Conv2d(in_channels=3, out_channels=20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),

            # 2nd convolutional layer
            # Input  : [20 X 12 X 12]
            # Output : [50 X  4 X  4]
            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.fc = nn.Linear(in_features=50 * 4 * 4, out_features=500)

    def forward(self, x):
        """ Forward the LeNet encoder """

        x = self.encoder(x)
        x = self.fc(x.view(-1, 50 * 4 * 4))
        return x


class Classifier(nn.Module):
    """ LeNet classifier for ADDA """

    def __init__(self):
        """ Initialize LeNet classifier """

        super(Classifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=500, out_features=10)
        )

    def forward(self, x):
        """ Forward the LeNet classifier """

        x = self.classifier(x)
        return x