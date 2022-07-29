r"""

McMahan Models
--------------

Models used in `Communication-Efficient Learning of Deep Networks from
Decentralized Data`_.

.. _Communication-Efficient Learning of Deep Networks from Decentralized
    Data: https://arxiv.org/abs/1602.05629
"""

# adopted from the following repository:
# https://github.com/c-gabri/Federated-Learning-PyTorch/blob/master/src/models.py
from torch import nn
from torchvision.transforms import Resize


class mlp_mnist(nn.Module):
    def __init__(self, num_classes=10, num_channels=1):
        super(mlp_mnist, self).__init__()

        self.resize = Resize((28, 28))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_channels * 28 * 28, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, num_classes),
        )

    def forward(self, x):
        x = self.resize(x)
        x = self.classifier(x)
        return x


# class cnn_mnist(nn.Module):

#     def __init__(self, num_classes=10, num_channels=1):
#         super(cnn_mnist, self).__init__()

#         self.resize = Resize((28, 28))

#         self.feature_extractor = nn.Sequential(
#             nn.Conv2d(num_channels, 32, kernel_size=5, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, stride=2, padding=1),
#             nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, stride=2, padding=1),
#         )

#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(64 * 7 * 7, 512),
#             nn.ReLU(),
#             nn.Linear(512, num_classes),
#         )

#     def forward(self, x):
#         x = self.resize(x)
#         x = self.feature_extractor(x)
#         x = self.classifier(x)
#         return x


class cnn_mnist(nn.Module):
    def __init__(self, num_classes=10, num_channels=1):
        super(cnn_mnist, self).__init__()

        self.resize = Resize((28, 28))

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.resize(x)
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x


# class cnn_cifar10(nn.Module):

#     def __init__(self, num_classes=10, num_channels=3, input_size=(24, 24)):
#         super(cnn_cifar10, self).__init__()

#         self.resize = Resize(input_size)

#         self.feature_extractor = nn.Sequential(
#             nn.Conv2d(num_channels,
#                       64,
#                       kernel_size=5,
#                       stride=1,
#                       padding='same'),
#             nn.ReLU(),
#             nn.ZeroPad2d((0, 1, 0, 1)),
#             nn.MaxPool2d(3, stride=2, padding=0),
#             nn.LocalResponseNorm(4, alpha=0.001 / 9),
#             nn.Conv2d(64, 64, kernel_size=5, stride=1, padding='same'),
#             nn.ReLU(),
#             nn.LocalResponseNorm(4, alpha=0.001 / 9),
#             nn.ZeroPad2d((0, 1, 0, 1)),
#             nn.MaxPool2d(3, stride=2, padding=0),
#         )

#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(64 * 6 * 6, 384),
#             nn.ReLU(),
#             nn.Linear(384, 192),
#             nn.ReLU(),
#             nn.Linear(192, num_classes),
#         )

#     def forward(self, x):
#         x = self.resize(x)
#         x = self.feature_extractor(x)
#         x = self.classifier(x)
#         return x


class cnn_cifar10(nn.Module):
    def __init__(self, num_classes=10, num_channels=3, input_size=(24, 24)):
        super(cnn_cifar10, self).__init__()

        self.resize = Resize(input_size)

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                num_channels,
                64,
                kernel_size=5,
                stride=1,
                padding="same",
            ),
            nn.ReLU(),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.MaxPool2d(3, stride=2, padding=0),
            nn.LocalResponseNorm(4, alpha=0.001 / 9),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding="same"),
            nn.ReLU(),
            nn.LocalResponseNorm(4, alpha=0.001 / 9),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.MaxPool2d(3, stride=2, padding=0),
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 384),
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(192, num_classes)

    def forward(self, x):
        x = self.resize(x)
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x


class cnn_cifar100(cnn_cifar10):
    def __init__(self, num_classes=100, num_channels=3, input_size=(24, 24)):
        super(cnn_cifar100, self).__init__(
            num_classes=num_classes,
            num_channels=num_channels,
            input_size=input_size,
        )
