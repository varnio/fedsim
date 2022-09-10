r"""

Simple Model Architectures
--------------------------

In this file, you can find a number of models that are commonly used in FL community.
These models are used in `Communication-Efficient Learning of Deep Networks from
Decentralized Data`_.

.. _Communication-Efficient Learning of Deep Networks from Decentralized
    Data: https://arxiv.org/abs/1602.05629
"""

# adopted from the following repository:
# https://github.com/c-gabri/Federated-Learning-PyTorch/blob/master/src/models.py
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import Resize

from .utils import get_output_size


class SimpleMLP(nn.Module):
    """A simple two layer Multi-Layer Perceptron.
    This is referred to as 2NN in McMahan's FedAvg paper.

    Args:
        num_classes (int, optional): number of classes. Defaults to 10.
            Assigning None or a negative integer means no classifier.
        num_channels (int, optional): number of channels of input. Defaults to 1.
        in_height (int, optional): input height to resize to. Defaults to 28.
        in_width (int, optional): input width to resize to. Defaults to 28.
        feature_size (int, optional): number of features. Defaults to 200.
    """

    def __init__(
        self,
        num_classes=10,
        num_channels=1,
        in_height=28,
        in_width=28,
        feature_size=200,
    ):
        super(SimpleMLP, self).__init__()
        self.feature_size = feature_size

        self.resize = Resize((in_height, in_width))
        self.fc1 = nn.Linear(num_channels * in_height * in_width, feature_size)
        self.fc2 = nn.Linear(feature_size, feature_size)
        if num_classes is not None and num_classes > 0:
            self.classifier = nn.Linear(feature_size, num_classes)
        else:
            self.classifier = None

    def forward(self, x):
        x = self.get_features(x)
        if self.classifier is not None:
            x = self.classifier(x)
        return x

    def get_features(self, x):
        r"""Gets the extracted features. Goes through all cells except the classifier.

        Args:
            x (Tensor): input tensor with shape
                :math:`(N\times C\times D_1\times D_2\times \dots\times D_n)`
                where ``N`` is batch size and ``C`` is dtermined by ``num_channels``.

        Returns:
            Tensor: output tensor with shape
                :math:`(N\times O)` where ``O`` is determined by ``feature_size``
        """
        x = self.resize(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class SimpleCNN(nn.Module):
    """A simple two layer CNN Perceptron.

    Args:
        num_classes (int, optional): number of classes. Defaults to 10.
            Assigning None or a negative integer means no classifier.
        num_channels (int, optional): number of channels of input. Defaults to 1.
        in_height (int, optional): input height to resize to. Defaults to 28.
        in_width (int, optional): input width to resize to. Defaults to 28.
        feature_size (int, optional): number of features. Defaults to 512.
    """

    def __init__(
        self,
        num_classes=10,
        num_channels=1,
        in_height=28,
        in_width=28,
        num_filters1=32,
        num_filters2=64,
        feature_size=512,
    ):
        super(SimpleCNN, self).__init__()
        self.feature_size = feature_size
        k = 5
        s = 1
        p = 1
        self.resize = Resize((in_height, in_width))

        self.conv1 = nn.Conv2d(
            num_channels, num_filters1, kernel_size=k, stride=s, padding=p
        )

        self.conv2 = nn.Conv2d(
            num_filters1, num_filters2, kernel_size=k, stride=s, padding=p
        )

        # calculate the output size
        # 1st conv
        out_h = get_output_size(in_height, p, k, s)
        out_w = get_output_size(in_height, p, k, s)
        # 1st maxpool
        out_h = get_output_size(out_h, 1, 2, 2)
        out_w = get_output_size(out_w, 1, 2, 2)
        # 2nd conv
        out_h = get_output_size(out_h, p, k, s)
        out_w = get_output_size(out_w, p, k, s)
        # 2nd maxpool
        out_h = get_output_size(out_h, 1, 2, 2)
        out_w = get_output_size(out_w, 1, 2, 2)

        self.fc = nn.Linear(num_filters2 * out_w * out_h, feature_size)
        if num_classes is not None and num_classes > 0:
            self.classifier = nn.Linear(feature_size, num_classes)
        else:
            self.classifier = None

    def forward(self, x):
        x = self.get_features(x)
        if self.classifier is not None:
            x = self.classifier(x)
        return x

    def get_features(self, x):
        r"""Gets the extracted features. Goes through all cells except the classifier.

        Args:
            x (Tensor): input tensor with shape
                :math:`(N\times C\times D_1\times D_2\times \dots\times D_n)`
                where ``N`` is batch size and ``C`` is dtermined by ``num_channels``.

        Returns:
            Tensor: output tensor with shape
                :math:`(N\times O)` where ``O`` is determined by ``feature_size``
        """
        x = self.resize(x)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=1)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=1)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc(x))
        return x


class SimpleCNN2(nn.Module):
    """A simple two layer CNN Perceptron.
    This is similar to CNN model in McMahan's FedAvg paper.

    Args:
        num_classes (int, optional): number of classes. Defaults to 10.
            Assigning None or a negative integer means no classifier.
        num_channels (int, optional): number of channels of input. Defaults to 1.
        in_height (int, optional): input height to resize to. Defaults to 28.
        in_width (int, optional): input width to resize to. Defaults to 28.
        hidden_size (int, optional): number of hidden neurons. Defaults to 384.
        feature_size (int, optional): number of features. Defaults to 192.
    """

    def __init__(
        self,
        num_classes=10,
        num_channels=3,
        in_height=24,
        in_width=24,
        num_filters1=64,
        num_filters2=64,
        hidden_size=384,
        feature_size=192,
    ):
        super(SimpleCNN2, self).__init__()
        self.feature_size = feature_size
        k = 5
        s = 1
        p = "same"
        self.resize = Resize((in_height, in_width))

        self.conv1 = nn.Conv2d(
            num_channels, num_filters1, kernel_size=k, stride=s, padding=p
        )

        self.conv2 = nn.Conv2d(
            num_filters1, num_filters2, kernel_size=k, stride=s, padding=p
        )

        # calculate the output size
        # 1st conv
        out_h = get_output_size(in_height, p, k, s)
        out_w = get_output_size(in_height, p, k, s)
        # 1st maxpool
        out_h = get_output_size(out_h, 0, 3, 2)
        out_w = get_output_size(out_w, 0, 3, 2)
        # 2nd conv
        out_h = get_output_size(out_h, p, k, s)
        out_w = get_output_size(out_w, p, k, s)
        # 2nd maxpool
        out_h = get_output_size(out_h, 1, 2, 2)
        out_w = get_output_size(out_w, 1, 2, 2)

        self.fc1 = nn.Linear(num_filters2 * out_w * out_h, hidden_size)
        self.fc2 = nn.Linear(hidden_size, feature_size)
        if num_classes is not None and num_classes > 0:
            self.classifier = nn.Linear(feature_size, num_classes)
        else:
            self.classifier = None

    def forward(self, x):
        x = self.get_features(x)
        if self.classifier is not None:
            x = self.classifier(x)
        return x

    def get_features(self, x):
        r"""Gets the extracted features. Goes through all cells except the classifier.

        Args:
            x (Tensor): input tensor with shape
                :math:`(N\times C\times D_1\times D_2\times \dots\times D_n)`
                where ``N`` is batch size and ``C`` is dtermined by ``num_channels``.

        Returns:
            Tensor: output tensor with shape
                :math:`(N\times O)` where ``O`` is determined by ``feature_size``
        """
        x = self.resize(x)
        x = F.relu(self.conv1(x))
        x = F.pad(x, (0, 1, 0, 1), value=0)
        x = F.local_response_norm(x, size=4, alpha=0.001 / 9)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=0)
        x = F.relu(self.conv2(x))
        x = F.local_response_norm(x, size=4, alpha=0.001 / 9)
        x = F.pad(x, (0, 1, 0, 1), value=0)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=0)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
