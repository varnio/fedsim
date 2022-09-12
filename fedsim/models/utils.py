"""
Model Utils
-----------

"""

import torch


class ModelReconstructor(torch.nn.Module):
    """reconstructs a model out of a feature_extractor and a classifier.

    Args:
            feature_extractor (Module): feature-extractor module
            classifier (Module): classifier module
            connection_fn (Callable, optional): optional connection function to apply
                on the output of feature-extractor before feeding to the classifier.
                Defaults to None.

    """

    def __init__(self, feature_extractor, classifier, connection_fn=None) -> None:
        super(ModelReconstructor, self).__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.connection_fn = connection_fn

    def forward(self, input):
        features = self.feature_extractor(input)
        if self.connection_fn is not None:
            features = self.connection_fn(features)
        return self.classifier(features)


def get_output_size(in_size, pad, kernel, stride):
    """Calculates the output size after applying a kernel (for one dimension).

    Args:
        in_size (int): input size.
        pad (int): padding size. If set to ``same``, input size is directly returned.
        kernel (int): kernel size.
        stride (int): size of strides.

    Returns:
        int: output size
    """
    if pad == "same":
        return in_size
    return ((in_size + 2 * pad - kernel) // stride) + 1
