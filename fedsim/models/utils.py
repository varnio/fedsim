"""
Model Utils
-----------

"""

import torch


class ModelReconstructor(torch.nn.Module):
    """reconstructs a model out of a features_extractor of a base model and a
    classifier.

    Args:
            base_model (Module): module that implements a get_features method.
            classifier (Module): classifier module
            connection_fn (Callable, optional): optional connection function to apply
                on the output of feature-extractor before feeding to the classifier.
                Defaults to None.

    """

    def __init__(self, base_model, classifier, connection_fn=None) -> None:
        super(ModelReconstructor, self).__init__()
        self.base_model = base_model
        self.classifier = classifier
        self.connection_fn = connection_fn

    # TODO define parameters() method to return all parameters except the old classifier

    def forward(self, input):
        features = self.base_model.get_features(input)
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
