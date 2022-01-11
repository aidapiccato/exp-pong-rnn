"""MLP model.
This file contains an MLP class. This class does not support batchnorm or other
fancy deep learning tricks, just a basic MLP with activation function.

From pytorch_nets repository in jazlab github
"""
# pylint: disable=import-error

import numpy as np
import torch


class MLP(torch.nn.Module):
    """MLP model."""
    
    def __init__(self, in_features, layer_features, activation=None, bias=True,
                 activate_final=False, apply_to_last_dim=False):
        """Create MLP module.
        Args:
            in_features: Number of features of the input.
            layer_features: Iterable of ints. Output sizes of the layers.
            activation: Activation function. If None, defaults to ReLU.
            bias: Bool. Whether to use bias.
            activate_final: Bool. Whether to apply activation function to the
                final output.
            apply_to_last_dim: Bool. If True, apply the MLP to only the last
                dimension of the input. The return shape will have all other
                dimensions the same as input.
        """
        super(MLP, self).__init__()

        self._in_features = in_features
        self._layer_features = layer_features
        self.bias = bias
        self._apply_to_last_dim = apply_to_last_dim
        
        if activation is None:
            activation = [torch.nn.ReLU() for _ in range(len(layer_features) + 1)]        
        self.activation = activation

        features_list = [in_features] + list(layer_features)
        module_list = []
        for i in range(len(features_list) - 1):
            if i > 0:
                module_list.append(activation[i - 1])
            layer = torch.nn.Linear(
                in_features=features_list[i],
                out_features=features_list[i + 1],
                bias=bias)
            module_list.append(layer)
        
        if activate_final:
            module_list.append(activation[-1])

        self.net = torch.nn.Sequential(*module_list)

    def forward(self, x):
        """Apply MLP to input.
        Args:
            x: Tensor of shape [batch_size, ..., in_features].
        Returns:
            Output of shape [batch_size, ..., self.out_features]. If
                self._apply_to_last_dim, then an arbitrary number of
                intermediate dimensions will be preserved.
        """
        if not self._apply_to_last_dim and len(x.shape) != 2:
            raise ValueError(
                'x.shape is {}, but must have length 2.'.format(self._in_features))

        return self.net(x)

    @property
    def in_features(self):
        return self._in_features

    @property
    def layer_features(self):
        return self._layer_features
        
    @property
    def out_features(self):
        return self._layer_features[-1]


def get_transpose_net(mlp, activate_final=False):
    """Get transpose mlp.
    
    Args:
        mlp: Instance of MLP.
        activate_final: Bool. Whether to activate the final output of the
            returned mlp.
    Returns:
        transpose_mlp: Instance of MLP that is structures so each layer is the
            shape of the transpose of the corresponding layer of mlp.
    """
    layer_features = mlp.layer_features[::-1][1:] + [mlp.in_features]
    transpose_mlp = MLP(
        in_features=mlp.out_features,
        layer_features=layer_features,
        activation=mlp.activation,
        bias=mlp.bias,
        activate_final=activate_final,
    )
    return transpose_mlp