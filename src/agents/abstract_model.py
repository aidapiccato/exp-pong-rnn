import abc
import torch

class AbstractModel(torch.nn.Module, metaclass=abc.ABCMeta):
    """Abstract class for RSG models."""

    def __init__(self):
        """Constructor only calls torch.nn.Module.__init__()."""
        super(AbstractModel, self).__init__()

    @abc.abstractmethod
    def loss_terms(self, outputs):
        """Get dictionary of loss terms to be summed for the final loss."""
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, data):
        """Run the model forward on data, getting outputs."""
        raise NotImplementedError

    @abc.abstractmethod
    def scalars(self):
        """Return dictionary of scalars to log."""
        raise NotImplementedError

    @abc.abstractproperty
    def scalar_keys(self):
        """Return tuple of strings, keys of self.scalars() output."""
        raise NotImplementedError
