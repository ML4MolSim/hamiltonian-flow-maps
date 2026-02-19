from collections import namedtuple
import abc
import flax.linen as nn
import jraph

from collections import namedtuple
from jaxtyping import Array
from typing import Optional


GenerativeLayer = namedtuple(
    'GenerativeLayer', field_names=("encoder")
)

FeatureRepresentations = namedtuple(
    'FeatureRepresentations', field_names=('nodes', 'edges')
)

class BaseLayer(nn.Module):
    @abc.abstractmethod
    def __call__(
            self,
            features: FeatureRepresentations,
            graph: Optional[jraph.GraphsTuple],
            **kwargs
    ) -> FeatureRepresentations:
        pass


class BaseTimeEmbedding(nn.Module):
    def __call__(
            self,
            time_latent: Array
    ):
        """

        Args:
            time_latent (): The latent times, (num_nodes)

        Returns:

        """
        pass


class BaseNodeEmbedding(nn.Module):
    def __call__(
            self,
            graph: jraph.GraphsTuple
    ):
        pass


class BaseEdgeEmbedding(nn.Module):
    def __call__(
            self,
            graph: jraph.GraphsTuple
    ):
        pass


class BaseMerger(nn.Module):
    @abc.abstractmethod
    def __call__(
            self,
            features: FeatureRepresentations,
            **kwargs
    ):
        pass


class BaseReadout(nn.Module):
    @abc.abstractmethod
    def __call__(
            self,
            features,
            *args,
            **kwargs
    ):
        pass


class BaseEncoder(nn.Module):
    @abc.abstractmethod
    def __call__(
            self,
            graph
    ) -> Array:

        pass
