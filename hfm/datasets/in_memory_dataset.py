import jax.numpy as jnp
from jax import random

from hfm.backbones.utils import build_graph


class InMemoryDataset:
    def __init__(self, data, static_features, skip_last, global_properties, shuffle=False):
        self.data = data
        self.shuffle = shuffle
        self.static_features = static_features
        self.skip_last = skip_last
        self.first_key = list(self.data.keys())[0]

        # ensure global_properties is a tuple if given as a list
        if isinstance(global_properties, list):
            self.global_properties = tuple(global_properties)
        elif isinstance(global_properties, tuple):
            self.global_properties = global_properties
        else:
            raise ValueError("global_properties must be a tuple or list of strings")

    def _yield_batch(self, data_epoch, batch_size, start_index):
        batch_data = {k: data_epoch[k][start_index:start_index + batch_size] for k in data_epoch}
        
        if len(batch_data[self.first_key]) == 0\
            or (len(batch_data[self.first_key]) < batch_size and self.skip_last):
            return None
        
        # append static features to each sample in the batch
        for k in self.static_features:
            batch_data[k] = self.static_features[k]

        return build_graph(batch_data, num_graph=batch_data["x"].shape[0], num_node=batch_data["x"].shape[1], global_properties=self.global_properties)

    def next_epoch(self, rng, batch_size):
        if self.shuffle:
            # Shuffle the data randomly and return an iterator
            indices = random.permutation(rng, jnp.arange(len(self)))
            data_epoch = {k: self.data[k][indices] for k in self.data}
        else:
            data_epoch = self.data
        
        n = 0
        while n < len(self):
            sample = self._yield_batch(data_epoch, batch_size, n)

            if sample is None:
                break
            
            yield sample
            n += batch_size

    def get_example_batch(self, rng, batch_size):
        return self._yield_batch(self.data, batch_size, 0)

    def __len__(self):
        return len(self.data[self.first_key])

    def __getattribute__(self, name):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            # If possible, return values from static_features
            static_features = super().__getattribute__("static_features")
            if name in static_features:
                return static_features[name]
            raise
