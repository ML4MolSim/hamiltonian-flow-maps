import jax.numpy as jnp
from jax import random
from hfm.datasets.in_memory_dataset import InMemoryDataset


class InMemoryDataModule:
    """Provides some functions for splitting in-memory datasets."""
    def __init__(self, 
                 train_data, 
                 val_data, 
                 test_data, 
                 static_features, 
                 global_properties=("Epot",), 
                 skip_last=True, 
                 shuffle_train=True, 
                 name="",
                 dihedral_atom_indices=None,
                 dihedral_angle_names=None,
                 dihedral_angle_symbols=None):

        if dihedral_atom_indices is not None or dihedral_angle_names is not None or dihedral_angle_symbols is not None:
            assert dihedral_atom_indices is not None and dihedral_angle_names is not None and dihedral_angle_symbols is not None, "dihedral_atom_indices, dihedral_angle_names, and dihedral_angle_symbols must all be provided"
            assert len(dihedral_atom_indices) == len(dihedral_angle_names) == len(dihedral_angle_symbols), "dihedral_atom_indices, dihedral_angle_names, and dihedral_angle_symbols must have the same length"
            for dihedrals in dihedral_atom_indices:
                assert len(dihedrals) == 4, "dihedral_atom_indices must have 4 indices"

        self.dihedral_atom_indices = dihedral_atom_indices
        self.dihedral_angle_names = dihedral_angle_names
        self.dihedral_angle_symbols = dihedral_angle_symbols

        self.name = name
        
        self.train_dataset = InMemoryDataset(train_data, static_features, skip_last, shuffle=shuffle_train, global_properties=global_properties)
        self.val_dataset = InMemoryDataset(val_data, static_features, skip_last, shuffle=False, global_properties=global_properties)
        self.test_dataset = InMemoryDataset(test_data, static_features, skip_last, shuffle=False, global_properties=global_properties)

        # store static features / data for easy access
        self.static_features = static_features

    def shutdown(self):
        pass
