from pathlib import Path
import numpy as np
import jax.numpy as jnp
import numpy as np
import ase
import jax.random as random

from hfm.datasets.in_memory_data_module import InMemoryDataModule
from hfm.datasets.utils import parse_unit


class NumpyDataModule(InMemoryDataModule):
    @staticmethod
    def load_numpy(file_path, pos_unit, force_unit, energy_unit, mom_unit=1, load_momenta=False, get_masses_from_ase=True, center_pos=True):
        energy_key = "E"
        force_key = "F"
        position_key = "R"
        z_key = "z"
        
        if "rmd17" in file_path:
            # use rmd17 keys
            energy_key = "energies"
            force_key = "forces"
            position_key = "coords"
            z_key = "nuclear_charges"

        data = np.load(file_path)
        R = jnp.array(data[position_key]) * pos_unit
        F = jnp.array(data[force_key]) * force_unit
        E = jnp.array(data[energy_key]) * energy_unit
        z = data[z_key].astype(int)

        if center_pos:
            # Preprocessing
            R = R - R.mean(axis=1, keepdims=True) # Center positions

        if get_masses_from_ase:
            # Compute static features
            mol = ase.Atoms(z)
            masses = mol.get_masses().reshape(1, -1, 1)
            atomic_numbers = mol.get_atomic_numbers().reshape(1, -1, 1)
        else:
            masses = jnp.array(data["masses"]).reshape(1, -1, 1)
            atomic_numbers = z.reshape(1, -1, 1)

        static_features = {
            "masses": jnp.array(masses),
            "atomic_numbers": jnp.array(atomic_numbers)
        }

        data_dict = {
            "x": R,
            "f": F,
            "Epot": E
        }

        if load_momenta:
            momenta = data["momenta"]
            data_dict["p"] = jnp.array(momenta) * mom_unit

        return data_dict, static_features

    def __init__(self, 
                 file_path,
                 dihedral_atom_indices=None,
                 dihedral_angle_names=None,
                 dihedral_angle_symbols=None,
                 force_unit="kcal/mol/Ang", 
                 energy_unit="kcal/mol", 
                 pos_unit="Ang", 
                 mom_unit=1, 
                 load_momenta=False, 
                 get_masses_from_ase=True,
                 energy_normalization="max", 
                 seed=42, 
                 split_train=0.8, 
                 split_val=0.1, 
                 split_test=0.1, 
                 center_pos=True,
                 **kwargs):

        force_unit = parse_unit(force_unit)
        energy_unit = parse_unit(energy_unit)
        pos_unit = parse_unit(pos_unit)

        data, static_features = self.load_numpy(file_path, pos_unit, force_unit, energy_unit, mom_unit, load_momenta, get_masses_from_ase, center_pos)

        # Important for simulations
        self.start_geometry = data["x"][0:1]  # (1, n_atoms, 3)
        self.masses = static_features["masses"]
        self.atomic_numbers = static_features["atomic_numbers"]

        # data, static_features are dictionaries with str keys and np.ndarray values
        # all data should have the same first dimension (length of dataset)
        # static features have shape (1, feature_dim)
        len_dataset = data[list(data.keys())[0]].shape[0]

        if isinstance(split_train, float):
            assert split_test + split_val + split_train == 1.0, "Float splits must sum to 1.0"
            split_test = int(len_dataset * split_test)
            split_val = int(len_dataset * split_val)
            split_train = len_dataset - split_test - split_val
        else:
            assert split_test + split_val + split_train <= len_dataset, "Integer splits must sum to less than or equal to dataset length"

        assert all(len(v.shape) >= 1 and v.shape[0] == len_dataset for v in data.values()), "All data arrays must have the same length"
        assert all(v.shape[0] == 1 for v in static_features.values()), "Static features must have shape (1, feature_dim)"

        # convert data to jnp arrays
        data = {k: jnp.array(v) for k, v in data.items()}
        static_features = {k: jnp.array(v) for k, v in static_features.items()}

        # Generate indices for splitting the dataset
        rng = random.PRNGKey(seed)
        indices = random.permutation(rng, jnp.arange(len_dataset))

        train_indices = indices[:split_train]
        val_indices = indices[split_train:split_train + split_val]
        test_indices = indices[split_train + split_val:split_train + split_val + split_test]

        train_data = {k: v[train_indices] for k, v in data.items()}
        val_data = {k: v[val_indices] for k, v in data.items()}
        test_data = {k: v[test_indices] for k, v in data.items()}

        E_max = data["Epot"][train_indices].max()
        E_mean = data["Epot"][train_indices].mean()
        if energy_normalization == "max":
            data["Epot"] = data["Epot"] - E_max # max epot = 0
            train_data["Epot"] = train_data["Epot"] - E_max # max epot = 0
            val_data["Epot"] = val_data["Epot"] - E_max # max epot = 0
            test_data["Epot"] = test_data["Epot"] - E_max # max epot = 0
        elif energy_normalization == "mean":
            data["Epot"] = data["Epot"] - E_mean
            train_data["Epot"] = train_data["Epot"] - E_mean
            val_data["Epot"] = val_data["Epot"] - E_mean
            test_data["Epot"] = test_data["Epot"] - E_mean
        else:
            raise ValueError(f"Unknown normalization strategy {energy_normalization}. Please choose from ['max', 'mean'].")

        self._data = data
        super().__init__(train_data, 
                         val_data, 
                         test_data, 
                         static_features,
                         dihedral_atom_indices=dihedral_atom_indices, 
                         dihedral_angle_names=dihedral_angle_names,
                         dihedral_angle_symbols=dihedral_angle_symbols,
                         **kwargs)
