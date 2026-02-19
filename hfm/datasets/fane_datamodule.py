from pathlib import Path
import numpy as np
import jax.numpy as jnp
import numpy as np
import ase
import jax.random as random

from hfm.datasets.in_memory_data_module import InMemoryDataModule
from hfm.datasets.numpy_datamodule import NumpyDataModule
from hfm.datasets.utils import parse_unit


class FANEDataModule(InMemoryDataModule):
    def __init__(self, 
                 file_path_train,
                 file_path_val,
                 file_path_test, 
                 force_unit="eV/Ang", 
                 energy_unit="eV", 
                 pos_unit="Ang", 
                 energy_normalization="max", 
                 **kwargs):
        force_unit = parse_unit(force_unit)
        energy_unit = parse_unit(energy_unit)
        pos_unit = parse_unit(pos_unit)

        train_data, static_features = NumpyDataModule.load_numpy(file_path_train, pos_unit, force_unit, energy_unit)
        val_data, _ = NumpyDataModule.load_numpy(file_path_val, pos_unit, force_unit, energy_unit)
        test_data, _ = NumpyDataModule.load_numpy(file_path_test, pos_unit, force_unit, energy_unit)

        # Important for simulations
        self.start_geometry = test_data["x"][0:1]  # (1, n_atoms, 3)
        self.masses = static_features["masses"]
        self.atomic_numbers = static_features["atomic_numbers"]

        E_max = train_data["Epot"].max()
        E_mean = train_data["Epot"].mean()
        if energy_normalization == "max":
            train_data["Epot"] = train_data["Epot"] - E_max # max epot = 0
            val_data["Epot"] = val_data["Epot"] - E_max # max epot = 0
            test_data["Epot"] = test_data["Epot"] - E_max # max epot = 0
        elif energy_normalization == "mean":
            train_data["Epot"] = train_data["Epot"] - E_mean
            val_data["Epot"] = val_data["Epot"] - E_mean
            test_data["Epot"] = test_data["Epot"] - E_mean
        else:
            raise ValueError(f"Unknown normalization strategy {energy_normalization}. Please choose from ['max', 'mean'].")

        #self._data = {k: jnp.concat([train_data[k], val_data[k], test_data[k]], axis=0) for k in train_data.keys()}
        self._data = test_data
        super().__init__(train_data, val_data, test_data, static_features, **kwargs)
