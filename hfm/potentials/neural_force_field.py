import ase
from hfm.backbones.utils import simulation_wrapper
from hfm.simulation.potential import Potential
from functools import partial
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.visualize import view
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from hfm.training.checkpoint import load_params_from_path
import wandb
from tqdm import tqdm
import hfm.utils as utils
import e3x


class NeuralForceField(Potential):
    def __init__(self, masses=None, model=None, params=None, atomic_numbers=None, data_module=None, predict_velocities=False):
        assert model is not None, "Model must be provided"
        assert params is not None, "Model parameters must be provided"
        
        if data_module is not None:
            # load static features from data module
            masses = data_module.masses
            atomic_numbers = data_module.atomic_numbers

        assert masses is not None, "masses or datamodule must be provided"
        assert atomic_numbers is not None, "atomic_numbers or datamodule must be provided"
        utils.verify_shapes(masses=masses, atomic_numbers=atomic_numbers)

        if isinstance(params, str):
            # load params from file
            params = load_params_from_path(params, model_version=model.model_version)

        super().__init__(masses=masses)
        self.model = model
        self.params = params
        self.atomic_numbers = atomic_numbers

        assert atomic_numbers.shape == masses.shape, (
            f"atomic_numbers and masses must have the same shape, got {atomic_numbers.shape} and {masses.shape}"
        )

        apply_fn = lambda t, x, p, a, m: simulation_wrapper(self.model, self.params, t, x, p, a, m, deterministic=True)
        self.apply_f = jax.jit(apply_fn)
        self.predict_velocities = predict_velocities

    def apply_model(self, x, p):
        if self.atomic_numbers is not None:
            atomic_numbers = self.atomic_numbers
        else:
            atomic_numbers = jnp.zeros((x.shape[0], x.shape[1]), dtype=jnp.int32)

        return self.apply_f(
            jnp.zeros((x.shape[0], 1), dtype=jnp.float32),
            x, 
            p, 
            atomic_numbers,
            self.masses)

    def compute_force(self, x, p):
        return self.apply_model(x, p)[1]

    def compute_epot(self, x, p):
        return self.apply_model(x, p)[2]

    def compute_force_and_epot(self, x, p):
        _, f, E = self.apply_model(x, p)
        return f, E
    
    def compute_velocity(self, x, p):
        if self.predict_velocities:
            return self.apply_model(x, p)[0]
        
        return super().compute_velocity(x, p)
