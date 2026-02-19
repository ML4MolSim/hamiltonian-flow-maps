# Mean flow NVE integrator

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from hfm.backbones.utils import simulation_wrapper
from hfm.simulation.base import Integrator
import hfm.utils as utils
from hfm.potentials.neural_force_field import NeuralForceField
from hfm.simulation.nve_integrator import NVEIntegrator
from hfm.training.checkpoint import load_params_from_path


class HFMIntegrator(Integrator):    
    def __init__(self, 
                 integration_timestep=None,
                 model=None,
                 params=None,
                 masses=None,
                 atomic_numbers=None,
                 data_module=None, **kwargs):
        
        assert model is not None, "model must be provided"
        assert params is not None, "params must be provided"

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

        self.model = model
        self.params = params
        self.atomic_numbers = atomic_numbers

        apply_fn = lambda t, x, p, a, m: simulation_wrapper(self.model, self.params, t, x, p, a, m, deterministic=True)
        self.apply_f = jax.jit(apply_fn)
            
        super().__init__(integration_timestep, masses, **kwargs)

    def apply_model(self, x, p, integration_timestep):
        mean_v, mean_f, energy = self.apply_f(
            jnp.ones((x.shape[0], 1)) * integration_timestep,
            x, 
            p, 
            self.atomic_numbers,
            self.masses)
        
        return mean_v, mean_f, energy

    def integration_step(self, x, p, integration_timestep, aux, filter_aux, rng):
        v_pred, f_pred, _ = self.apply_model(x, p, integration_timestep)
        x += integration_timestep * v_pred
        p += integration_timestep * f_pred
        return x, p, v_pred, f_pred, aux, filter_aux
