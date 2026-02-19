import ase
import jax.numpy as jnp
import jax
from hfm.simulation.base import Integrator
import hfm.utils as utils


class LangevinOperatorSplittingIntegrator(Integrator):
    """Implements a Langevin integrator via the operator splitting formalism."""
    def __init__(self, 
                 nve_integrator,
                 integration_timestep,
                 temperature,
                 time_constant: float = 100.0 * ase.units.fs,
                 zero_drift_and_rot: bool = False):

        self.temperature = jnp.array([temperature]).reshape(1, 1)
        self.gamma = 1.0 / time_constant
        self.zero_drift_and_rot = zero_drift_and_rot
        super().__init__(integration_timestep, nve_integrator.masses, nested_integrator=nve_integrator)

    def draw_zero_drift_and_rot(self, x, rng):
        p = utils.maxwell_boltzmann_distribution(rng, self.masses, self.temperature, n_dim=x.shape[-1])
        temp0 = utils.get_temperature(p, self.masses, zero_drift=False, zero_rot=False)

        p = utils.zero_rotation(x, p, self.masses, force_temperature=False)
        p = utils.stationary(p, self.masses, force_temperature=False)

        # redistribute kin energy to remaining dof
        return utils.force_temperature_fn(p, self.masses, temp0, zero_drift=True, zero_rot=True)

    def langevin_half_step(self, x, p, dt, rng):
        """
        Apply half a Langevin (thermostat) step to the momentum.
        """
        if self.zero_drift_and_rot:
            p_rand = self.draw_zero_drift_and_rot(x, rng)
        else:
            # kbT = utils.convert_temperature(self.temperature, unit_from="K", unit_to="eV")
            # equals: std_normal * jnp.sqrt(masses * kbT)
            p_rand = utils.maxwell_boltzmann_distribution(rng, self.masses, self.temperature, n_dim=x.shape[-1])

        # alpha = jnp.exp(-self.gamma * dt / 2)
        # sigma = jnp.sqrt(kbT * ((1 - alpha**2) / self.masses.reshape((1, -1, 1))))
        # noise = jax.random.normal(rng, shape=velocity.shape)

        alpha = jnp.exp(-self.gamma * dt / 2)
        sigma = jnp.sqrt(1.0 - jnp.exp(-self.gamma * dt))
        return alpha * p + sigma * p_rand

    def integration_step(self, x, p, integration_timestep, aux, filter_aux, rng):
        rng1, rng2, rng_nve = jax.random.split(rng, 3)

        # Langevin half step
        p = self.langevin_half_step(x, p, integration_timestep, rng1)
        x_, p_, v_, f_, aux, filter_aux = self.call_nested_integrator(x, p, integration_timestep, rng_nve, aux, filter_aux)
        
        # keep the last element of each simulation
        x = x_[:, -1]
        p = p_[:, -1]
        v = v_[:, -1]
        f = f_[:, -1]

        # second Langevin half step
        p = self.langevin_half_step(x, p, integration_timestep, rng2)
        return x, p, v, f, aux, filter_aux
