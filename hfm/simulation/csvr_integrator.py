import ase
from hfm import utils
from hfm.simulation.base import Integrator
import jax
import jax.numpy as jnp


class CSVROperatorSplittingIntegrator(Integrator):
    """Using an NVE integrator, CSVR performs velocity rescaling with a thermostat.
    Code partially taken from: https://github.com/IBM/trajcast/blob/c52f107a02176671b21a420b2d4c67e0327b04f7/trajcast/model/forecast_tools/_thermostat.py
    """

    def __init__(
        self,
        nve_integrator,
        integration_timestep,
        temperature,
        time_constant = 100.0 * ase.units.fs,
        zero_drift: bool = False,
        zero_rot: bool = False,
    ):
        self.temperature = jnp.array([temperature]).reshape(1, 1)
        self.time_constant = time_constant
        self.zero_drift = zero_drift
        self.zero_rot = zero_rot
        super().__init__(integration_timestep, nve_integrator.masses, nested_integrator=nve_integrator)

    def _draw_sum_noises_from_gamma_dist(
        self, rng: jax.random.PRNGKey, n_dofs_1: int, num_batches: int
    ):
        # for sampling noise we initialise a gamma distribution
        if n_dofs_1 == 0:
            return jnp.zeros(num_batches)
        elif n_dofs_1 == 1:
            r2 = jax.random.normal(rng, (num_batches,))
            return r2 * r2
        elif n_dofs_1 % 2 == 0:
            gamma = jax.random.gamma(rng, (n_dofs_1) / 2, (num_batches,))
            return 2 * gamma
        else:
            rr = jax.random.normal(rng, (num_batches,))
            gamma = jax.random.gamma(rng, (n_dofs_1 - 1) / 2, (num_batches,))
            return 2 * gamma + rr**2

    def get_rescale_factor(self, rng, e_kin_current, n_dofs, num_batches, n_dim, timestep):
        rng_r, rng_gamma = jax.random.split(rng)

        # target kinetic energy, this is in eV
        e_kin_target = utils.get_kin_energy_from_temperature(
            self.temperature,
            self.masses,
            n_dim=n_dim,
            zero_drift=self.zero_drift,
            zero_rot=self.zero_rot,
        ).reshape(-1)

        c1 = jnp.exp(-timestep / self.time_constant)
        c2 = (1 - c1) * e_kin_target / e_kin_current / n_dofs
        # draw random number from Gaussian distribution with unitary variance (R1)
        r1 = jax.random.normal(rng_r, (num_batches,))

        # draw n_dofs random numbers and sum the squares. As suggested in the original
        #  reference, this can be drawn directly from the gamma distribution.
        # Note: in the flashMD code (https://github.com/lab-cosmo/flashmd/blob/main/src/flashmd/ase/bussi.py)
        # they simply sample n_degrees_of_freedom normal variables, and sum the squares sum(r1*r1).
        r2 = self._draw_sum_noises_from_gamma_dist(
            rng_gamma, n_dofs_1=n_dofs - 1, num_batches=num_batches
        )
        alpha_2 = c1 + c2 * (r1 * r1 + r2) + 2 * r1 * jnp.sqrt(c1 * c2)

        return jnp.sqrt(alpha_2)
    
    def bussi_half_step(self, x, p, rng_csvr, integration_timestep):
        # CSVR step
        n_dofs = utils.get_dof(
            self.masses,
            n_dim=x.shape[2],
            zero_drift=self.zero_drift,
            zero_rot=self.zero_rot,
        )

        e_kin = utils.kinetic_energy(p, self.masses).reshape(-1)
        alpha = self.get_rescale_factor(
            rng_csvr, e_kin, n_dofs, num_batches=x.shape[0], n_dim=x.shape[2], timestep=0.5*integration_timestep
        )

        return p * alpha[..., None, None]

    def integration_step(self, x, p, integration_timestep, aux, filter_aux, rng):
        assert x.ndim == 3, "x and p must be batched"
        assert x.shape == p.shape, (
            f"x and p must have the same shape, got {x.shape} and {p.shape}"
        )

        rng_nve, rng_csvr1, rng_csvr2 = jax.random.split(rng, 3)
        
        p = self.bussi_half_step(x, p, rng_csvr1, integration_timestep)

        # x_, p_, v_, f_ = self.nve_integrator(
        #     x, p, integration_timestep, rng_nve, nested=True
        # )
        x_, p_, v_, f_, aux, filter_aux = self.call_nested_integrator(x, p, integration_timestep, rng_nve, aux, filter_aux)

        # keep the last element of each simulation
        x = x_[:, -1]
        p = p_[:, -1]
        v = v_[:, -1]
        f = f_[:, -1]

        p = self.bussi_half_step(x, p, rng_csvr2, integration_timestep)        
        return x, p, v, f, aux, filter_aux
