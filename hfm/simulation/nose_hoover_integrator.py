import jax
import jax.numpy as jnp
from hfm.simulation.base import Integrator
import hfm.utils as utils
import ase.units as units
import jax
import jax.numpy as jnp
import ase.units as units


class YSWeights:
    """
    Weights for Yoshida-Suzuki integration used in propagating the Nose-Hoover chain thermostats.
    See: https://github.com/atomistic-machine-learning/schnetpack/blob/master/src/schnetpack/md/utils/thermostat_utils.py#L10

    Args:
        device (str): Device used for computation (default='cuda').
    """

    YS_weights = {
        3: jnp.array(
            [1.35120719195966, -1.70241438391932, 1.35120719195966]
        ),
        5: jnp.array(
            [
                0.41449077179438,
                0.41449077179438,
                -0.65796308717750,
                0.41449077179438,
                0.41449077179438,
            ]
        ),
        7: jnp.array(
            [
                0.78451361047756,
                0.23557321335936,
                -1.17767998417887,
                1.31518632068390,
                -1.17767998417887,
                0.23557321335936,
                0.78451361047756,
            ]
        ),
    }

    def get_weights(self, order):
        """
        Get the weights required for an integration scheme of the desired order.

        Args:
            order (int): Desired order of the integration scheme.

        Returns:
            torch.tensor: Tensor of the integration weights
        """
        if order not in self.YS_weights:
            raise ValueError(
                "Order {:d} not supported for YS integration weights".format(order)
            )
        else:
            return self.YS_weights[order]


class NoseHooverChainOperatorSplittingIntegrator(Integrator):
    """Adapted from here: https://github.com/atomistic-machine-learning/schnetpack/blob/82c327fb9b1be5e9fcfd5d104c808e180efa37b6/src/schnetpack/md/simulation_hooks/thermostats.py#L264"""
    def __init__(self,
                 nve_integrator,
                 integration_timestep,
                 temperature,
                 chain_length: int = 3,
                 time_constant: float = 100.0 * units.fs,
                 multi_step: int = 2,
                 integration_order: int = 3,
                 gamma: float = 0.0,
                 zero_drift: bool = False,
                 zero_rot: bool = False,
                 **kwargs):

        super().__init__(integration_timestep, nve_integrator.masses, nested_integrator=nve_integrator, **kwargs)

        self.chain_length = chain_length
        self.frequency = 1.0 / time_constant
        self.multi_step = multi_step
        self.integration_order = integration_order
        self.gamma = gamma
        self.temperature = jnp.array([temperature]).reshape(1, 1) # just for reference
        self.kbT = utils.convert_temperature(self.temperature, "K", "eV")
        self.dof = utils.get_dof(self.masses, zero_drift=zero_drift, zero_rot=zero_rot)

        # Yoshida–Suzuki integration weights
        self.weights = YSWeights().get_weights(self.integration_order)

    # ---------------- Initialization ---------------- #
    def init_aux(self, x0, p0):
        aux = super().init_aux(x0, p0)
        batch_size = x0.shape[0]
        shape = (batch_size, self.chain_length, 1)

        velocities = jnp.zeros(shape)
        forces = jnp.zeros(shape)

        self.thermostat_masses = jnp.zeros((1, self.chain_length, 1))
        self.thermostat_masses = self.thermostat_masses.at[:, 0].set(self.dof * self.kbT / self.frequency**2)
        self.thermostat_masses = self.thermostat_masses.at[:, 1:].set(self.kbT / self.frequency**2)

        aux["velocities"] = velocities
        aux["forces"] = forces

        return aux

    # ---------------- OU half-step ---------------- #
    def _ou_half_step(self, velocities, dt, rng):
        if self.gamma <= 0.0:
            return velocities
        
        alpha = jnp.exp(-self.gamma * dt / 2)
        sigma = jnp.sqrt(self.kbT.reshape(-1, 1, 1) * ((1 - alpha**2) / self.thermostat_masses))
        noise = jax.random.normal(rng, shape=velocities.shape)

        return alpha * velocities + sigma * noise

    # ---------------- NHC half-step ---------------- #
    def _nhc_half_step(self, p, velocities, forces, time_steps):
        kinetic_energy = utils.kinetic_energy(p, self.masses) * 2 # multiply with 2 (like schnetpack reference)
        # Compute forces on innermost thermostat
        forces = forces.at[:, 0].set((kinetic_energy - self.dof * self.kbT) / self.thermostat_masses[:, 0])

        scaling_factor = 1.0

        for _ in range(self.multi_step):
            for time_step in time_steps:
                # Update velocities of outermost bath
                velocities = velocities.at[:, -1].add(0.25 * forces[:, -1] * time_step)

                # Update the velocities moving through the beads of the chain, from outer to inner
                for chain in range(self.chain_length - 2, -1, -1):
                    coeff = jnp.exp(-0.125 * time_step * velocities[:, chain + 1])
                    velocities_i = velocities[:, chain] * coeff**2 + 0.25 * forces[:, chain] * coeff * time_step
                    velocities = velocities.at[:, chain].set(velocities_i)

                # Accumulate momentum scaling
                scaling_factor *= jnp.exp(-0.5 * time_step * velocities[:, 0])

                # Compute forces on innermost thermostat
                forces = forces.at[:, 0].set((scaling_factor * scaling_factor * kinetic_energy - self.dof * self.kbT) / self.thermostat_masses[:, 0])

                # Update the thermostat velocities, from inner to outer
                for chain in range(self.chain_length - 1):
                    coeff = jnp.exp(-0.125 * time_step * velocities[:, chain + 1])
                    velocities_i = velocities[:, chain] * coeff**2 + 0.25 * forces[:, chain] * coeff * time_step
                    velocities = velocities.at[:, chain].set(velocities_i)
                    forces = forces.at[:, chain + 1].set(
                        (self.thermostat_masses[:, chain] * velocities[:, chain]**2 - self.kbT) / self.thermostat_masses[:, chain + 1]
                    )

                # Update velocities of outermost thermostat
                velocities = velocities.at[:, -1].add(0.25 * forces[:, -1] * time_step)

        p = p * scaling_factor.reshape((-1, 1, 1))
        return p, velocities, forces

    # ---------------- Main Integration ---------------- #
    def integration_step(self, x, p, integration_timestep, aux, filter_aux, rng):
        dt = integration_timestep
        velocities, forces = aux["velocities"], aux["forces"]
        time_steps = dt * self.weights / self.multi_step

        # RNG splits
        rng_ou1, rng_nve, rng_ou2 = jax.random.split(rng, 3)

        # First OU half step
        velocities = self._ou_half_step(velocities, dt, rng_ou1)

        # First NHC half step
        p, velocities, forces = self._nhc_half_step(p, velocities, forces, time_steps)

        # NVE propagation
        x_, p_, v_, f_, aux, filter_aux = self.call_nested_integrator(x, p, dt, rng_nve, aux, filter_aux)

        x, p, v, f = x_[:, -1], p_[:, -1], v_[:, -1], f_[:, -1]

        # Second NHC half step
        p, velocities, forces = self._nhc_half_step(p, velocities, forces, time_steps)

        # Second OU half step
        velocities = self._ou_half_step(velocities, dt, rng_ou2)

        aux["velocities"] = velocities 
        aux["forces"] = forces
        return x, p, v, f, aux, filter_aux
