from hfm.simulation.potential import Potential
from functools import partial
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.visualize import view
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import wandb
from tqdm import tqdm
import hfm.utils as utils
import e3x


class MuellerBrownPotential(Potential):
    def __init__(self):
        super().__init__(masses=jnp.array([0.5]).reshape(1, -1, 1))

    @staticmethod
    def plot_mueller_brown(ax=None):
        if ax is None:
            ax = plt.gca()

        bins = 150
        (x_min, x_max), (y_min, y_max) = jnp.array([[-1.8, 1.1], [-0.5, 2.0]])
        # make a nice contour plot of the potential
        x, y = jnp.linspace(x_min, x_max, bins), jnp.linspace(y_min, y_max, bins)
        x, y = jnp.meshgrid(x, y, indexing="ij")
        z = MuellerBrownPotential.mueller_brown_potential(jnp.stack([x, y], -1).reshape(-1, 2)).reshape([bins, bins])
        ax.contour(x, y, z, levels=[-120, -90, -50, -20, 0, 20, 35, 70, 150, 250, 500, 1000], colors="black")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')

    @staticmethod
    def mueller_brown_potential(xs: jnp.ndarray, beta: float = 1.0) -> jnp.ndarray:
        """Compute the Müller-Brown potential energy."""
        if xs.ndim == 1:
            xs = xs.reshape(1, -1)

        x, y = xs[..., 0], xs[..., 1]
        e1 = -200 * jnp.exp(-((x - 1) ** 2) - 10 * y**2)
        e2 = -100 * jnp.exp(-(x**2) - 10 * (y - 0.5) ** 2)
        e3 = -170 * jnp.exp(-6.5 * (0.5 + x) ** 2 + 11 * (x + 0.5) * (y - 1.5) - 6.5 * (y - 1.5) ** 2)
        e4 = 15.0 * jnp.exp(0.7 * (1 + x) ** 2 + 0.6 * (x + 1) * (y - 1) + 0.7 * (y - 1) ** 2)
        return beta * (e1 + e2 + e3 + e4)

    @staticmethod
    def mueller_brown_force(xs: jnp.ndarray) -> jnp.ndarray:
        """Compute the force as the negative gradient of the potential."""
        return jax.grad(lambda _x: -MuellerBrownPotential.mueller_brown_potential(_x).sum())(xs)
    
    def compute_force(self, x, p=None):
        return MuellerBrownPotential.mueller_brown_force(x)

    def compute_epot(self, x, p=None):
        return MuellerBrownPotential.mueller_brown_potential(x)

    def compute_force_and_epot(self, x, p=None):
        return self.compute_force(x), self.compute_epot(x)
