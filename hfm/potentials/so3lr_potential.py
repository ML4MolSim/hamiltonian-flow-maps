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
from so3lr import So3lr


class SO3LRPotential(Potential):
    def __init__(self, masses, atomic_numbers):
        super().__init__(masses=masses)
        self.atomic_numbers = atomic_numbers.reshape(-1)
        self.so3lr_model = jax.jit(So3lr())

    def apply_model(self, positions):
        assert positions.shape[0] == 1, "only batch size one is supported"
        positions = positions.reshape(-1, 3)

        idx_i, idx_j = e3x.ops.sparse_pairwise_indices(len(positions))

        theory_mask = np.zeros((len(positions), 16)).astype(bool)
        theory_mask[:, 0] = True

        inputs = dict(
            positions=positions,
            atomic_numbers=self.atomic_numbers,
            idx_i=idx_i,
            idx_j=idx_j,
            idx_i_lr=idx_i,
            idx_j_lr=idx_j,
            total_charge=np.array([0.0]),
            num_unpaired_electrons=np.array([0.0]),
            node_mask=np.ones((len(positions), )).astype(bool),
            theory_mask=theory_mask,
        )

        return self.so3lr_model(inputs)

    def compute_force(self, x, p):
        return self.apply_model(x)["forces"]

    def compute_epot(self, x, p):
        return self.apply_model(x)["energy"]

    def compute_force_and_epot(self, x, p):
        output = self.apply_model(x)
        return output["forces"], output["energy"]
