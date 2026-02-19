import e3x
from hfm import utils
from hfm.simulation.base import IntegrationFilter
import jax.numpy as jnp
import jax


class EnergyConservationFilter(IntegrationFilter):
    """Filter to conserve energy without changing angular momentum."""

    def __init__(self, 
                 potential,
                 min_alpha=-.5,
                 max_alpha=.5,
                 min_ekin=1e-5):
        
        print("WARNING: EnergyConservationFilter is deprecated. Use CoupledConservationFilter instead.")

        # the potential is required for explicit energy conservation
        self.potential = potential
        self.min_alpha=min_alpha
        self.max_alpha=max_alpha
        self.min_ekin = min_ekin
        super().__init__()

    def init_aux(self, x0, p0, masses, filter_aux: dict):
        filter_aux['totenergy'] = jnp.zeros(x0.shape[0])
        filter_aux['potenergy'] = self.potential.compute_epot(x0, p0).reshape(x0.shape[0])  # we provide the first potenergy
        return x0, p0, filter_aux

    def in_call(self, x, p, integration_timestep, masses, filter_aux: dict, rng):
        epot = filter_aux['potenergy']
        ekin = utils.kinetic_energy(p, masses).reshape(x.shape[0])

        totenergy = epot + ekin  
        filter_aux['totenergy'] = totenergy
        
        return x, p, filter_aux

    def out_call(self, x, p, integration_timestep, v, f, masses, filter_aux: dict, rng):
        totenergy = filter_aux['totenergy']
        
        new_ekin = utils.kinetic_energy(p, masses).reshape(x.shape[0])
        new_epot = self.potential.compute_epot(x, p).reshape(x.shape[0])
        new_etot = new_epot + new_ekin
        
        p_L0 = utils.remove_global_rotation_3d(x, p, masses)
        
        # solve a quadratic equation for alpha
        a = jnp.sum((p**2) / (2 * masses), axis=(-1, -2))
        b = jnp.sum((p * p_L0) / masses, axis=(-1, -2))
        c = new_etot - totenergy

        alpha1 = (-b + jnp.sqrt(b**2 - 4 * a * c)) / (2 * a + 1e-6)
        alpha2 = (-b - jnp.sqrt(b**2 - 4 * a * c)) / (2 * a + 1e-6)

        alpha = jnp.where(jnp.abs(alpha1) < jnp.abs(alpha2), alpha1, alpha2)        
        alpha = jnp.clip(alpha, self.min_alpha, self.max_alpha)

        # if ekin is too small, only allow changes that increase ekin
        alpha_pos = jnp.clip(alpha, 0, self.max_alpha)
        alpha = jnp.where(new_ekin > self.min_ekin, alpha, alpha_pos)

        filter_aux['potenergy'] = new_epot
        p_new = p + alpha.reshape(-1, 1, 1) * p_L0
        return x, p_new, filter_aux


class RandomRotationFilter(IntegrationFilter):
    def __init__(self):
        super().__init__()

    def init_aux(self, x0, p0, masses, filter_aux: dict):
        filter_aux["R"] = jnp.zeros((x0.shape[0], 3, 3))
        return x0, p0, filter_aux

    """Filter to apply a random rotation to the system and remove it again after integration"""
    def in_call(self, x, p, integration_timestep, masses, filter_aux: dict, rng):
        random_rotation_fn = jax.vmap(e3x.so3.random_rotation)
        R = random_rotation_fn(jax.random.split(rng, x.shape[0]))

        x_mean = jnp.mean(x, axis=1, keepdims=True)
        x_rot = jnp.einsum('bij,bnj->bni', R, x - x_mean) + x_mean
        p_rot = jnp.einsum('bij,bnj->bni', R, p)

        filter_aux['R'] = R
        return x_rot, p_rot, filter_aux

    def out_call(self, x, p, integration_timestep, v, f, masses, filter_aux: dict, rng):
        R = filter_aux['R']
        R_inv = jnp.transpose(R, axes=(0, 2, 1))

        x_mean = jnp.mean(x, axis=1, keepdims=True)
        x_rot = jnp.einsum('bij,bnj->bni', R_inv, x - x_mean) + x_mean
        p_rot = jnp.einsum('bij,bnj->bni', R_inv, p)

        return x_rot, p_rot, filter_aux


class RemoveDriftFilterFlashMD(IntegrationFilter):
    def __init__(self):
        super().__init__()

    def init_aux(self, x0, p0, masses, filter_aux: dict):
        filter_aux["x_com_before"] = jnp.zeros((x0.shape[0], 1, 3), dtype=x0.dtype)
        filter_aux["v_com_before"] = jnp.zeros((x0.shape[0], 1, 3), dtype=x0.dtype)
        return x0, p0, filter_aux

    def _com_position(self, x, masses):
        # x: (B, N, 3), masses: (B, N, 1)
        total_masses = jnp.sum(masses, axis=1, keepdims=True) # (B, 1, 1)
        x_com = jnp.sum(x * masses, axis=1, keepdims=True) / total_masses # (B, 1, 3)
        return x_com

    def _com_velocity(self, p, masses):
        # p: (B, N, 3), masses: (B, N, 1)
        total_masses = jnp.sum(masses, axis=1, keepdims=True) # (B, 1, 1)
        v_com = jnp.sum(p, axis=1, keepdims=True) / total_masses # (B, 1, 3)
        return v_com

    def in_call(self, x, p, integration_timestep, masses, filter_aux: dict, rng):
        filter_aux["x_com_before"] = self._com_position(x, masses)
        filter_aux["v_com_before"] = self._com_velocity(p, masses)
        return x, p, filter_aux

    def out_call(self, x, p, integration_timestep, v, f, masses, filter_aux: dict, rng):
        x_com_now = self._com_position(x, masses)
        x = x - x_com_now + filter_aux["x_com_before"] + filter_aux["v_com_before"] * integration_timestep

        v_com_now = self._com_velocity(p, masses)
        v = p / masses
        v = v - v_com_now + filter_aux["v_com_before"]
        p = v * masses
        return x, p, filter_aux
    

class PreserveAngularMomentumFilter(IntegrationFilter):
    def __init__(self):
        super().__init__()

    def init_aux(self, x0, p0, masses, filter_aux: dict):
        filter_aux["L_in"] = jnp.zeros((x0.shape[0], 3))
        return x0, p0, filter_aux

    def in_call(self, x, p, integration_timestep, masses, filter_aux: dict, rng):
        filter_aux["L_in"] = utils.global_angular_momentum_3d(x, p, masses)
        return x, p, filter_aux

    def out_call(self, x, p, integration_timestep, v, f, masses, filter_aux: dict, rng):
        target_L = filter_aux["L_in"]
        target_L = jnp.where(jnp.abs(target_L) > 1e-5, target_L, 0.0)
        p_new = utils.remove_global_rotation_3d(x, p, masses, target_L=target_L)
        return x, p_new, filter_aux


class CoupledConservationFilter(IntegrationFilter):
    """Filter to conserve energy and angular momentum as coupled optimization problem."""

    def __init__(
        self,
        potential,
        min_alpha=-0.99,
        max_alpha=99.0,
        min_ekin=1e-2,
        min_L_abs=1e-5,
        track_excess_energy=True,
    ):
        # the potential is required for explicit energy conservation
        self.potential = potential
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.min_ekin = min_ekin
        self.min_L_abs = min_L_abs
        self.track_excess_energy = track_excess_energy
        super().__init__()

    def init_aux(self, x0, p0, masses, filter_aux: dict):
        filter_aux["totenergy"] = jnp.zeros(x0.shape[0])
        filter_aux["excess_energy"] = jnp.zeros(x0.shape[0])
        # we provide the first potenergy
        filter_aux["potenergy"] = self.potential.compute_epot(
            x0, p0
        ).reshape(x0.shape[0])
        filter_aux["L_in"] = jnp.zeros((x0.shape[0], 3))

        return x0, p0, filter_aux

    def in_call(self, x, p, integration_timestep, masses, filter_aux: dict, rng):
        epot = filter_aux["potenergy"]  # potential stored from previous step
        ekin = utils.kinetic_energy(p, masses).reshape(x.shape[0])

        totenergy = epot + ekin - filter_aux["excess_energy"]
        filter_aux["totenergy"] = totenergy
        filter_aux["L_in"] = utils.global_angular_momentum_3d(x, p, masses)

        return x, p, filter_aux
    
    def solve_quadratic(self, a, b, c):
        eps = 1e-8
        D = b**2 - 4 * a * c
        denominator = 2 * a + jnp.copysign(eps, a)

        lambda1 = (-b + jnp.sqrt(D)) / denominator
        lambda2 = (-b - jnp.sqrt(D)) / denominator

        is_real = D >= 0
        alpha1 = jnp.where(is_real, lambda1 - 1, self.max_alpha)
        alpha2 = jnp.where(is_real, lambda2 - 1, self.max_alpha)

        alpha = jnp.where(jnp.abs(alpha1) < jnp.abs(alpha2), alpha1, alpha2)        
        alpha = jnp.clip(alpha, self.min_alpha, self.max_alpha)

        return alpha

    def compute_beta(self, I, L):
        def solve_beta(Ii, Li):
            return jnp.linalg.solve(Ii + jnp.eye(3) * 1e-8, Li)

        return jax.vmap(solve_beta)(I, L)

    def out_call(self, x, p, integration_timestep, v, f, masses, filter_aux: dict, rng):
        totenergy = filter_aux["totenergy"]
        target_L = filter_aux["L_in"]
        target_L = jnp.where(jnp.abs(target_L) > self.min_L_abs, target_L, 0.0)

        new_epot = self.potential.compute_epot(x, p).reshape(x.shape[0])
        Ekin_tgt = totenergy - new_epot
        Ekin_tgt = jnp.clip(Ekin_tgt, min=self.min_ekin, max=None)

        com = jnp.sum(x * masses, axis=1) / jnp.sum(masses, axis=1)
        r = x - com[:, None, :]  # (B, N, 3)

        # Calculate the angular momentum and the moments of inertia.
        L = jnp.sum(jnp.cross(r, p), axis=1)  # (B, 3)
        I = utils.inertia_tensor(r, masses)  # (B, 3, 3)
        
        beta0 = self.compute_beta(I, L)  # (B, 3)
        beta1 = self.compute_beta(I, target_L)  # (B, 3)

        p0 = p - masses * jnp.cross(beta0[:, None, :], r)
        p1 = masses * jnp.cross(beta1[:, None, :], r)

        A = (jnp.sum(p1**2 / masses, axis=(-1))).sum(axis=1) - Ekin_tgt * 2
        B = (2 * jnp.sum(p0 * p1 / masses, axis=(-1))).sum(axis=1)
        C = (jnp.sum(p0**2 / masses, axis=(-1))).sum(axis=1)

        alpha = self.solve_quadratic(A, B, C)

        beta = beta0 - (1 + alpha)[:, None] * beta1
        p_new = 1 / (1 + alpha)[:, None, None] * (p - masses * jnp.cross(beta[:, None, :], r))
        filter_aux["potenergy"] = new_epot

        if self.track_excess_energy:
            ekin_new = utils.kinetic_energy(p_new, masses).reshape(x.shape[0])
            filter_aux["excess_energy"] = ekin_new - (totenergy - new_epot)
           
        return x, p_new, filter_aux
