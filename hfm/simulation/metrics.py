from abc import ABC
from pathlib import Path
from ase import Atoms
from ase.neighborlist import NeighborList, natural_cutoffs
from ase.visualize.plot import plot_atoms
from matplotlib.colors import LogNorm
from pyparsing import abstractmethod
from IPython.display import Video
from hfm import utils
from hfm.simulation.utils import align_positions, compute_dihedral_batch, get_total_energy, plot_spectrum
from hfm.utils import global_angular_momentum_3d

import tempfile
import wandb
import ase
import imageio.v2 as imageio
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


class SimulationMetrics(ABC):
    def __init__(self, data_module, integration_timestep, subsample=1, gt_data_integration_timestep=0.5 * ase.units.fs, keep_gt_as_is=False):
        self.integration_timestep = integration_timestep
        self.subsample = subsample  # for large trajectories, save some computation
        self.masses = data_module.train_dataset.masses
        self.atomic_numbers = data_module.train_dataset.atomic_numbers

        if self.integration_timestep < gt_data_integration_timestep:
            keep_gt_as_is = True
            print("Warning: integration_timestep is smaller than gt_data_integration_timestep. Results that depend on time may be inaccurate.")

        if keep_gt_as_is:
            self.reference_data = {k: v for k, v in data_module._data.items()}
        else:
            gt_subsampling = int(self.integration_timestep / gt_data_integration_timestep) * self.subsample
            self.reference_data = {k: v[::gt_subsampling] for k, v in data_module._data.items()}

    @abstractmethod
    def _compute_metrics(self, ts, xs, ps, vs, fs, aux, log):
        pass

    def __call__(self, traj_data, log=True):
        xs, ps, vs, fs = traj_data["xs"], traj_data["ps"], traj_data["vs"], traj_data["fs"]
        xs, ps, vs, fs = xs[::self.subsample], ps[::self.subsample], vs[::self.subsample], fs[::self.subsample]
        ts = jnp.arange(len(xs)) * self.integration_timestep * self.subsample / (ase.units.fs * 1_000)  # time in ps
        new_data = self._compute_metrics(ts, xs, ps, vs, fs, traj_data, log)

        if new_data is not None:
            for k in new_data:
                assert k not in traj_data, f"Key {k} already exists in traj_data. Please use a different key name."
                traj_data[k] = new_data[k]

        return traj_data
    

class LogFigure:
    def __init__(self, name, log, **figure_kwargs):
        self.name = name
        self.log = log
        self.figure_kwargs = figure_kwargs

    def __enter__(self):
        self.f, self.ax = plt.subplots(**self.figure_kwargs)
        return self
    
    def __exit__(self, exc_type, exc_value, exc_tb):
        if wandb.run and self.log:
            wandb.log({self.name: wandb.Image(self.f)})
        plt.close(self.f)


class PltVideoWriter:
    def __init__(self, filename, fps):
        self.filename = filename
        self.writer = imageio.get_writer(filename, fps=fps, format="ffmpeg")
        
    def add_frame(self, figure):
        figure.canvas.draw()
        self.writer.append_data((np.array(figure.canvas.renderer.buffer_rgba())))
        plt.close(figure)

    def close(self):
        self.writer.close()

    def display(self):
        return Video(self.filename)
    
    def log(self, name, log):
        if wandb.run and log:
            wandb.log({name: wandb.Video(self.filename, format="mp4")})


class PltVideo:
    def __init__(self, filename, fps=10):
        self.video_writer = PltVideoWriter(filename, fps)
    def __enter__(self):
        return self.video_writer
    def __exit__(self, exc_type, exc_value, exc_tb):
        self.video_writer.close()


class PlotAngularAndMeanMomentum(SimulationMetrics):
    def _compute_metrics(self, ts, xs, ps, vs, fs, aux, log):
        mean_momenta = jnp.mean(ps, axis=1)
        with LogFigure("Mean Momentum", log):
            plt.plot(ts, mean_momenta, alpha=0.3)
            plt.xlabel("Time (ps)")
            plt.ylabel("Mean Momentum")
            plt.yscale('log')

        mean_momenta_magnitude = jnp.linalg.norm(mean_momenta, axis=-1)
        with LogFigure("Mean Momentum Magnitude", log):
            plt.plot(ts, mean_momenta_magnitude)
            plt.xlabel("Time (ps)")
            plt.ylabel("Mean Momentum Magnitude")
            plt.yscale('log')

        total_angular_momentum = global_angular_momentum_3d(xs, ps, self.masses)
        with LogFigure("Total Angular Momentum", log):
            plt.plot(ts, total_angular_momentum, alpha=0.3)
            plt.xlabel("Time (ps)")
            plt.ylabel("Total Angular Momentum")


class PlotTempAndEnergy(SimulationMetrics):
    def __init__(self, potential, T_equilibrium, running_avg_window=1, zero_drift=False, zero_rot=False, **kwargs):
        super().__init__(**kwargs)
        self.T_equilibrium = T_equilibrium
        self.running_avg_window = running_avg_window
        self.potential = potential
        self.zero_drift = zero_drift
        self.zero_rot = zero_rot

    def _compute_metrics(self, ts, xs, ps, vs, fs, aux, log):
        # Plot temperature and energy
        etot, ekin, epot = get_total_energy(xs, ps, self.potential.compute_epot, self.masses, self.running_avg_window, return_ekin_epot=True)
        with LogFigure("Energy", log):
            plt.plot(ts, etot, label=f"Total Energy (running avg)", alpha=0.7)
            plt.plot(ts, ekin, label=f"Kin. Energy (running avg)", alpha=0.7)
            plt.plot(ts, epot, label=f"Pot. Energy (running avg)", alpha=0.7)
            plt.xlabel("Time (ps)")
            plt.legend()

        temp = utils.get_temperature(ps, self.masses, zero_drift=self.zero_drift, zero_rot=self.zero_rot)
        with LogFigure("Temperature", log):
            plt.plot(ts, temp, label=f"Temperature")
            plt.axhline(temp.mean(), color='red', linestyle='--', label='Mean Temperature')
            plt.axhline(self.T_equilibrium, color='green', linestyle='--', label='Target Temperature')
            plt.xlabel("Time (ps)")
            plt.legend()

        if wandb.run and log:
            wandb.log({"mean temperature (K)": temp.mean()})

        result = {"etot": etot, "ekin": ekin, "epot": epot, "temp": temp}

        from ase.data import chemical_symbols
        atomic_symbols = [chemical_symbols[z] for z in self.atomic_numbers.reshape(-1)]
        element_indices = {
            element: [i for i, x in enumerate(atomic_symbols) if x == element] 
            for element in set(atomic_symbols)
        }
        
        for element in element_indices.keys():
            indices = element_indices[element]
            temp = utils.get_temperature(ps[:, indices, :], self.masses[:, indices, :], zero_drift=self.zero_drift, zero_rot=self.zero_rot)
            with LogFigure(f"Temperature for {element}", log):
                plt.plot(ts, temp, label=f"Temperature")
                plt.axhline(temp.mean(), color='red', linestyle='--', label='Mean Temperature')
                plt.axhline(self.T_equilibrium, color='green', linestyle='--', label='Target Temperature')
                plt.xlabel("Time (ps)")
                plt.legend()
            if wandb.run and log:
                wandb.log({f"mean temperature (K) for {element}": temp.mean()})
            

        return result


class PlotEPotHistogram(SimulationMetrics):
    def _compute_metrics(self, ts, xs, ps, vs, fs, aux, log):
        assert "epot" in aux, "epot must be computed before plotting epot histogram"
        epot = aux["epot"]
        gt_epot = self.reference_data["Epot"]

        with LogFigure("Potential Energy Histogram", log):
            plt.hist(gt_epot, bins=100, density=True, alpha=0.5, label="GT Potential Energy", color='blue')
            plt.hist(epot, bins=100, density=True, alpha=0.5, label="Simulation Potential Energy", color='orange')
            plt.xlabel('Potential Energy (eV)')
            plt.ylabel('Density')
            plt.title('Histogram of Potential Energy')
            plt.legend()


class PlotSpectrum(SimulationMetrics):
    def __init__(self, sigma=10, window_length=50, use_momenta=False, burn_in_x=0, use_sgdml=False, use_gaussian_filter=False, align_positions=False, **kwargs):
        super().__init__(**kwargs)
        self.burn_in_x = burn_in_x
        self.sigma = sigma
        self.window_length = window_length
        self.use_momenta = use_momenta
        self.algorithm = "sgdml_jax" if use_sgdml else "fft_max"
        self.filter = "gaussian" if use_gaussian_filter else "savitzky"
        self.align_positions = align_positions

    def _compute_metrics(self, ts, xs, ps, vs, fs, aux, log):
        xs = xs[self.burn_in_x // self.subsample:]
        ps = ps[self.burn_in_x // self.subsample:]
        vel = ps / self.masses if self.use_momenta else None

        if self.align_positions:
            xs = align_positions(xs)

        timestep_fs = self.integration_timestep / ase.units.fs
        gt_trajectory = self.reference_data["x"]

        text = "from momenta" if self.use_momenta else "from positions"
        with LogFigure(f"Power Spectra - {text}", log):
            plt.title(f"Power Spectra with Number of Samples: {xs.shape[0]}")
            xlim1 = plot_spectrum(gt_trajectory, dt=self.integration_timestep, sigma=self.sigma, label="GT Spectrum", algorithm=self.algorithm, filter=self.filter, window_length=self.window_length, log=log)
            xlim2 = plot_spectrum(xs, velocities=vel, dt=self.integration_timestep, sigma=self.sigma, label=f"Simulation (time step {timestep_fs:.2f}fs)", color="red", algorithm=self.algorithm, filter=self.filter, window_length=self.window_length, log=log)
            
            plt.xlim(0, max(xlim1, xlim2) * 1.05)
            plt.xlabel("Frequency")
            plt.ylabel("Power Spectrum")
            plt.legend()


class PlotDihedralHistogram(SimulationMetrics):
    def __init__(self, data_module, angle_index, keep_gt_as_is=True, log_axis=False, **kwargs):
        super().__init__(data_module=data_module, keep_gt_as_is=keep_gt_as_is, **kwargs)
        self.skip = False
        if data_module.dihedral_atom_indices is None:
            self.skip = True
        else:
            assert angle_index < len(data_module.dihedral_atom_indices), "angle_index must be less than the number of dihedral angles"
            self.dihedral_atom_indices = data_module.dihedral_atom_indices[angle_index]
            self.angle_name = data_module.dihedral_angle_names[angle_index]
            self.angle_symbol = data_module.dihedral_angle_symbols[angle_index]
            self.log_axis = log_axis

    def _compute_metrics(self, ts, xs, ps, vs, fs, aux, log):
        if self.skip:
            print("Warning: No information about dihedral angles provided in data module. Skipping Dihedral Histogram plot.")
            return

        timestep_fs = self.integration_timestep / ase.units.fs
        gt_trajectory = self.reference_data["x"]

        idx = self.dihedral_atom_indices
        angles_gt = compute_dihedral_batch(gt_trajectory[:, idx[0]], gt_trajectory[:, idx[1]], gt_trajectory[:, idx[2]], gt_trajectory[:, idx[3]])
        angles_sim = compute_dihedral_batch(xs[:, idx[0]], xs[:, idx[1]], xs[:, idx[2]], xs[:, idx[3]])

        with LogFigure(f"Dihedral Angle {self.angle_name}", log):
            plt.hist(angles_gt, bins=100, density=True, alpha=0.5, label="GT Dihedral Angles")
            plt.hist(angles_sim, bins=100, density=True, alpha=0.5, label=f"Simulation (time step {timestep_fs:.2f}fs)")
            plt.xlabel(f'Dihedral Angle ${self.angle_symbol}$ (degrees)')
            plt.ylabel('Frequency')
            plt.title(f'Histogram of Dihedral Angles ${self.angle_symbol}$')
            if self.log_axis:
                plt.yscale('log')

            plt.legend()


class PlotRamachandran(SimulationMetrics):
    def __init__(self, data_module, angle_indices=(0, 1), keep_gt_as_is=True, **kwargs):
        super().__init__(data_module=data_module, keep_gt_as_is=keep_gt_as_is, **kwargs)
        self.skip = False
        assert len(angle_indices) == 2, "angle_indices must have 2 indices"
        if data_module.dihedral_atom_indices is None:
            self.skip = True
        else:
            self.angle_indices = angle_indices
            self.dihedral_atom_indices = data_module.dihedral_atom_indices
            self.dihedral_angle_names = data_module.dihedral_angle_names
            self.dihedral_angle_symbols = data_module.dihedral_angle_symbols

    def _compute_metrics(self, ts, xs, ps, vs, fs, aux, log):
        if self.skip:
            print("Warning: No information about dihedral angles provided in data module. Skipping Ramachandran plot.")
            return

        gt_trajectory = self.reference_data["x"]

        iphi = self.dihedral_atom_indices[self.angle_indices[0]]
        ipsi = self.dihedral_atom_indices[self.angle_indices[1]]

        phi_gt = compute_dihedral_batch(gt_trajectory[:, iphi[0]], gt_trajectory[:, iphi[1]], gt_trajectory[:, iphi[2]], gt_trajectory[:, iphi[3]])
        psi_gt = compute_dihedral_batch(gt_trajectory[:, ipsi[0]], gt_trajectory[:, ipsi[1]], gt_trajectory[:, ipsi[2]], gt_trajectory[:, ipsi[3]])
        phi_sim = compute_dihedral_batch(xs[:, iphi[0]], xs[:, iphi[1]], xs[:, iphi[2]], xs[:, iphi[3]])
        psi_sim = compute_dihedral_batch(xs[:, ipsi[0]], xs[:, ipsi[1]], xs[:, ipsi[2]], xs[:, ipsi[3]])

        plot_range = [-jnp.pi, jnp.pi]
        with LogFigure("Ramachandran Plot", log, nrows=1, ncols=2, figsize=(12,4)) as lfig:
            for idx, (title, phi, psi) in enumerate([("Ground Truth", phi_gt, psi_gt), ("Prediction", phi_sim, psi_sim)]):
                lfig.ax[idx].hist2d(
                    np.deg2rad(phi),
                    np.deg2rad(psi),
                    bins=80,
                    norm=LogNorm(vmin=None, vmax=None),
                    range=[plot_range, plot_range],
                    cmap='turbo',
                    density=True,
                )
                lfig.ax[idx].set_xticks(
                    [-jnp.pi, -jnp.pi / 2, 0, jnp.pi / 2, jnp.pi],
                    [r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"],
                )
                lfig.ax[idx].set_yticks(
                    [-jnp.pi, -jnp.pi / 2, 0, jnp.pi / 2, jnp.pi],
                    [r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"],
                )
                lfig.ax[idx].set_xlim(plot_range)
                lfig.ax[idx].set_ylim(plot_range)
                lfig.ax[idx].set_box_aspect(1)
                lfig.ax[idx].set_title(title)
                lfig.ax[idx].set_xlabel(f"${self.dihedral_angle_symbols[self.angle_indices[0]]}$")
                lfig.ax[idx].set_ylabel(f"${self.dihedral_angle_symbols[self.angle_indices[1]]}$")

            plt.tight_layout()


class ForcesAreNotEnoughMetrics(SimulationMetrics):
    def __init__(self, stability_threshold=0.5, n_bins = 500, xlim = 10, **kwargs):
        super().__init__(**kwargs)
        self.stability_threshold = stability_threshold
        self.n_bins = n_bins
        self.xlim = xlim
    
    def _compute_metrics(self, ts, xs, ps, vs, fs, aux, log):
        def cdist(x, y):
            diffs = x[:, :, None, :] - y[:, None, :, :]  # (N, M, D)
            return jnp.sqrt(jnp.sum(diffs ** 2, axis=-1))  # Euclidean

        def get_hr(traj, bins):
            pdist = cdist(traj, traj).flatten()
            hist, _ = jnp.histogram(pdist[:].flatten(), bins, density=True)
            return hist

        def distance_pbc(x0, x1, lattices):
            delta = jnp.abs(x0 - x1)
            lattices = lattices.reshape(-1,1,3)
            delta = jnp.where(delta > 0.5 * lattices, delta - lattices, delta)
            return jnp.sqrt((delta ** 2).sum(axis=-1))

        def mae(x, y, factor):
            return jnp.abs(x-y).mean() * factor

        bins = np.linspace(1e-6, self.xlim, self.n_bins + 1) # for computing h(r)
        gt_traj = self.reference_data["x"]
        hist_gt = get_hr(gt_traj, bins)

        atoms = ase.Atoms(positions=xs[0], numbers=self.atomic_numbers.reshape(-1))
        NL = NeighborList(natural_cutoffs(atoms), self_interaction=False)
        NL.update(atoms)
        bonds = NL.get_connectivity_matrix().todense().nonzero()

        bond_lens = distance_pbc(
        gt_traj[:, bonds[0]], gt_traj[:, bonds[1]], jnp.array([30., 30., 30.]))
        mean_bond_lens = bond_lens.mean(axis=0)

        # compute bond lengths with PBC
        bond_lens = distance_pbc(xs[:, bonds[0]], xs[:, bonds[1]], jnp.array([30., 30., 30.]))  # (T, nbonds)

        # compute max deviation per frame
        devs = jnp.abs(bond_lens - mean_bond_lens)  # (T, nbonds)
        max_devs = devs.max(axis=-1)                # (T,)

        # find first time index above threshold
        mask = max_devs > self.stability_threshold
        collapse_pt = jnp.where(mask, size=1, fill_value=-1)[0].item()  # -1 if none found

        # compute h(r)
        hist_pred = get_hr(xs[0:collapse_pt], bins)
        hr_mae = mae(hist_pred, hist_gt, self.xlim)

        # compute collapse time in ps
        collapse_time_ps = (self.integration_timestep * self.subsample * collapse_pt) / (ase.units.fs * 1_000) if collapse_pt != -1 else -1

        with LogFigure("Radial distribution function", log):
            plt.plot(bins[1:], hist_gt, label='Reference', linewidth=2, linestyle='--')
            plt.plot(bins[1:], hist_pred, label='Simulation', linewidth=2, linestyle='--')
            plt.xlabel('r')
            plt.ylabel('h(r)')
            plt.legend()

        if wandb.run and log:
            wandb.log({"collapse_time_ps": collapse_time_ps})
            wandb.log({"hr_mae": hr_mae})

        return {"hr_mae": hr_mae}


class LogASETraj(SimulationMetrics):
    # default for subsample should be something high
    def __init__(self, subsample=100, align_positions=False, **kwargs):
        self.align_positions = align_positions
        super().__init__(subsample=subsample, **kwargs)

    def _compute_metrics(self, ts, xs, ps, vs, fs, aux, log):
        if self.align_positions:
            xs = align_positions(xs)

        xs_filtered = xs[jnp.isfinite(xs).all(axis=(1, 2))]
        x_min, y_min = xs_filtered.reshape(-1, 3)[:, :2].min(axis=0)
        x_max, y_max = xs_filtered.reshape(-1, 3)[:, :2].max(axis=0)
        w = x_max - x_min
        h = y_max - y_min
        padding = 0.1 * max(w, h)

        with tempfile.NamedTemporaryFile(suffix=".mp4") as temp_video_path:
            with PltVideo(temp_video_path.name) as pltvid:
                for x in xs:
                    f, ax = plt.subplots()
                    mol = Atoms(numbers=self.atomic_numbers.reshape(-1), positions=x)
                    plot_atoms(mol)

                    ax.set_xlim(-padding, w + padding)
                    ax.set_ylim(-padding, h + padding)
                    pltvid.add_frame(f)

            pltvid.log("Trajectory", log)
