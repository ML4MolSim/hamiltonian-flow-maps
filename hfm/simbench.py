from omegaconf import DictConfig, OmegaConf
import hydra


def get_hydra_output_dir():
    return hydra.core.hydra_config.HydraConfig.get().runtime.output_dir


def load_model(model_key, model_path):
    import glob
    import os
    import re
    from hfm.datasets.utils import register_omegaconf_unit_resolvers
    from pathlib import Path
    from datetime import datetime

    mod_time = os.path.getmtime(model_path)
    print(f"Loading model {model_key} from path: {model_path}")
    print("Model was trained at:", datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S"))

    model_workdir = Path(model_path).parent
    cfg_model = OmegaConf.load(os.path.join(model_workdir, ".hydra", "config.yaml"))
    register_omegaconf_unit_resolvers()
    OmegaConf.resolve(cfg_model)
    reference = "not found"

    # Try to reference the previous wandb run that created the model
    matches = glob.glob(os.path.join(model_workdir, "wandb", "latest-run", "run-*.wandb"))
    if matches:
        filename = os.path.basename(matches[0])
        wandb_run_id = re.findall(r'run-([a-z0-9]+)\.wandb', filename)[0]
        print("Run id from where the model was created: {}".format(wandb_run_id))
        reference = f"https://wandb.ai/{cfg_model.wandb.entity}/{cfg_model.wandb.project}/runs/{wandb_run_id}"
    else:
        print("Can't retrieve wandb_run_id. No matching wandb log found.")

    return cfg_model.model, cfg_model.data_module, reference


def init_config(cfg: DictConfig):
    from hfm.app import init_hydra_wandb

    # Load relevant stuff from the model config to update simulator config
    first_element = True
    wandb_references = {}
    for model_key, model_path in cfg.load_models.items():
        model_cfg, data_module_cfg, wref = load_model(model_key, model_path)
        wandb_references[model_key] = wref
        
        if first_element and cfg.get("load_dm", True):
            # use the first referenced model to update the data module config
            OmegaConf.update(cfg, "data_module", data_module_cfg, force_add=True)
            first_element = False

        OmegaConf.update(cfg, f"model_{model_key}", model_cfg, force_add=True)
        OmegaConf.update(cfg, f"params_{model_key}", model_path, force_add=True) 

    # Now init the simulator config
    init_hydra_wandb(cfg)

    import wandb
    notes = ""
    for key, ref in wandb_references.items():
        wandb.run.config[f"model_{key}_reference"] = ref
        notes = notes + f"Run that created model {key}:\n{ref}\n"
    wandb.run.notes = notes

    datamodule = hydra.utils.instantiate(
        cfg.data_module
    )
    return datamodule, cfg


def run_simbench(datamodule, cfg: DictConfig):
    import os
    import numpy as np
    import wandb
    from hfm import utils
    import jax
    import jax.numpy as jnp
    from hfm.utils import force_temperature_fn

    if cfg.globals.get("debug_nans", False):
        jax.config.update("jax_debug_nans", True)
        jax.config.update("jax_debug_infs", True)

    rng = jax.random.PRNGKey(cfg.globals.seed)
    rng_init, rng_sim, rng_dummy = jax.random.split(rng, 3)

    start_pos = datamodule.start_geometry
    masses = datamodule.masses
    T_start = jnp.array([cfg.globals.T_start]).reshape(1, 1)

    start_mom = utils.maxwell_boltzmann_distribution(rng_init, masses, T_start, n_dim=cfg.globals.n_dim, force_temperature=True)
    zero_rot = cfg.globals.get("zero_rot", True)
    zero_drift = cfg.globals.get("zero_drift", True)

    print("Generating start conditions...")
    print("zero rot:", zero_rot)
    print("zero drift:", zero_drift)

    if zero_rot:
        start_mom = utils.zero_rotation(start_pos, start_mom, masses, force_temperature=False)

    if zero_drift:
        start_mom = utils.stationary(start_mom, masses, force_temperature=False)

    # here we correctly rescale the momenta with the proper dof
    start_mom = force_temperature_fn(start_mom, masses, T_start, zero_drift=zero_drift, zero_rot=zero_rot)  # (B, n_dim)

    simulator = hydra.utils.instantiate(
        cfg.sim_env.simulator,
    )

    # instantiate metrics already to check for potential errors
    metrics = [hydra.utils.instantiate(metric) for metric in cfg.metrics]

    with jax.debug_nans(False):
        with jax.debug_infs(False):
        # instantiate metrics already to check for potential errors
            xs = ps = vs = fs = jax.random.normal(rng_dummy, (25, *start_pos.shape[1:]))
            traj_data = {"xs": xs, "ps": ps, "vs": vs, "fs": fs}

            for metric in metrics:
                # test the metric with dummy data
                metric(traj_data, log=False)

    # repeat the start pos and mom for multiple parallel runs
    start_pos = start_pos.repeat(cfg.globals.n_parallel_runs, axis=0)
    start_mom = start_mom.repeat(cfg.globals.n_parallel_runs, axis=0)

    print("Starting simulation...")
    xs, ps, vs, fs = simulator(start_pos, start_mom, cfg.globals.simulation_length, rng_sim, save_every=cfg.globals.save_every)
    print("Simulation done, generated {} steps".format(xs.shape[1]))

    print(f"Saving trajectory to {get_hydra_output_dir()}/traj.npz")
    np.savez(f"{get_hydra_output_dir()}/traj.npz", **{"xs": xs, "ps": ps, "vs": vs, "fs": fs})

    hr_mae = []
    for idx in range(xs.shape[0]):
        print(f"Computing metrics for run {idx}...")
        traj_data = {"xs": xs[idx], "ps": ps[idx], "vs": vs[idx], "fs": fs[idx]}
        for metric in metrics:
            try:
                print("Computing metric: {}".format(metric.__class__.__name__))
                traj_data = metric(traj_data, log=True)
            except Exception as e:
                print(f"Error computing metric {metric.__class__.__name__} for run {idx}: {e}")

        if "hr_mae" in traj_data:
            hr_mae.append(traj_data["hr_mae"])

        if wandb.run and cfg.get("upload_traj_wandb", False):
            print(f"Saving trajectory for run {idx} to wandb...")
            fname = f"/tmp/traj_{wandb.run.id}_{idx}.npz"
            np.savez(fname, **traj_data)

            file_size_mb = os.path.getsize(fname) / (1024 * 1024)
            if file_size_mb > 20:
                print(f"Warning: File {fname} is larger than 20 MiB ({file_size_mb:.2f} MiB). Skipping upload to wandb.")
            else:
                artifact = wandb.Artifact(f"simulation_traj{wandb.run.id}_{idx}", type="numpy file")
                artifact.add_file(fname)
                wandb.log_artifact(artifact)
            os.remove(fname)  # clean up

        print(f"{idx+1}/{xs.shape[0]} parallel runs logged.\n")

    hr_mae = jnp.array(hr_mae)
    wandb.summary["hr_mae_mean"] = jnp.mean(hr_mae)
    wandb.summary["hr_mae_std"] = jnp.std(hr_mae)

    print("Exiting...")
    wandb.finish()


@hydra.main(version_base=None, config_path="../configs", config_name="simbench")
def main(cfg: DictConfig):
    import sys
    datamodule = None
    try:
        datamodule, cfg = init_config(cfg)
        run_simbench(datamodule, cfg)
    except KeyboardInterrupt:
        print("\n[!] Caught CTRL+C — terminating...")
        if datamodule is not None:
            datamodule.shutdown()
        sys.exit(1)
    except Exception as e:
        print(f"Exception occurred: {e}")
        if datamodule is not None:
            datamodule.shutdown()
        raise e

if __name__ == "__main__":
    main()
