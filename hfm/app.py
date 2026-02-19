from omegaconf import DictConfig, OmegaConf
import hydra
from typing import Optional
import os

def get_hydra_output_dir():
    return hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

def get_slurm_job_id() -> Optional[str]:
    if "SLURM_JOB_ID" in os.environ:
        return os.environ["SLURM_JOB_ID"]
    return None

def init_hydra_wandb(cfg: DictConfig):
    import subprocess
    try:
        nvidia_smi_output = subprocess.check_output(["nvidia-smi"], text=True)
        print("nvidia-smi output:\n", nvidia_smi_output)
    except Exception as e:
        print("Could not run nvidia-smi:", e)

    import wandb
    import os
    import glob
    import re
    from pathlib import Path
    from hfm.datasets.utils import register_omegaconf_unit_resolvers

    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"
    register_omegaconf_unit_resolvers()

    if cfg.get("load_params", False):
        # try to resume wandb run
        print("Resume from previous run...")
        workdir = Path(cfg.load_params).parent
        assert workdir is not None, "Please provide a workdir to resume from."

        cfg = OmegaConf.load(os.path.join(workdir, ".hydra", "config.yaml"))
        
        matches = glob.glob(os.path.join(workdir, "wandb", "latest-run", "run-*.wandb"))
        if matches:
            filename = os.path.basename(matches[0])
            wandb_run_id = re.findall(r'run-([a-z0-9]+)\.wandb', filename)[0]
        else:
            raise Exception("Can't retrieve wandb_run_id. No matching wandb log found.")

        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=wandb_run_id,
            resume="must",
            dir=workdir,
        )
    else:
        workdir = get_hydra_output_dir()
        config = OmegaConf.to_container(cfg, resolve=True)
        config["workdir"] = workdir
        wandb.init(            
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            group=cfg.wandb.group,
            name=f"EXP-{cfg.globals.exp}",
            dir=workdir,
            config=config,
            mode=cfg.wandb.mode
        )

        print(f"Running on machine: {hostname} in {workdir}")
    slurm_job_id = get_slurm_job_id()
    if slurm_job_id is not None:
        print(f"SLURM job id: {slurm_job_id}")
    return workdir


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    import sys
    trainer = None
    try:
        workdir = init_hydra_wandb(cfg)

        trainer = hydra.utils.instantiate(
            cfg.trainer,
            workdir=workdir
        )

        debug = cfg.get("debug", False)
        seed = cfg.get("seed", 42)
        trainer(debug=debug, seed=seed, load_params=cfg.load_params)

        # after training has completed, compute metrics if specified
        for metric in cfg.get("metrics", []):
            print(f"Computing metric: {metric}")
            hydra.utils.instantiate(
                metric,
                trainer=trainer
            )()
    except KeyboardInterrupt:
        print("\n[!] Caught CTRL+C — terminating...")
        if trainer is not None:
            trainer.shutdown()
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"Exception occurred: {e}")
        traceback.print_exc()
        if trainer is not None:
            trainer.shutdown()
        raise e
    
    if trainer is not None:
        trainer.shutdown()

if __name__ == "__main__":
    main()
