import pickle
from dataclasses import dataclass
from typing import Any
import jax.numpy as jnp
import jax

MIN_CKPT_VERSION = "0.0.1"

@dataclass
class Checkpoint:
    version: int
    params: Any


def load_params_from_path(path: str, model_version: str):
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    
    if not isinstance(ckpt, Checkpoint):
        print("WARNING: Loading legacy checkpoint!! Please retrain the model using the correct cutoff implementation.")
        ckpt = Checkpoint(version="0.0.1", params=ckpt)

    if not (ckpt.version >= MIN_CKPT_VERSION):
        raise RuntimeError(
            f"Checkpoint version {ckpt.version} is out of date."
            f" Please use version >= {MIN_CKPT_VERSION}."
        )
    
    if not ckpt.version == model_version:
        raise RuntimeError(
            f"Checkpoint version {ckpt.version} does not match the model version {model_version}."
        )

    return ckpt.params


def save_params_to_path(params: Any, path: str, model_version: str):
    ckpt = Checkpoint(version=model_version, params=params)
    with open(path, "wb") as f:
        pickle.dump(ckpt, f)


def summarize_leaf(x):
    if isinstance(x, jnp.ndarray):
        return f"{x.dtype}[{', '.join(map(str, x.shape))}]"
    return type(x).__name__


def pretty_print_params(params):
    print(jax.tree_util.tree_map(summarize_leaf, params))
