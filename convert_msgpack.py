import pickle
from jax import numpy as jnp
import toml
from src.common import NerfSynthetic
from src.model import ENVRTrainer
import os
import msgpack
import numpy as np

name = "ficus"
with open(f"{name}.bin", "rb") as f:
    snapshot = pickle.load(f)

#a = snapshot["params"]["envr_network"]

config = toml.load("configs/ngp/base.toml")
density_grid = jnp.array(snapshot["density_grid"])
trainer = ENVRTrainer(
    {
        "config": None,
        "density_grid": snapshot["density_grid"],
        "params": snapshot["params"]
    }, None)
trainer.save_msgpack(f"{name}.msgpack")