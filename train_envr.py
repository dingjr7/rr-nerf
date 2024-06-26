from src.model import *
from src.common import NerfSynthetic
import toml
import optax
from tqdm import tqdm
import jax
from jax import dlpack, numpy as jnp
import os
import shutil

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "true"

# dataset_dir = "../baseline/data/nerf_synthetic"
dataset_dir = "../nerf_synthetic"

# 输出目录设置
output_dir = "output"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

def train(dataset_name):
    config = toml.load("configs/ngp/base.toml")
    dataset = NerfSynthetic(f"{dataset_dir}/{dataset_name}")
    snapshot = NGPTrainer.load_msgpack(f"filtered/{dataset_name}.msgpack")
    density_grid = dlpack.from_dlpack(snapshot["density_grid"])
    trainer = ENVRTrainer(
        {
            "config": config,
            "density_grid": density_grid,
            "ngp_params": snapshot["params"]
        }, dataset)
    optimizer = optax.adam(learning_rate=1e-2, b1=0.9, b2=0.99, eps=1e-15)
    opt_state = optimizer.init(trainer.trainable_params)

    @jax.jit
    def update_with(grads, opt_state, params):
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return opt_state, params

    with tqdm(total=35000) as pbar:
        for j in range(0, 35000):
            loss, grads = trainer.train_step(1 << 18)
            opt_state, trainer.trainable_params = update_with(
                grads, opt_state, trainer.trainable_params)
            pbar.update(1)
            pbar.set_description(f'Loss: {-10 * jnp.log10(loss)} dB')

    trainer.save_snapshot(f"{output_dir}/{dataset_name}.bin")


if __name__ == "__main__":
    for name in [
            "chair", "lego", "drums", "ficus", "hotdog", "materials", "mic",
            "ship"
    ]:
        train(name)
