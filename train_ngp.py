from src.model import NGPTrainer
from src.common import NerfSynthetic
import toml
import optax
from tqdm import tqdm
import jax
from jax import numpy as jnp
import os
import shutil

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"
# dataset_dir = "../baseline/data/nerf_synthetic"
dataset_dir = "../nerf_synthetic"

def train_ngp(log2_hashmap_size, dataset_name):
    config = toml.load("configs/ngp/base.toml")
    config["Hash"]["log2_hashmap_size"] = log2_hashmap_size
    dataset = NerfSynthetic(f"{dataset_dir}/{dataset_name}")
    snapshot = NGPTrainer.load_msgpack(f"baseline/{dataset_name}.msgpack")
    trainer = NGPTrainer(
        {
            "config": config,
            "density_grid": snapshot["density_grid"]
        }, dataset)

    optimizer = optax.adam(learning_rate=1e-2, b1=0.9, b2=0.99, eps=1e-15)
    opt_state = optimizer.init(trainer.params)

    @jax.jit
    def update_with(grads, opt_state, params):
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return opt_state, params

    with tqdm(total=10000) as pbar:
        for j in range(0, 10000):
            loss, grads = trainer.train_step(1 << 18)
            opt_state, trainer.params = update_with(grads, opt_state,
                                                    trainer.params)
            pbar.update(1)
            pbar.set_description(f'Loss: {-10 * jnp.log10(loss)} dB')

    shutil.copy(f"baseline/{dataset_name}.msgpack",
                f"a/{dataset_name}_{log2_hashmap_size}.msgpack")
    trainer.save_snapshot(f"a/{dataset_name}_{log2_hashmap_size}.msgpack")


if __name__ == "__main__":
    for i in range(16, 20):
        for name in [
                "chair", "drums", "ficus", "hotdog", "lego", "materials",
                "mic", "ship"
        ]:
            train_ngp(i, name)
