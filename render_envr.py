import shutil
from functools import partial
import math
from jax import dlpack, lax, numpy as jnp
import jax
import toml
import pickle
import haiku as hk
from tqdm import tqdm

from src import jax_nerf
from src.common import NerfSynthetic
from src.model import ENVRTrainer
from src.common import MIN_CONE_STEPSIZE, NERF_CASCADES

from scripts.common import *

import os
import numpy as np
import imageio.v3 as imageio

# 输出目录设置
output_dir = "output"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

resolution = [800, 800]

# dataset_dir = "../baseline/data/nerf_synthetic"
dataset_dir = "../nerf_synthetic"
dataset_name = "chair"
dataset = NerfSynthetic(f"{dataset_dir}/{dataset_name}",
                        ["test"])

x = ((jnp.linspace(0, resolution[0], resolution[0], endpoint=False) + 0.5) /
     resolution[0] - dataset.principal[0]) * resolution[0] / dataset.focal[0]
y = ((jnp.linspace(0, resolution[1], resolution[1], endpoint=False) + 0.5) /
     resolution[1] - dataset.principal[1]) * resolution[1] / dataset.focal[1]

xv, yv = jnp.meshgrid(x, y)
xv = jnp.ravel(xv)
yv = jnp.ravel(yv)


@partial(jax.vmap, in_axes=(0, 0, None))
def get_ray(xv, yv, transform_matrix):
    ray_o = transform_matrix[:, 3]
    ray_d = jnp.array([xv, yv, 1.0])
    ray_d = jnp.matmul(transform_matrix[:3, :3], ray_d)
    ray_d = ray_d / jnp.linalg.norm(ray_d)

    return jnp.array([ray_o, ray_d])


with open(f"{output_dir}/{dataset_name}.bin", "rb") as f:
    snapshot = pickle.load(f)
config = toml.load("configs/ngp/base.toml")
density_grid = jnp.array(snapshot["density_grid"])
trainer = ENVRTrainer(
    {
        "config": config,
        "density_grid": snapshot["density_grid"],
        "params": snapshot["params"]
    }, dataset)
density_grid = dlpack.to_dlpack(density_grid)


def unwarp_dt(dt):
    max_stepsize = MIN_CONE_STEPSIZE() * (1 << (NERF_CASCADES() - 1))
    return dt * (max_stepsize - MIN_CONE_STEPSIZE()) + MIN_CONE_STEPSIZE()


def logistic(x):
    return 1.0 / (1.0 + jnp.exp(-x))


def composite(local_network_output, local_rgba, local_dt):
    T = 1. - local_rgba[3]
    dt = unwarp_dt(local_dt)
    alpha = 1. - jnp.exp(-jnp.exp(local_network_output[3]) * dt)
    rgb = logistic(local_network_output[0:3])
    weight = alpha * T
    return jnp.append(rgb * weight, weight)


def composite_or_not(local_network_output, alive, local_rgba, local_dt):
    return lax.cond(alive, composite, lambda x, y, z: jnp.zeros(shape=(4, )),
                    local_network_output, local_rgba, local_dt)


def early_termination(local_rgba):
    return local_rgba[3] <= 1 - 1e-4


params = hk.data_structures.merge(trainer.trainable_params,
                                  trainer.non_trainable_params)

n_images = len(dataset.transform_matrix)
totpsnr = 0
totssim = 0
minpsnr = 1000
maxpsnr = 0

with tqdm(total=n_images) as pbar:
    img_dir = os.path.join(output_dir, "img")
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    for ith in range(0, n_images):
        ray = get_ray(xv, yv, dataset.transform_matrix[ith])
        alive = jnp.ones((resolution[0] * resolution[1], ), dtype=jnp.bool_)
        t = jnp.zeros((resolution[0] * resolution[1], ))
        coords = jnp.zeros((resolution[0] * resolution[1], 7))

        ray = dlpack.to_dlpack(ray)
        t = dlpack.to_dlpack(t)

        rgba = jnp.zeros(shape=(resolution[0] * resolution[1], 4),
                         dtype=jnp.float32)

        while jnp.count_nonzero(alive) != 0:

            coords = dlpack.to_dlpack(coords, True)
            alive = dlpack.to_dlpack(alive, True)
            jax_nerf.ngp_advance_pos_nerf(ray, density_grid, alive, t, coords,
                                          [dataset.aabb.min, dataset.aabb.max],
                                          resolution[0] * resolution[1])

            alive = dlpack.from_dlpack(alive)
            coords = dlpack.from_dlpack(coords)
            network_output = trainer.model(params, coords)
            partial_result = jax.vmap(composite_or_not,
                                      in_axes=(0, 0, 0,
                                               0))(network_output, alive, rgba,
                                                   jnp.swapaxes(coords, 0,
                                                                1)[3])
            rgba += partial_result
            ea = jax.vmap(early_termination)(rgba)
            alive = jnp.logical_and(ea, alive)
            #print(jnp.count_nonzero(alive))

        arrOut = np.array(rgba)
        arrOut = np.uint8(arrOut * 255.0)
        arrOut = np.reshape(arrOut, (resolution[1], resolution[0], 4))
        imageio.imwrite(f"{img_dir}/out_{ith}.png", arrOut)

        ref = np.array(dataset.images[ith])
        ref_image = ref.astype(np.float32) / 255.0

        ref_image[..., :3] = srgb_to_linear(ref_image[..., :3])
        ref_image[..., :3] *= ref_image[..., 3:4]
        ref_image[..., :3] = linear_to_srgb(ref_image[..., :3])

        rgba = np.reshape(np.array(rgba), (resolution[0], resolution[1], 4))

        diff = np.abs(rgba[..., :3] - ref_image[..., :3]) * 255.0
        diff = np.uint8(diff)
        imageio.imwrite(f"{img_dir}/diff_{ith}.png", diff)

        rgba = np.clip(rgba[..., :3], 0.0, 1.0)
        ref_image = np.clip(ref_image[..., :3], 0.0, 1.0)
        mse = float(compute_error("MSE", rgba, ref_image))
        psnr = mse2psnr(mse)
        ssim = float(compute_error("SSIM", rgba, ref_image))

        totpsnr += psnr
        totssim += ssim
        minpsnr = psnr if psnr < minpsnr else minpsnr
        maxpsnr = psnr if psnr > maxpsnr else maxpsnr

        pbar.update(1)
        pbar.set_description(f'PSNR: {psnr} dB')

print(
    f"PSNR={totpsnr / n_images} [min={minpsnr} max={maxpsnr}] SSIM={totssim / n_images}"
)
