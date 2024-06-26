import haiku as hk
from jax import nn
from jax import random, lax, dlpack, numpy as jnp
import toml

from .encoding.hash import *
from .encoding.spherical_harmonics import *

from .ngp import NGPNetwork

from .networks import *

from ..quantization import *
from .. import jax_nerf

import pickle
import msgpack


class ENVRNetwork(hk.Module):

    def __init__(self, max_offset, name=None):
        super().__init__(name)
        self.max_offset = int(max_offset)

    def __call__(self, x):
        pos = x[0:3]

        #get which region is in
        pos = 8 * jnp.float32(pos)
        temp = jnp.floor(pos)
        region = jnp.uint32(temp)
        pos = pos - temp

        offset_table = hk.get_parameter(f"offset_table",
                                        shape=(8 * 8 * 8, ),
                                        dtype=jnp.uint32,
                                        init=hk.initializers.Constant(0))

        offset = offset_table[8 * 8 * region[2] + 8 * region[1] + region[0]]

        #positional encoding
        resolution = 32
        grid = hk.get_parameter(
            f"grid",
            shape=(self.max_offset,
                   (resolution + 1) * (resolution + 1) * (resolution + 1), 16),
            dtype=jnp.float32,
            init=hk.initializers.RandomUniform(minval=-1e-4, maxval=1e-4))
        s = hk.get_parameter(f"grid_lsq",
                             shape=(1, ),
                             dtype=jnp.float32,
                             init=hk.initializers.Constant(init_lsq(grid, 4)))
        grid = lsq(grid, s, 4)

        pos = resolution * jnp.float32(pos)
        temp = jnp.floor(pos)
        pos_grid = jnp.uint32(temp)
        pos = pos - temp

        pos_grid = jnp.minimum(
            pos_grid,
            jnp.array([resolution - 1, resolution - 1, resolution - 1]))

        # pos_grid_local
        pos_grid_local = list(
            product([pos_grid[0], pos_grid[0] + 1],
                    [pos_grid[1], pos_grid[1] + 1],
                    [pos_grid[2], pos_grid[2] + 1]))
        # manual loop unrolling
        pos_grid_local = jnp.array(pos_grid_local)
        #
        weight = list(
            product([1 - pos[0], pos[0]], [1 - pos[1], pos[1]],
                    [1 - pos[2], pos[2]]))
        weight = jnp.array(weight)
        weight = jnp.prod(weight, axis=1)

        stride = jnp.asarray(
            [1, resolution + 1, (resolution + 1) * (resolution + 1)],
            dtype=jnp.uint32)
        pos_grid_local = jnp.sum(pos_grid_local * stride, axis=1)

        value = grid.at[offset, pos_grid_local].get()

        pos_encoding = jnp.sum(value * weight[:, jnp.newaxis], axis=0)

        #directional encoding
        sh_encoding = jnp.zeros((18, ), dtype=jnp.float32)
        dir = x[4:7] * 2 - 1
        for i in range(0,3):
           sh_encoding = sh_encoding.at[i * 6].set(jnp.sin(jnp.pi*dir[i]))
           sh_encoding = sh_encoding.at[i * 6 + 1].set(jnp.cos(jnp.pi*dir[i]))
           sh_encoding = sh_encoding.at[i * 6 + 2].set(jnp.sin(2*jnp.pi*dir[i]))
           sh_encoding = sh_encoding.at[i * 6 + 3].set(jnp.cos(2*jnp.pi*dir[i]))
           sh_encoding = sh_encoding.at[i * 6 + 4].set(jnp.sin(4*jnp.pi*dir[i]))
           sh_encoding = sh_encoding.at[i * 6 + 5].set(jnp.cos(4*jnp.pi*dir[i]))

        x = nn.relu(pos_encoding)
        weight = hk.get_parameter(f"density_1",
                                  shape=(self.max_offset, 16, 8),
                                  dtype=jnp.float32,
                                  init=hk.initializers.VarianceScaling(
                                      1.0, "fan_avg", "uniform"))
        s = hk.get_parameter(f"density_1_lsq",
                             shape=(1, ),
                             dtype=jnp.float32,
                             init=hk.initializers.Constant(init_lsq(weight,
                                                                    8)))
        weight = lsq(weight, s, 8)
        density_nn_output = jnp.matmul(x, weight[offset])

        weight = hk.get_parameter(f"rgb_0",
                                  shape=(self.max_offset, 26, 32),
                                  dtype=jnp.float32,
                                  init=hk.initializers.VarianceScaling(
                                      1.0, "fan_avg", "uniform"))
        s = hk.get_parameter(f"rgb_0_lsq",
                             shape=(1, ),
                             dtype=jnp.float32,
                             init=hk.initializers.Constant(init_lsq(weight,
                                                                    8)))
        weight = lsq(weight, s, 8)
        x = jnp.matmul(jnp.concatenate([density_nn_output, sh_encoding]),
                       weight[offset])
        x = nn.relu(x)

        weight = hk.get_parameter(f"rgb_1",
                                  shape=(self.max_offset, 32, 3),
                                  dtype=jnp.float32,
                                  init=hk.initializers.VarianceScaling(
                                      1.0, "fan_avg", "uniform"))
        s = hk.get_parameter(f"rgb_1_lsq",
                             shape=(1, ),
                             dtype=jnp.float32,
                             init=hk.initializers.Constant(init_lsq(weight,
                                                                    8)))
        weight = lsq(weight, s, 8)
        rgb_nn_output = jnp.matmul(x, weight[offset])

        return jnp.concatenate([rgb_nn_output[0:3], density_nn_output[0:1]])


class ENVRTrainer():

    def __init__(self, config: dict, dataset):
        self.config = config["config"]

        self.density_grid = config["density_grid"]
        density_grid = list(np.array(self.density_grid))
        self.density_grid = dlpack.to_dlpack(self.density_grid, True)

        offset_table, self.max_offset = jax_nerf.envr_get_offset_table(
            density_grid)

        self.offset_table = jnp.array(offset_table, dtype=jnp.uint32)

        self._model = hk.transform(lambda x: hk.vmap(
            ENVRNetwork(self.max_offset, "envr_network"), split_rng=False)(x))
        self._model = hk.without_apply_rng(self._model)
        self.model = jax.jit(self._model.apply)

        def composite(trainable_params, non_trainable_params, input, dl_dmlp):
            params = hk.data_structures.merge(trainable_params,
                                              non_trainable_params)
            mlp_result = self.model(params, input)
            return jnp.vdot(mlp_result, dl_dmlp)

        self.grad = jax.jit(jax.grad(composite))

        self.key = random.PRNGKey(42)

        if "params" in config.keys():
            params = config["params"]
        else:
            params = self._model.init(rng=self.key, x=jnp.zeros((1, 7)))
        self.trainable_params, self.non_trainable_params = hk.data_structures.partition(
            lambda m, n, p: n != "offset_table", params)
        self.non_trainable_params["envr_network"][
            "offset_table"] = self.offset_table

        if "ngp_params" in config.keys():
            cfg = toml.load("configs/ngp/base.toml")
            # TODO load ngp params
            model = hk.transform(lambda x: hk.vmap(
                NGPNetwork(cfg, "ngp_network"), split_rng=False)(x))
            model = hk.without_apply_rng(model)
            self.ngp_model = jax.jit(model.apply)
            self.ngp_params = config["ngp_params"]

            def distill(trainable_params, non_trainable_params, ngp_params,
                        coords):
                params = hk.data_structures.merge(trainable_params,
                                                  non_trainable_params)
                students_result = self.model(params, coords)
                teacher_result = self.ngp_model(ngp_params, coords)
                return jnp.mean((teacher_result - students_result)**2)

            self.distill = jax.jit(jax.value_and_grad(distill))

        self.dataset = dataset

        self.n_rays = 40960

    def save_snapshot(self, path):
        result = {}
        result["config"] = self.config
        params = hk.data_structures.merge(self.trainable_params,
                                          self.non_trainable_params)
        result["params"] = params
        self.density_grid = dlpack.from_dlpack(self.density_grid)
        result["density_grid"] = self.density_grid
        self.density_grid = dlpack.to_dlpack(self.density_grid)

        with open(path, 'wb') as f:
            pickle.dump(result, f)

    def save_msgpack(self, path):
        params = hk.data_structures.merge(self.trainable_params,
                                          self.non_trainable_params)
        params["envr_network"]["grid"] = lsq(
            params["envr_network"]["grid"], params["envr_network"]["grid_lsq"],
            4)
        params["envr_network"]["density_1"] = lsq(
            params["envr_network"]["density_1"],
            params["envr_network"]["density_1_lsq"], 8)
        params["envr_network"]["rgb_0"] = lsq(
            params["envr_network"]["rgb_0"],
            params["envr_network"]["rgb_0_lsq"], 8)
        params["envr_network"]["rgb_1"] = lsq(
            params["envr_network"]["rgb_1"],
            params["envr_network"]["rgb_1_lsq"], 8)
        config = {
            key: np.array(value).tobytes()
            for key, value in params["envr_network"].items()
        }

        self.density_grid = dlpack.from_dlpack(self.density_grid)
        config["density_grid"] = np.array(self.density_grid).tobytes()
        self.density_grid = dlpack.to_dlpack(self.density_grid)

        with open(path, 'wb') as f:
            f.write(msgpack.packb(config))

    def train_step(self, batch_size):

        self.key, subkey = random.split(self.key)

        pixels, bg, ray = self.dataset.sample(subkey, self.n_rays)
        numsteps = jnp.zeros((self.n_rays, 2), dtype=jnp.uint32)
        ray_indices = jnp.zeros((self.n_rays, ), dtype=jnp.uint32)
        coords = jnp.zeros((batch_size * 2, 7), dtype=jnp.float32)

        ray = dlpack.to_dlpack(ray, True)
        numsteps = dlpack.to_dlpack(numsteps, True)
        ray_indices = dlpack.to_dlpack(ray_indices, True)
        coords = dlpack.to_dlpack(coords, True)

        n_ray_total = jax_nerf.ngp_generate_training_sample(
            ray, self.density_grid, numsteps, ray_indices, coords,
            [self.dataset.aabb.min, self.dataset.aabb.max], self.n_rays,
            batch_size * 2)

        coords = dlpack.from_dlpack(coords)

        params = hk.data_structures.merge(self.trainable_params,
                                          self.non_trainable_params)
        network_output = self.model(params, coords)

        assert not jnp.any(jnp.isnan(network_output))

        loss = jnp.zeros((self.n_rays, ), dtype=jnp.float32)
        dloss_doutput = jnp.zeros((batch_size, 4), dtype=jnp.float32)
        coords_compated = jnp.zeros((batch_size, 7), dtype=jnp.float32)

        coords = dlpack.to_dlpack(coords, True)
        network_output = dlpack.to_dlpack(network_output, True)
        pixels = dlpack.to_dlpack(pixels, True)
        bg = dlpack.to_dlpack(bg, True)
        loss = dlpack.to_dlpack(loss, True)
        dloss_doutput = dlpack.to_dlpack(dloss_doutput, True)
        coords_compated = dlpack.to_dlpack(coords_compated, True)

        measured_batch_size, compacted_ray_counter = jax_nerf.ngp_compute_loss(
            numsteps, ray_indices, coords, pixels, bg, network_output, loss,
            dloss_doutput, coords_compated, self.n_rays, n_ray_total,
            batch_size)

        #self.n_rays = int(self.n_rays * batch_size / measured_batch_size)
        #self.n_rays = min(self.n_rays, 1 << 18)

        loss = dlpack.from_dlpack(loss)
        coords_compated = dlpack.from_dlpack(coords_compated)
        dloss_doutput = dlpack.from_dlpack(dloss_doutput)

        assert not jnp.any(jnp.isnan(dloss_doutput))

        gradient = self.grad(self.trainable_params, self.non_trainable_params,
                             coords_compated, dloss_doutput)

        loss = jnp.sum(loss) / compacted_ray_counter

        return loss, gradient

    def train_step_distill(self, batch_size):
        self.key, subkey = random.split(self.key)

        pixels, bg, ray = self.dataset.sample(subkey, self.n_rays)
        numsteps = jnp.zeros((self.n_rays, 2), dtype=jnp.uint32)
        ray_indices = jnp.zeros((self.n_rays, ), dtype=jnp.uint32)
        coords = jnp.zeros((batch_size * 2, 7), dtype=jnp.float32)

        ray = dlpack.to_dlpack(ray, True)
        numsteps = dlpack.to_dlpack(numsteps, True)
        ray_indices = dlpack.to_dlpack(ray_indices, True)
        coords = dlpack.to_dlpack(coords, True)

        n_ray_total = jax_nerf.ngp_generate_training_sample(
            ray, self.density_grid, numsteps, ray_indices, coords,
            [self.dataset.aabb.min, self.dataset.aabb.max], self.n_rays,
            batch_size * 2)

        coords = dlpack.from_dlpack(coords)

        loss, gradient = self.distill(self.trainable_params,
                                      self.non_trainable_params,
                                      self.ngp_params, coords)
        return loss, gradient