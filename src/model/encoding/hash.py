import math
from typing import Callable
from dataclasses import dataclass
import jax
from jax import lax, numpy as jnp
import numpy as np
from functools import partial
from itertools import product
import haiku as hk
from ...quantization import lsq, init_lsq


# grad val
def normal_index(grid_local, grid_resolution):
    stride = jnp.asarray(
        [1, grid_resolution, grid_resolution * grid_resolution],
        dtype=jnp.uint32)
    return jnp.sum(grid_local * stride, axis=1)


def hash_index(hashmap_size, grid_local, grid_resolution):
    primes = jnp.asarray([1, 2654435761, 805459861], dtype=jnp.uint32)
    index = primes * grid_local
    index = lax.bitwise_xor(lax.bitwise_xor(index[:, 0], index[:, 1]),
                            index[:, 2])
    index = index % hashmap_size
    return index


def hash_encoding_level(index_function, i, val):
    grid = val["grid"]
    offset = val["hashmap_offsets"][i]
    scale = val["scales"][i]
    grid_resolution = val["grid_resolutions"][i]
    inputs = val["inputs"]
    result = val["result"]

    # pos fract
    pos = scale * jnp.float32(inputs) + 0.5
    temp = jnp.floor(pos)
    pos_grid = jnp.uint32(temp)
    pos = pos - temp

    # pos_grid_local
    pos_grid_local = list(
        product([pos_grid[0], pos_grid[0] + 1], [pos_grid[1], pos_grid[1] + 1],
                [pos_grid[2], pos_grid[2] + 1]))
    # manual loop unrolling
    pos_grid_local = jnp.array(pos_grid_local)
    #
    weight = list(
        product([1 - pos[0], pos[0]], [1 - pos[1], pos[1]],
                [1 - pos[2], pos[2]]))
    weight = jnp.array(weight)
    weight = jnp.prod(weight, axis=1)

    index = index_function(pos_grid_local, grid_resolution) + offset

    value = grid[index]
    result = result.at[i].set(jnp.sum(value * weight[:, jnp.newaxis], axis=0))

    val["result"] = result
    return val


class HashEncoding(hk.Module):
    N_POS_DIMS: int = 3
    N_FEATURES_PER_LEVEL: int = 2

    def __init__(self, encoding: dict, name=None):
        super().__init__(name)

        self.log2_hashmap_size = encoding.get("log2_hashmap_size", 19)
        if ("n_features" in encoding or "n_grid_features" in encoding):
            self.n_features = encoding["n_features"] if "n_features" in encoding \
                else encoding["n_grid_features"]
        else:
            self.n_features = self.N_FEATURES_PER_LEVEL * \
                encoding.get("n_levels", 16)
        self.base_resolution = encoding.get("base_resolution", 16)

        self.n_levels = math.ceil(self.n_features / self.N_FEATURES_PER_LEVEL)
        self.per_level_scale = math.exp(
            math.log(2048 * 1 / self.base_resolution) / (self.n_levels - 1))

        self.hashmap_offsets = []
        self.scales = []
        self.grid_resolutions = []

        self.n_params = 0
        self.n_levels_normal = 0
        for i in range(0, self.n_levels):
            scale = math.pow(2.0, i * math.log2(
                self.per_level_scale)) * self.base_resolution - 1.0
            grid_resolution = int(math.ceil(scale)) + 1
            params_in_level = math.pow(grid_resolution, self.N_POS_DIMS)
            # find next multiple of 8, for memory alignment
            params_in_level = math.ceil(params_in_level / 8) * 8
            params_in_level = min(params_in_level,
                                  round(math.pow(2, self.log2_hashmap_size)))
            params_in_level = int(params_in_level)

            if math.pow(grid_resolution, 3) <= params_in_level:
                self.n_levels_normal += 1

            self.hashmap_offsets.append(self.n_params)
            self.scales.append(scale)
            self.grid_resolutions.append(grid_resolution)
            self.n_params += params_in_level
        self.hashmap_offsets = jnp.array(self.hashmap_offsets)
        self.scales = jnp.array(self.scales)
        self.grid_resolutions = jnp.array(self.grid_resolutions)

    def __call__(self, inputs):
        result = jnp.zeros((self.n_levels, 2), dtype=jnp.float32)
        grid = hk.get_parameter(f"grid",
                                shape=(self.n_params, 2),
                                dtype=jnp.float32,
                                init=hk.initializers.RandomUniform(
                                    minval=-1e-4, maxval=1e-4))
        s = hk.get_parameter(f"lsq",
                             shape=(1, ),
                             dtype=jnp.float32,
                             init=hk.initializers.Constant(init_lsq(grid, 4)))
        #grid = lsq(grid, s, 4)

        val = {
            "grid": grid,
            "hashmap_offsets": self.hashmap_offsets,
            "scales": self.scales,
            "grid_resolutions": self.grid_resolutions,
            "inputs": inputs,
            "result": result
        }
        val = lax.fori_loop(0, self.n_levels_normal,
                            partial(hash_encoding_level, normal_index), val)

        val = lax.fori_loop(
            self.n_levels_normal, self.n_levels,
            partial(
                hash_encoding_level,
                partial(hash_index,
                        round(math.pow(2, self.log2_hashmap_size)))), val)

        return jnp.reshape(val["result"], (32, ))
