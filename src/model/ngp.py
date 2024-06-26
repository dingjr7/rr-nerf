import haiku as hk
from jax import nn
from jax import random, dlpack, numpy as jnp

from .encoding.hash import *
from .encoding.spherical_harmonics import *

from .networks import *

from ..quantization import *
from .. import jax_nerf

import msgpack


class NGPNetwork(hk.Module):

    def __init__(self, config: dict, name=None):
        super().__init__(name)
        self.config = config

        self.config["DensityNetwork"]["n_input_dims"] = 32
        self.config["DensityNetwork"]["n_output_dims"] = 16

        self.config["RgbNetwork"]["n_input_dims"] = 32
        self.config["RgbNetwork"]["n_output_dims"] = 3

    def __call__(self, x):
        hash_encoding = HashEncoding(self.config["Hash"],
                                     "hash_encoding")(x[0:3])
        sh_encoding = SphericalHarmonics()(x[4:7])

        density_nn_output = MLPNetwork(self.config["DensityNetwork"],
                                       "density_network")(hash_encoding)

        rgb_nn_output = MLPNetwork(self.config["RgbNetwork"], "rgb_network")(
            jnp.concatenate([density_nn_output, sh_encoding]))

        return jnp.concatenate([rgb_nn_output[0:3], density_nn_output[0:1]])


class NGPTrainer():

    def __init__(self, config: dict, dataset):
        self.config = config["config"]

        self._model = hk.transform(lambda x: hk.vmap(
            NGPNetwork(self.config, "ngp_network"), split_rng=False)(x))
        self._model = hk.without_apply_rng(self._model)
        self.model = jax.jit(self._model.apply)

        def composite(params, input, dl_dmlp):
            mlp_result = self.model(params, input)
            return jnp.vdot(mlp_result, dl_dmlp)

        self.grad = jax.jit(jax.grad(composite))

        self.key = random.PRNGKey(42)

        if "params" in config.keys():
            self.params = config["params"]
        else:
            self.params = self._model.init(rng=self.key, x=jnp.zeros((1, 7)))

        self.dataset = dataset
        self.density_grid = config["density_grid"]

        self.n_rays = 40960

    @staticmethod
    def load_msgpack(path):
        result = {}

        with open(path, 'rb') as f:
            unpacker = msgpack.Unpacker(f, raw=False)
            config = next(unpacker)

        density_grid = jnp.array(np.frombuffer(
            config['snapshot']['density_grid_binary'], dtype=np.float16),
                                 dtype=jnp.float32)
        density_grid = jnp.pad(density_grid, (0, 128 * 128 * 128 * 8))
        density_grid_bitfield = jnp.zeros((128 * 128 * 128, ), dtype=jnp.uint8)

        density_grid = dlpack.to_dlpack(density_grid)
        density_grid_bitfield = dlpack.to_dlpack(density_grid_bitfield)

        jax_nerf.update_density_grid_mean_and_bitfield(density_grid,
                                                       density_grid_bitfield,
                                                       1)

        result["density_grid"] = density_grid_bitfield

        #load snapshot
        params_binary = config['snapshot']['params_binary']
        params_binary = jnp.float32(
            jnp.asarray(
                np.frombuffer(params_binary, dtype=np.float16, offset=0)))

        params = {}
        params['ngp_network/density_network'] = {}
        params['ngp_network/density_network']["linear_0"] = jnp.transpose(
            jnp.reshape(params_binary[:32 * 64], (64, 32)))
        params_binary = params_binary[32 * 64:]

        params['ngp_network/density_network']["linear_1"] = jnp.transpose(
            jnp.reshape(params_binary[:16 * 64], (16, 64)))
        params_binary = params_binary[16 * 64:]

        params['ngp_network/rgb_network'] = {}
        params['ngp_network/rgb_network']["linear_0"] = jnp.transpose(
            jnp.reshape(params_binary[:32 * 64], (64, 32)))
        params_binary = params_binary[32 * 64:]

        params['ngp_network/rgb_network']["linear_1"] = jnp.transpose(
            jnp.reshape(params_binary[:64 * 64], (64, 64)))
        params_binary = params_binary[64 * 64:]

        params['ngp_network/rgb_network']["linear_2"] = jnp.transpose(
            jnp.reshape(params_binary[:3 * 64], (3, 64)))
        params_binary = params_binary[16 * 64:]

        params["ngp_network/hash_encoding"] = {}
        params["ngp_network/hash_encoding"]["grid"] = jnp.reshape(
            params_binary, (int(params_binary.size / 2), 2))

        params['ngp_network/density_network']["lsq_0"] = jnp.array([1.])
        params['ngp_network/density_network']["lsq_1"] = jnp.array([1.])
        params['ngp_network/rgb_network']["lsq_0"] = jnp.array([1.])
        params['ngp_network/rgb_network']["lsq_1"] = jnp.array([1.])
        params['ngp_network/rgb_network']["lsq_2"] = jnp.array([1.])
        params["ngp_network/hash_encoding"]["lsq"] = jnp.array([1.])

        result["params"] = params

        return result

    def save_snapshot(self, path):
        with open(path, 'rb') as f:
            unpacker = msgpack.Unpacker(f, raw=False)
            config = next(unpacker)
        config["quantization"] = {}
        config["quantization"]["density"] = {}
        config["quantization"]["rgb"] = {}
        config["quantization"]["hash"] = {}

        result = []
        temp = lsq(self.params['ngp_network/density_network']["linear_0"],
                   self.params['ngp_network/density_network']["lsq_0"], 8)
        scale = roundscale(self.params['ngp_network/density_network']["lsq_0"],
                           8)
        config["quantization"]["density"]["lsq_0"] = float(scale)
        print(f"linear_0 {scale}")
        result.append(jnp.ravel(jnp.transpose(temp)))

        temp = lsq(self.params['ngp_network/density_network']["linear_1"],
                   self.params['ngp_network/density_network']["lsq_1"], 8)
        scale = roundscale(self.params['ngp_network/density_network']["lsq_1"],
                           8)
        config["quantization"]["density"]["lsq_1"] = float(scale)
        print(f"linear_1 {scale}")
        result.append(jnp.ravel(jnp.transpose(temp)))

        temp = lsq(self.params['ngp_network/rgb_network']["linear_0"],
                   self.params['ngp_network/rgb_network']["lsq_0"], 8)
        scale = roundscale(self.params['ngp_network/rgb_network']["lsq_0"], 8)
        config["quantization"]["rgb"]["lsq_0"] = float(scale)
        print(f"linear_0 {scale}")
        result.append(jnp.ravel(jnp.transpose(temp)))

        temp = lsq(self.params['ngp_network/rgb_network']["linear_1"],
                   self.params['ngp_network/rgb_network']["lsq_1"], 8)
        scale = roundscale(self.params['ngp_network/rgb_network']["lsq_1"], 8)
        config["quantization"]["rgb"]["lsq_1"] = float(scale)
        print(f"linear_1 {scale}")
        result.append(jnp.ravel(jnp.transpose(temp)))

        temp = lsq(self.params['ngp_network/rgb_network']["linear_2"],
                   self.params['ngp_network/rgb_network']["lsq_2"], 8)
        scale = roundscale(self.params['ngp_network/rgb_network']["lsq_2"], 8)
        config["quantization"]["rgb"]["lsq_2"] = float(scale)
        print(f"linear_2 {scale}")
        temp = jnp.pad(temp, ((0, 0), (0, 13)))
        result.append(jnp.ravel(jnp.transpose(temp)))

        temp = lsq(self.params["ngp_network/hash_encoding"]["grid"],
                   self.params["ngp_network/hash_encoding"]["lsq"], 4)
        scale = roundscale(self.params["ngp_network/hash_encoding"]["lsq"], 4)
        config["quantization"]["hash"]["lsq"] = float(scale)
        print(f"hash {scale}")
        result.append(jnp.ravel(temp))

        result = jnp.concatenate(result)

        config["snapshot"]["params_binary"] = jnp.float16(result).tobytes()
        config['encoding']['log2_hashmap_size'] = self.config["Hash"][
            "log2_hashmap_size"]
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

        network_output = self.model(self.params, coords)

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

        gradient = self.grad(self.params, coords_compated, dloss_doutput)

        loss = jnp.sum(loss) / compacted_ray_counter

        return loss, gradient
