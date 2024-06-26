from jax import dlpack, lax, numpy as jnp
import jax
import toml
from src.common.constant import STEPSIZE
from src.common.dataset import NerfSynthetic
from src.model import NGPTrainer
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import haiku
import msgpack, Morton3D

name = "lego"
res = 512

stride = jnp.linspace(0, 1, res)
z, y, x = jnp.meshgrid(stride, stride, stride, indexing='ij')
sample = jnp.asarray([jnp.ravel(x), jnp.ravel(y), jnp.ravel(z)])
sample = jnp.swapaxes(sample, 0, 1)
sample = jnp.pad(sample, ((0, 0), (0, 4)))

dataset = NerfSynthetic(f"../baseline/data/nerf_synthetic/{name}")
snapshot = NGPTrainer.load_msgpack(f"baseline/{name}.msgpack")
config = toml.load("configs/ngp/base.toml")
trainer = NGPTrainer(
    {
        "config": config,
        "density_grid": snapshot["density_grid"],
        "params": snapshot["params"]
    }, dataset)

batch = int(res**3 / 128)
density = []

for i in range(0, res**3, batch):
    rgba = trainer.model(trainer.params, sample[i:i + batch])
    density.append(jnp.exp(jnp.swapaxes(rgba, 0, 1)[3]) * STEPSIZE())

density = jnp.resize(jnp.ravel(jnp.array(density)), (res, res, res))
density = np.array(density)

density = ndimage.gaussian_filter(density, sigma=0.6)
density[np.abs(density) < 0.05] = 0

density_grid = haiku.max_pool(jnp.array(density),
                              (int(res / 128), int(res / 128), int(res / 128)),
                              (int(res / 128), int(res / 128), int(res / 128)),
                              "SAME")
density_grid = np.array(density_grid)
print(np.count_nonzero(density_grid))
# fig = plt.figure()

# x, y, z = np.where(density_grid > 0)
# x = np.array(x)
# y = np.array(y)
# z = np.array(z)
# density_grid = density_grid[z, y, x]

# ax = fig.add_subplot(projection='3d')
# ax.set_xlim(0, 128)
# ax.set_xlabel("Z")
# ax.set_ylim(0, 128)
# ax.set_ylabel("X")
# ax.set_zlim(0, 128)
# ax.set_zlabel("Y")
# pts = ax.scatter(z, x, y, c=density_grid, cmap="rainbow", s=0.5)
# fig.colorbar(pts)
# plt.show()

with open(f"baseline/{name}.msgpack", 'rb') as f:
    unpacker = msgpack.Unpacker(f, raw=False)
    config = next(unpacker)

grid_morton = np.zeros((128**3, ), dtype=np.float16)

z_order = Morton3D.zorder()

z, y, x = np.where(density_grid > 0)

for i in range(0, len(x)):
    idx = z_order.Morton(x[i], y[i], z[i])[0]
    grid_morton[idx] = density_grid[z[i], y[i], x[i]]

config['snapshot']['density_grid_binary'] = grid_morton.tobytes()

with open("a.msgpack", 'wb') as f:
    f.write(msgpack.packb(config))