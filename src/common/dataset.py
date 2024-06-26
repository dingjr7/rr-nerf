import os
import json
import imageio.v3 as imageio
from jax import random, numpy as jnp
import jax
from .aabb import AABB
import numpy as np


# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
# torch ngp
@jax.jit
def _nerf_matrix_to_ngp(pose, scale=0.33, offset=[0.5, 0.5, 0.5]):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = jnp.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
        [0, 0, 0, 1],
    ],
                         dtype=jnp.float32)
    return new_pose


class NerfSynthetic:

    #copy from nerfacc
    def __init__(self, data_dir: str, splits=["test", "val", "train"]):
        images = []
        transform_matrix = []
        for split in splits:
            with open(
                    os.path.join(data_dir, "transforms_{}.json".format(split)),
                    "r") as fp:
                meta = json.load(fp)

            self.scale = 0.33
            self.offset = [0.5, 0.5, 0.5]

            for i in range(len(meta["frames"])):
                frame = meta["frames"][i]
                fname = os.path.join(data_dir, frame["file_path"] + ".png")
                rgba = imageio.imread(fname)
                transform_matrix.append(
                    _nerf_matrix_to_ngp(jnp.array(frame["transform_matrix"]),
                                        self.scale, self.offset))
                images.append(rgba)

        self.images = jnp.array(images)

        self.transform_matrix = jnp.array(transform_matrix)[:, :3, :]

        h, w = self.images.shape[1:3]
        camera_angle_x = float(meta["camera_angle_x"])
        if "camera_angle_y" in meta.keys():
            camera_angle_y = float(meta["camera_angle_y"])
        else:
            camera_angle_y = camera_angle_x
        self.focal = jnp.array([
            0.5 * w / jnp.tan(0.5 * camera_angle_x),
            0.5 * h / jnp.tan(0.5 * camera_angle_y)
        ])
        self.principal = jnp.array([0.5, 0.5])
        self.hw = jnp.array([self.images.shape[2], self.images.shape[1]])

        if "aabb_scale" in meta.keys():
            self.aabb_scale = meta["aabb_scale"]
        else:
            self.aabb_scale = 1

        self.aabb = AABB([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        assert (self.aabb_scale < (1 << 7))
        self.aabb.inflate(0.5 * self.aabb_scale)

    @staticmethod
    def ray(transform_matrix, x, y, focal, principal, hw):
        x = (x + 0.5) / hw[0]
        y = (y + 0.5) / hw[1]
        ray_o = transform_matrix[:, 3]
        ray_d = jnp.array([
            (x - principal[0]) * hw[0] / focal[0],
            (y - principal[1]) * hw[1] / focal[1],
            1.0,
        ])
        ray_d = jnp.matmul(transform_matrix[:3, :3], ray_d)
        ray_d = ray_d / jnp.linalg.norm(ray_d)

        return jnp.array([ray_o, ray_d])

    def sample(self, key, n_rays):
        keys = random.split(key, 4)
        image_id = random.randint(keys[0], (n_rays, ), 0, self.images.shape[0])
        h_id = random.randint(keys[1], (n_rays, ), 0, self.hw[1])
        w_id = random.randint(keys[2], (n_rays, ), 0, self.hw[0])

        pixels = self.images[image_id, h_id, w_id]
        bg = random.uniform(keys[3], (n_rays, 3), dtype=jnp.float32)
        transform_matrixs = self.transform_matrix[image_id]

        ray = jax.vmap(self.ray,
                       in_axes=(0, 0, 0, None, None,
                                None))(transform_matrixs, w_id, h_id,
                                       self.focal, self.principal, self.hw)

        return (pixels, bg, ray)
