from dataclasses import dataclass
from jax import numpy as jnp
import haiku as hk


class SphericalHarmonics(hk.Module):

    def __init__(self, name=None):
        super().__init__(name)

    def __call__(self, inputs):
        x = inputs.at[0].get() * 2 - 1
        y = inputs.at[1].get() * 2 - 1
        z = inputs.at[2].get() * 2 - 1

        xy = x * y
        xz = x * z
        yz = y * z
        x2 = x * x
        y2 = y * y
        z2 = z * z

        encoded_directions = jnp.zeros((16, ), dtype=jnp.float32)

        encoded_directions = encoded_directions.at[0].set(0.28209479177387814)
        encoded_directions = encoded_directions.at[1].set(
            -0.48860251190291987 * y)
        encoded_directions = encoded_directions.at[2].set(0.48860251190291987 *
                                                          z)
        encoded_directions = encoded_directions.at[3].set(
            -0.48860251190291987 * x)
        encoded_directions = encoded_directions.at[4].set(1.0925484305920792 *
                                                          xy)
        encoded_directions = encoded_directions.at[5].set(-1.0925484305920792 *
                                                          yz)
        encoded_directions = encoded_directions.at[6].set(0.94617469575755997 *
                                                          z2 -
                                                          0.31539156525251999)
        encoded_directions = encoded_directions.at[7].set(-1.0925484305920792 *
                                                          xz)
        encoded_directions = encoded_directions.at[8].set(
            0.54627421529603959 * x2 - 0.54627421529603959 * y2)
        encoded_directions = encoded_directions.at[9].set(0.59004358992664352 *
                                                          y * (-3.0 * x2 + y2))
        encoded_directions = encoded_directions.at[10].set(2.8906114426405538 *
                                                           xy * z)
        encoded_directions = encoded_directions.at[11].set(
            0.45704579946446572 * y * (1.0 - 5.0 * z2))
        encoded_directions = encoded_directions.at[12].set(
            0.3731763325901154 * z * (5.0 * z2 - 3.0))
        encoded_directions = encoded_directions.at[13].set(
            0.45704579946446572 * x * (1.0 - 5.0 * z2))
        encoded_directions = encoded_directions.at[14].set(1.4453057213202769 *
                                                           z * (x2 - y2))
        encoded_directions = encoded_directions.at[15].set(
            0.59004358992664352 * x * (-x2 + 3.0 * y2))

        return encoded_directions
