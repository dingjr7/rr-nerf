from jax import nn, numpy as jnp
import haiku as hk
from ...quantization import *


class MLPNetwork(hk.Module):

    def __init__(self, config: dict, name=None):
        super().__init__(name)
        self.n_neurons = config['n_neurons']
        self.n_hidden_layers = config['n_hidden_layers']
        self.n_output_dims = config["n_output_dims"]
        self.n_input_dims = config["n_input_dims"]

    def __call__(self, inputs):
        weight = hk.get_parameter(f"linear_0",
                                  shape=(self.n_input_dims, self.n_neurons),
                                  dtype=jnp.float32,
                                  init=hk.initializers.VarianceScaling(
                                      1.0, "fan_avg", "uniform"))
        s = hk.get_parameter(f"lsq_0",
                             shape=(1, ),
                             dtype=jnp.float32,
                             init=hk.initializers.Constant(init_lsq(weight,
                                                                    8)))
        #weight = lsq(weight, s, 8)
        x = jnp.matmul(inputs, weight)
        x = nn.relu(x)

        for i in range(0, self.n_hidden_layers - 1):
            weight = hk.get_parameter(f"linear_{i+1}",
                                      shape=(self.n_neurons, self.n_neurons),
                                      dtype=jnp.float32,
                                      init=hk.initializers.VarianceScaling(
                                          1.0, "fan_avg", "uniform"))
            s = hk.get_parameter(f"lsq_{i+1}",
                                 shape=(1, ),
                                 dtype=jnp.float32,
                                 init=hk.initializers.Constant(
                                     init_lsq(weight, 8)))
            #weight = lsq(weight, s, 8)
            x = jnp.matmul(x, weight)
            x = nn.relu(x)

        weight = hk.get_parameter(f"linear_{self.n_hidden_layers}",
                                  shape=(self.n_neurons, self.n_output_dims),
                                  dtype=jnp.float32,
                                  init=hk.initializers.VarianceScaling(
                                      1.0, "fan_avg", "uniform"))
        s = hk.get_parameter(f"lsq_{self.n_hidden_layers}",
                             shape=(1, ),
                             dtype=jnp.float32,
                             init=hk.initializers.Constant(init_lsq(weight,
                                                                    8)))
        #weight = lsq(weight, s, 8)
        x = jnp.matmul(x, weight)

        return x
