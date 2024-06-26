from jax import lax, numpy as jnp


def gradscale(x, scale):
    yOut = x
    yGrad = x * scale
    y = lax.stop_gradient(yOut - yGrad) + yGrad
    return y


def roundpass(x):
    yOut = jnp.round(x)  # Round to nearest
    yGrad = x
    y = lax.stop_gradient(yOut - yGrad) + yGrad
    return y


def roundscale(s, p):
    yOut = jnp.maximum(0.00001, s)
    yOut = jnp.round(jnp.log2(yOut))
    yOut = jnp.minimum(yOut, 0)
    yOut = jnp.exp2(yOut)

    yGrad = s
    y = lax.stop_gradient(yOut - yGrad) + yGrad
    return y


def lsq(v, s, p):
    #TODO add option to enable quantization
    Qn = -2**(p - 1)
    Qp = 2**(p - 1) - 1
    gradScaleFactor = 1 / jnp.sqrt(v.size * Qp)

    s = gradscale(s, gradScaleFactor)
    s = roundscale(s, p)
    v = v / s
    v = jnp.clip(v, Qn, Qp)
    vbar = roundpass(v)
    vhat = vbar * s
    return vhat


def init_lsq(v, p):
    v_mean = jnp.mean(v)
    v_std = jnp.std(v)
    Qp = 2**(p - 1) - 1
    return jnp.maximum(jnp.abs(v_mean - 3 * v_std),
                       jnp.abs(v_mean + 3 * v_std)) / Qp
