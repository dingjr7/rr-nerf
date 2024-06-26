from jax import lax, numpy as jnp


def NERF_CASCADES():
    return jnp.uint32(8)


def NERF_GRIDSIZE():
    return jnp.uint32(128)


def NERF_STEPS():
    return jnp.uint32(1024)


def SQRT3():
    return 1.73205080757


def STEPSIZE():
    return (SQRT3() / NERF_STEPS())


def MIN_CONE_STEPSIZE():
    return STEPSIZE()


def MAX_CONE_STEPSIZE():
    return STEPSIZE() * (
        1 << (NERF_CASCADES() - 1)) * NERF_STEPS() / NERF_GRIDSIZE()


def NERF_MIN_OPTICAL_THICKNESS():
    return 0.01