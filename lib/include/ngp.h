#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <pybind11/pybind11.h>
#include <vector>
namespace py = pybind11;

namespace jax_nerf {

// size of the density/occupancy grid in number of cells along an axis.
inline constexpr __host__ __device__ uint32_t NERF_GRIDSIZE() { return 128; };

inline constexpr __host__ __device__ uint32_t NERF_CASCADES() { return 8; };

// Any alpha below this is considered "invisible" and is thus culled away.
inline constexpr __device__ float NERF_MIN_OPTICAL_THICKNESS() { return 0.01f; }

inline constexpr __device__ float SQRT3() { return 1.73205080757f; }

inline constexpr __device__ uint32_t NERF_STEPS() {
    return 1024;
} // finest number of steps per unit length

inline constexpr __device__ float STEPSIZE() {
    return (SQRT3() / NERF_STEPS());
} // for nerf raymarch

inline constexpr __device__ float MIN_CONE_STEPSIZE() { return STEPSIZE(); }
// Maximum step size is the width of the coarsest gridsize cell.
inline constexpr __device__ float MAX_CONE_STEPSIZE() {
    return STEPSIZE() * (1 << (NERF_CASCADES() - 1)) * NERF_STEPS() /
           NERF_GRIDSIZE();
}

inline constexpr __device__ float CONE_ANGLE() {
    return 0.f;
} // for small scene

inline __host__ __device__ uint32_t grid_mip_offset(uint32_t mip) {
    return (NERF_GRIDSIZE() * NERF_GRIDSIZE() * NERF_GRIDSIZE()) * mip;
}

void update_density_grid_mean_and_bitfield(
    py::capsule density_grid_cap, py::capsule density_grid_bitfield_cap,
    uint32_t n_level);

struct NerfCoordinate {

    Eigen::Vector3f pos;
    float dt;
    Eigen::Vector3f dir;

    __device__ NerfCoordinate(const Eigen::Vector3f &pos,
                              const Eigen::Vector3f &dir, float dt)
        : pos{pos}, dt{dt}, dir{dir} {};
};

uint32_t ngp_generate_training_sample(py::capsule ray_jax,
                                      py::capsule density_grid_jax,
                                      py::capsule numsteps_jax,
                                      py::capsule ray_indices_jax,
                                      py::capsule coords_jax,
                                      std::vector<Eigen::Vector3f> aabb_np,
                                      uint32_t n_rays, uint32_t max_samples);

enum class ENerfActivation : int {
    None,
    ReLU,
    Logistic,
    Exponential,
};

enum class ELossType : int {
    L2,
    L1,
    Mape,
    Smape,
    Huber,
    LogL1,
    RelativeL2,
};

struct LossAndGradient {
    Eigen::Array3f loss;
    Eigen::Array3f gradient;

    __host__ __device__ LossAndGradient operator*(float scalar) {
        return {loss * scalar, gradient * scalar};
    }

    __host__ __device__ LossAndGradient operator/(float scalar) {
        return {loss / scalar, gradient / scalar};
    }
};

static constexpr float LOSS_SCALE = 128.f;

py::tuple ngp_compute_loss(py::capsule numsteps_jax,
                           py::capsule ray_indices_jax, py::capsule coords_jax,
                           py::capsule pixel_jax, py::capsule bg_jax,
                           py::capsule network_output_jax, py::capsule loss_jax,
                           py::capsule dloss_doutput_jax,
                           py::capsule compacted_coords_jax,
                           const uint32_t n_rays, const uint32_t n_rays_total,
                           const uint32_t max_samples_compacted);

void ngp_advance_pos_nerf(py::capsule ray_jax, py::capsule density_grid_jax,
                          py::capsule alive_jax, py::capsule t_jax,
                          py::capsule coords_jax,
                          const std::vector<Eigen::Vector3f> aabb_np,
                          const uint32_t n_rays);

} // namespace jax_nerf