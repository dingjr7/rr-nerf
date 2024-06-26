#include <Eigen/Core>
#include <bounding_box.h>
#include <cmath>
#include <common.h>
#include <cstddef>
#include <cstdint>
#include <ngp.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/reduce_sum.h>
#include <vector>
namespace py = pybind11;

using namespace Eigen;

namespace jax_nerf {
__global__ void grid_to_bitfield(const uint32_t n_elements,
                                 const uint32_t n_nonzero_elements,
                                 const float *__restrict__ grid,
                                 uint8_t *__restrict__ grid_bitfield,
                                 const float *__restrict__ mean_density_ptr) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements)
        return;
    if (i >= n_nonzero_elements) {
        grid_bitfield[i] = 0;
        return;
    }

    uint8_t bits = 0;

    float thresh = std::min(NERF_MIN_OPTICAL_THICKNESS(), *mean_density_ptr);

#pragma unroll
    for (uint8_t j = 0; j < 8; ++j) {
        bits |= grid[i * 8 + j] > thresh ? ((uint8_t)1 << j) : 0;
    }

    grid_bitfield[i] = bits;
}

__global__ void bitfield_max_pool(const uint32_t n_elements,
                                  const uint8_t *__restrict__ prev_level,
                                  uint8_t *__restrict__ next_level) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements)
        return;

    uint8_t bits = 0;

#pragma unroll
    for (uint8_t j = 0; j < 8; ++j) {
        // If any bit is set in the previous level, set this
        // level's bit. (Max pooling.)
        bits |= prev_level[i * 8 + j] > 0 ? ((uint8_t)1 << j) : 0;
    }

    uint32_t x = tcnn::morton3D_invert(i >> 0) + NERF_GRIDSIZE() / 8;
    uint32_t y = tcnn::morton3D_invert(i >> 1) + NERF_GRIDSIZE() / 8;
    uint32_t z = tcnn::morton3D_invert(i >> 2) + NERF_GRIDSIZE() / 8;

    next_level[tcnn::morton3D(x, y, z)] |= bits;
}

uint8_t *get_density_grid_bitfield_mip(uint8_t *density_grid_bitfield,
                                       uint32_t mip) {
    return density_grid_bitfield + grid_mip_offset(mip) / 8;
}

void update_density_grid_mean_and_bitfield(
    py::capsule density_grid_cap, py::capsule density_grid_bitfield_cap,
    uint32_t n_level) {
    cudaStream_t stream = nullptr;
    const uint32_t n_elements =
        NERF_GRIDSIZE() * NERF_GRIDSIZE() * NERF_GRIDSIZE();

    size_t size_including_mips = grid_mip_offset(NERF_CASCADES()) / 8;

    tcnn::GPUMemoryArena::Allocation alloc;

    auto scratch = tcnn::allocate_workspace_and_distribute<float>(
        stream, &alloc, tcnn::reduce_sum_workspace_size(n_elements));

    float *density_grid_mean = std::get<0>(scratch);
    float *density_grid = dltensor_data<float>(density_grid_cap);
    uint8_t *density_grid_bitfield =
        dltensor_data<uint8_t>(density_grid_bitfield_cap);

    CUDA_CHECK_THROW(
        cudaMemsetAsync(density_grid_mean, 0, sizeof(float), stream));
    tcnn::reduce_sum(
        density_grid,
        [n_elements] __device__(float val) {
            return fmaxf(val, 0.f) / (n_elements);
        },
        density_grid_mean, n_elements, stream);

    tcnn::linear_kernel(grid_to_bitfield, 0, stream,
                        n_elements / 8 * NERF_CASCADES(),
                        n_elements / 8 * n_level, density_grid,
                        density_grid_bitfield, density_grid_mean);

    for (uint32_t level = 1; level < NERF_CASCADES(); ++level) {
        tcnn::linear_kernel(
            bitfield_max_pool, 0, stream, n_elements / 64,
            get_density_grid_bitfield_mip(density_grid_bitfield, level - 1),
            get_density_grid_bitfield_mip(density_grid_bitfield, level));
    }

    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
}

inline __host__ __device__ float calc_dt(float t) {
    return tcnn::clamp(t * CONE_ANGLE(), MIN_CONE_STEPSIZE(),
                       MAX_CONE_STEPSIZE());
}

__device__ uint32_t cascaded_grid_idx_at(Vector3f pos, uint32_t mip) {
    float mip_scale = scalbnf(1.0f, -mip);
    pos -= Vector3f::Constant(0.5f);
    pos *= mip_scale;
    pos += Vector3f::Constant(0.5f);

    Vector3i i = (pos * NERF_GRIDSIZE()).cast<int>();

    if (i.x() < -1 || i.x() > NERF_GRIDSIZE() || i.y() < -1 ||
        i.y() > NERF_GRIDSIZE() || i.z() < -1 || i.z() > NERF_GRIDSIZE()) {
        printf("WTF %d %d %d\n", i.x(), i.y(), i.z());
    }

    uint32_t idx =
        tcnn::morton3D(tcnn::clamp(i.x(), 0, (int)NERF_GRIDSIZE() - 1),
                       tcnn::clamp(i.y(), 0, (int)NERF_GRIDSIZE() - 1),
                       tcnn::clamp(i.z(), 0, (int)NERF_GRIDSIZE() - 1));

    return idx;
}

__device__ bool density_grid_occupied_at(const Vector3f &pos,
                                         const uint8_t *density_grid_bitfield,
                                         uint32_t mip) {
    uint32_t idx = cascaded_grid_idx_at(pos, mip);
    return density_grid_bitfield[idx / 8 + grid_mip_offset(mip) / 8] &
           (1 << (idx % 8));
}

inline __device__ int mip_from_pos(const Vector3f &pos,
                                   uint32_t max_cascade = NERF_CASCADES() - 1) {
    int exponent;
    float maxval = (pos - Vector3f::Constant(0.5f)).cwiseAbs().maxCoeff();
    frexpf(maxval, &exponent);
    return min(max_cascade, max(0, exponent + 1));
}

inline __device__ int mip_from_dt(float dt, const Vector3f &pos,
                                  uint32_t max_cascade = NERF_CASCADES() - 1) {
    int mip = mip_from_pos(pos, max_cascade);
    dt *= 2 * NERF_GRIDSIZE();
    if (dt < 1.f)
        return mip;
    int exponent;
    frexpf(dt, &exponent);
    return min(max_cascade, max(exponent, mip));
}

inline __device__ float distance_to_next_voxel(const Vector3f &pos,
                                               const Vector3f &dir,
                                               const Vector3f &idir,
                                               uint32_t res) { // dda like step
    Vector3f p = res * pos;
    float tx = (floorf(p.x() + 0.5f + 0.5f * copysignf(1.0, dir.x())) - p.x()) *
               idir.x();
    float ty = (floorf(p.y() + 0.5f + 0.5f * copysignf(1.0, dir.y())) - p.y()) *
               idir.y();
    float tz = (floorf(p.z() + 0.5f + 0.5f * copysignf(1.0, dir.z())) - p.z()) *
               idir.z();
    float t = min(min(tx, ty), tz);

    return fmaxf(t / res, 0.0f);
}

inline __device__ float advance_to_next_voxel(float t, const Vector3f &pos,
                                              const Vector3f &dir,
                                              const Vector3f &idir,
                                              uint32_t res) {
    // Analytic stepping by a multiple of dt. Make empty space unequal to
    // non-empty space due to the different stepping. float dt = calc_dt(t,
    // cone_angle); return t + ceilf(fmaxf(distance_to_next_voxel(pos, dir,
    // idir, res) / dt, 0.5f)) * dt;

    // Regular stepping (may be slower but matches non-empty space)
    float t_target = t + distance_to_next_voxel(pos, dir, idir, res);
    do {
        t += calc_dt(t);
    } while (t < t_target);
    return t;
}

__host__ __device__ Vector3f warp_direction(const Vector3f &dir) {
    return (dir + Vector3f::Ones()) * 0.5f;
}

__device__ float warp_dt(float dt) {
    float max_stepsize = MIN_CONE_STEPSIZE() * (1 << (NERF_CASCADES() - 1));
    return (dt - MIN_CONE_STEPSIZE()) / (max_stepsize - MIN_CONE_STEPSIZE());
}

__device__ Vector3f unwarp_direction(const Vector3f &dir) {
    return dir * 2.0f - Vector3f::Ones();
}

__device__ float unwarp_dt(float dt) {
    float max_stepsize = MIN_CONE_STEPSIZE() * (1 << (NERF_CASCADES() - 1));
    return dt * (max_stepsize - MIN_CONE_STEPSIZE()) + MIN_CONE_STEPSIZE();
}

__device__ Vector3f warp_position(const Vector3f &pos,
                                  const BoundingBox &aabb) {
    // return {tcnn::logistic(pos.x() - 0.5f), tcnn::logistic(pos.y() - 0.5f),
    // tcnn::logistic(pos.z() - 0.5f)}; return pos;

    return aabb.relative_pos(pos);
}

__device__ Vector3f unwarp_position(const Vector3f &pos,
                                    const BoundingBox &aabb) {
    // return {logit(pos.x()) + 0.5f, logit(pos.y()) + 0.5f, logit(pos.z()) +
    // 0.5f}; return pos;

    return aabb.min + pos.cwiseProduct(aabb.diag());
}

__global__ void generate_training_sample_kernel(
    const uint32_t n_rays, const uint32_t max_samples, const BoundingBox aabb,
    const Vector3f *__restrict__ ray, const uint8_t *__restrict__ density_grid,
    uint32_t *__restrict__ numsteps_counter, uint32_t *__restrict__ ray_counter,
    uint32_t *__restrict__ numsteps_out, uint32_t *__restrict__ ray_indices_out,
    NerfCoordinate *__restrict__ coords_out) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_rays)
        return;

    ray += i * 2;

    Vector3f ray_o = ray[0];
    Vector3f ray_d = ray[1];

    Vector2f tminmax = aabb.ray_intersect(ray_o, ray_d);

    // The near distance prevents learning of camera-specific fudge right in
    // front of the camera
    tminmax.x() = fmaxf(tminmax.x(), 0.0f);

    float startt = tminmax.x();
    startt += calc_dt(startt) * 0.1; // prevent numerical issue
    Vector3f idir = ray_d.cwiseInverse();

    // first pass to compute an accurate number of steps
    uint32_t j = 0;
    float t = startt;
    Vector3f pos;

    while (aabb.contains(pos = ray_o + t * ray_d) && j < NERF_STEPS()) {
        float dt = calc_dt(t);
        uint32_t mip = mip_from_dt(dt, pos);
        if (density_grid_occupied_at(pos, density_grid, mip)) {
            ++j;
            t += dt;
        } else {
            uint32_t res = NERF_GRIDSIZE() >> mip;
            t = advance_to_next_voxel(t, pos, ray_d, idir, res);
        }
    }

    if (j == 0) {
        return;
    }

    uint32_t numsteps = j;
    uint32_t base = atomicAdd(
        numsteps_counter, numsteps); // first entry in the array is a counter
    if (base + numsteps > max_samples) {
        return;
    }

    uint32_t ray_idx = atomicAdd(ray_counter, 1);

    coords_out += base;

    ray_indices_out[ray_idx] = i;
    numsteps_out[ray_idx * 2 + 0] = numsteps;
    numsteps_out[ray_idx * 2 + 1] = base;

    Vector3f warped_dir = warp_direction(ray_d);
    t = startt;
    j = 0;
    while (aabb.contains(pos = ray_o + t * ray_d) && j < numsteps) {
        float dt = calc_dt(t);
        uint32_t mip = mip_from_dt(dt, pos);
        if (density_grid_occupied_at(pos, density_grid, mip)) {
            coords_out[j] = {warp_position(pos, aabb), warped_dir, warp_dt(dt)};
            ++j;
            t += dt;
        } else {
            uint32_t res = NERF_GRIDSIZE() >> mip;
            t = advance_to_next_voxel(t, pos, ray_d, idir, res);
        }
    }
}

// TODO: add ray counter and cone constant
uint32_t ngp_generate_training_sample(
    py::capsule ray_jax, py::capsule density_grid_jax, py::capsule numsteps_jax,
    py::capsule ray_indices_jax, py::capsule coords_jax,
    std::vector<Vector3f> aabb_np, uint32_t n_rays, uint32_t max_samples) {
    cudaStream_t stream = nullptr;

    uint32_t *counter;
    CUDA_CHECK_THROW(cudaMallocAsync(&counter, sizeof(uint32_t) * 2, stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(counter, 0, sizeof(uint32_t) * 2, stream));

    Vector3f *ray = dltensor_data<Vector3f>(ray_jax);
    uint8_t *density_grid = dltensor_data<uint8_t>(density_grid_jax);
    uint32_t *numsteps_out = dltensor_data<uint32_t>(numsteps_jax);
    uint32_t *ray_indices_out = dltensor_data<uint32_t>(ray_indices_jax);
    NerfCoordinate *coords_out = dltensor_data<NerfCoordinate>(coords_jax);

    assert(aabb_np.size() == 2);
    BoundingBox aabb = {aabb_np[0], aabb_np[1]};

    tcnn::linear_kernel(generate_training_sample_kernel, 0, stream, n_rays,
                        max_samples, aabb, ray, density_grid, counter,
                        counter + 1, numsteps_out, ray_indices_out, coords_out);

    uint32_t ray_counter;
    CUDA_CHECK_THROW(cudaMemcpyAsync(&ray_counter, counter + 1,
                                     sizeof(uint32_t), cudaMemcpyDeviceToHost,
                                     stream));
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    return ray_counter;
}

__device__ float network_to_rgb(float val, ENerfActivation activation) {
    switch (activation) {
    case ENerfActivation::None:
        return val;
    case ENerfActivation::ReLU:
        return val > 0.0f ? val : 0.0f;
    case ENerfActivation::Logistic:
        return tcnn::logistic(val);
    case ENerfActivation::Exponential:
        return __expf(tcnn::clamp(val, -10.0f, 10.0f));
    default:
        assert(false);
    }
    return 0.0f;
}

__device__ Array3f network_to_rgb(const Array4f &local_network_output,
                                  ENerfActivation activation) {
    return {network_to_rgb(float(local_network_output[0]), activation),
            network_to_rgb(float(local_network_output[1]), activation),
            network_to_rgb(float(local_network_output[2]), activation)};
}

__device__ float network_to_density(float val, ENerfActivation activation) {
    switch (activation) {
    case ENerfActivation::None:
        return val;
    case ENerfActivation::ReLU:
        return val > 0.0f ? val : 0.0f;
    case ENerfActivation::Logistic:
        return tcnn::logistic(val);
    case ENerfActivation::Exponential:
        return __expf(val);
    default:
        assert(false);
    }
    return 0.0f;
}

inline __device__ Array3f copysign(const Array3f &a, const Array3f &b) {
    return {
        copysignf(a.x(), b.x()),
        copysignf(a.y(), b.y()),
        copysignf(a.z(), b.z()),
    };
}

inline __device__ LossAndGradient l2_loss(const Array3f &target,
                                          const Array3f &prediction) {
    Array3f difference = prediction - target;
    return {difference * difference, 2.0f * difference};
}

inline __device__ LossAndGradient relative_l2_loss(const Array3f &target,
                                                   const Array3f &prediction) {
    Array3f difference = prediction - target;
    Array3f factor =
        (prediction * prediction + Array3f::Constant(1e-2f)).inverse();
    return {difference * difference * factor, 2.0f * difference * factor};
}

inline __device__ LossAndGradient l1_loss(const Array3f &target,
                                          const Array3f &prediction) {
    Array3f difference = prediction - target;
    return {
        difference.abs(),
        copysign(Array3f::Ones(), difference),
    };
}

inline __device__ LossAndGradient huber_loss(const Array3f &target,
                                             const Array3f &prediction,
                                             float alpha = 1) {
    Array3f difference = prediction - target;
    Array3f abs_diff = difference.abs();
    Array3f square = 0.5f / alpha * difference * difference;
    return {
        {
            abs_diff.x() > alpha ? (abs_diff.x() - 0.5f * alpha) : square.x(),
            abs_diff.y() > alpha ? (abs_diff.y() - 0.5f * alpha) : square.y(),
            abs_diff.z() > alpha ? (abs_diff.z() - 0.5f * alpha) : square.z(),
        },
        {
            abs_diff.x() > alpha ? (difference.x() > 0 ? 1.0f : -1.0f)
                                 : (difference.x() / alpha),
            abs_diff.y() > alpha ? (difference.y() > 0 ? 1.0f : -1.0f)
                                 : (difference.y() / alpha),
            abs_diff.z() > alpha ? (difference.z() > 0 ? 1.0f : -1.0f)
                                 : (difference.z() / alpha),
        },
    };
}

inline __device__ LossAndGradient log_l1_loss(const Array3f &target,
                                              const Array3f &prediction) {
    Array3f difference = prediction - target;
    Array3f divisor = difference.abs() + Array3f::Ones();
    return {
        divisor.log(),
        copysign(divisor.inverse(), difference),
    };
}

inline __device__ LossAndGradient smape_loss(const Array3f &target,
                                             const Array3f &prediction) {
    Array3f difference = prediction - target;
    Array3f factor =
        (0.5f * (prediction.abs() + target.abs()) + Array3f::Constant(1e-2f))
            .inverse();
    return {
        difference.abs() * factor,
        copysign(factor, difference),
    };
}

inline __device__ LossAndGradient mape_loss(const Array3f &target,
                                            const Array3f &prediction) {
    Array3f difference = prediction - target;
    Array3f factor = (prediction.abs() + Array3f::Constant(1e-2f)).inverse();
    return {
        difference.abs() * factor,
        copysign(factor, difference),
    };
}

__device__ LossAndGradient loss_and_gradient(const Vector3f &target,
                                             const Vector3f &prediction,
                                             ELossType loss_type) {
    switch (loss_type) {
    case ELossType::RelativeL2:
        return relative_l2_loss(target, prediction);
        break;
    case ELossType::L1:
        return l1_loss(target, prediction);
        break;
    case ELossType::Mape:
        return mape_loss(target, prediction);
        break;
    case ELossType::Smape:
        return smape_loss(target, prediction);
        break;
    // Note: we divide the huber loss by a factor of 5 such that its L2 region
    // near zero matches with the L2 loss and error numbers become more
    // comparable. This allows reading off dB numbers of ~converged models and
    // treating them as approximate PSNR to compare with other NeRF methods.
    // Self-normalizing optimizers such as Adam are agnostic to such constant
    // factors; optimization is therefore unaffected.
    case ELossType::Huber:
        return huber_loss(target, prediction, 0.1f) / 5.0f;
        break;
    case ELossType::LogL1:
        return log_l1_loss(target, prediction);
        break;
    default:
    case ELossType::L2:
        return l2_loss(target, prediction);
        break;
    }
}

__device__ float network_to_rgb_derivative(float val,
                                           ENerfActivation activation) {
    switch (activation) {
    case ENerfActivation::None:
        return 1.0f;
    case ENerfActivation::ReLU:
        return val > 0.0f ? 1.0f : 0.0f;
    case ENerfActivation::Logistic: {
        float density = tcnn::logistic(val);
        return density * (1 - density);
    };
    case ENerfActivation::Exponential:
        return __expf(tcnn::clamp(val, -10.0f, 10.0f));
    default:
        assert(false);
    }
    return 0.0f;
}

__device__ float network_to_density_derivative(float val,
                                               ENerfActivation activation) {
    switch (activation) {
    case ENerfActivation::None:
        return 1.0f;
    case ENerfActivation::ReLU:
        return val > 0.0f ? 1.0f : 0.0f;
    case ENerfActivation::Logistic: {
        float density = tcnn::logistic(val);
        return density * (1 - density);
    };
    case ENerfActivation::Exponential:
        return __expf(tcnn::clamp(val, -15.0f, 15.0f));
    default:
        assert(false);
    }
    return 0.0f;
}

inline __device__ float srgb_to_linear(float srgb) {
    if (srgb <= 0.04045f) {
        return srgb / 12.92f;
    } else {
        return std::pow((srgb + 0.055f) / 1.055f, 2.4f);
    }
}

inline __device__ float linear_to_srgb(float linear) {
    if (linear < 0.0031308f) {
        return 12.92f * linear;
    } else {
        return 1.055f * std::pow(linear, 0.41666f) - 0.055f;
    }
}

inline __device__ Eigen::Array3f linear_to_srgb(const Eigen::Array3f &x) {
    return {linear_to_srgb(x.x()), linear_to_srgb(x.y()),
            (linear_to_srgb(x.z()))};
}

__global__ void ngp_compute_loss_kernel(
    const uint32_t n_rays, const uint32_t n_rays_total,
    const uint32_t max_samples_compacted, const ENerfActivation rgb_activation,
    const ENerfActivation density_activation, const ELossType loss_type,
    uint32_t *__restrict__ numsteps_in,
    const uint32_t *__restrict__ ray_indices_in,
    const NerfCoordinate *__restrict__ coords_in,
    const Array<uint8_t, 4, 1> *__restrict__ pixels,
    const Array<float, 3, 1> *__restrict__ bgs,
    const Array4f *__restrict__ network_output, float *__restrict__ loss_output,
    Array4f *__restrict__ dloss_doutput,
    uint32_t *__restrict__ numsteps_counter, uint32_t *__restrict__ ray_counter,
    NerfCoordinate *__restrict__ coords_out) {

    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= n_rays_total)
        return;

    // grab the number of samples for this ray, and the first sample
    uint32_t numsteps = numsteps_in[i * 2 + 0];
    uint32_t base = numsteps_in[i * 2 + 1];

    coords_in += base;
    network_output += base; // for jax, no need to pad

    float T = 1.f;

    float EPSILON = 1e-4f;

    Array3f rgb_ray = Array3f::Zero();

    uint32_t compacted_numsteps = 0;
    for (; compacted_numsteps < numsteps; ++compacted_numsteps) {
        if (T < EPSILON) {
            break;
        }

        const Array4f local_network_output = network_output[compacted_numsteps];
        const Array3f rgb =
            network_to_rgb(local_network_output, rgb_activation);
        const float dt = unwarp_dt(coords_in[compacted_numsteps].dt);
        float density = network_to_density(float(local_network_output[3]),
                                           density_activation);

        const float alpha = 1.f - __expf(-density * dt);
        const float weight = alpha * T;
        rgb_ray += weight * rgb;
        T *= (1.f - alpha);
    }

    uint32_t ray_idx = ray_indices_in[i];

    // the alpha channel is always in linear mode
    Array<uint8_t, 4, 1> pixel = pixels[ray_idx];
    Array<float, 3, 1> background_color = bgs[ray_idx];
    float alpha = (float)pixel[3] * (1.0f / 255.0f);
    Eigen::Array3f rgb_target_linear(
        srgb_to_linear((float)pixel[0] * (1.0f / 255.0f)) * alpha,
        srgb_to_linear((float)pixel[1] * (1.0f / 255.0f)) * alpha,
        srgb_to_linear((float)pixel[2] * (1.0f / 255.0f)) * alpha);
    Array3f rgb_target =
        linear_to_srgb(rgb_target_linear) + (1.0f - alpha) * background_color;

    if (compacted_numsteps == numsteps) {
        // support arbitrary background colors
        rgb_ray += T * background_color;
    }

    LossAndGradient lg = loss_and_gradient(rgb_target, rgb_ray, loss_type);
    uint32_t compacted_base =
        atomicAdd(numsteps_counter,
                  compacted_numsteps); // first entry in the array is a counter
    compacted_numsteps =
        min(max_samples_compacted - min(max_samples_compacted, compacted_base),
            compacted_numsteps);
    numsteps_in[i * 2 + 0] = compacted_numsteps;
    numsteps_in[i * 2 + 1] = compacted_base;
    if (compacted_numsteps == 0) {
        return;
    }

    dloss_doutput += compacted_base;
    coords_out += compacted_base;

    loss_output[i] = lg.loss.mean();
    atomicAdd(ray_counter, 1);

    float loss_scale = LOSS_SCALE / n_rays;
    const float output_l2_reg =
        rgb_activation == ENerfActivation::Exponential ? 1e-4f : 0.0f;

    Array3f rgb_ray2 = Array3f::Zero();
    T = 1.f;
    for (uint32_t j = 0; j < compacted_numsteps; ++j) {
        const Array4f local_network_output = network_output[j];
        const Array3f rgb =
            network_to_rgb(local_network_output, rgb_activation);
        const float dt = unwarp_dt(coords_in[j].dt);
        float density = network_to_density(float(local_network_output[3]),
                                           density_activation);

        const float alpha = 1.f - __expf(-density * dt);
        const float weight = alpha * T;
        rgb_ray2 += weight * rgb;
        T *= (1.f - alpha);

        // we know the suffix of this ray compared to where we are up to. note
        // the suffix depends on this step's alpha as suffix =
        // (1-alpha)*(somecolor), so dsuffix/dalpha = -somecolor =
        // -suffix/(1-alpha)
        const Array3f suffix = rgb_ray - rgb_ray2;
        const Array3f dloss_by_drgb = weight * lg.gradient;

        Array4f local_dL_doutput;

        // chain rule to go from dloss/drgb to dloss/dmlp_output
        local_dL_doutput[0] =
            loss_scale *
            (dloss_by_drgb.x() * network_to_rgb_derivative(
                                     local_network_output[0], rgb_activation) +
             fmaxf(0.0f, output_l2_reg * (float)local_network_output[0]));
        local_dL_doutput[1] =
            loss_scale *
            (dloss_by_drgb.y() * network_to_rgb_derivative(
                                     local_network_output[1], rgb_activation) +
             fmaxf(0.0f, output_l2_reg * (float)local_network_output[1]));
        local_dL_doutput[2] =
            loss_scale *
            (dloss_by_drgb.z() * network_to_rgb_derivative(
                                     local_network_output[2], rgb_activation) +
             fmaxf(0.0f, output_l2_reg * (float)local_network_output[2]));

        float density_derivative = network_to_density_derivative(
            float(local_network_output[3]), density_activation);

        float dloss_by_dmlp =
            density_derivative *
            (dt * (lg.gradient.matrix().dot((T * rgb - suffix).matrix())));

        local_dL_doutput[3] = loss_scale * dloss_by_dmlp;

        dloss_doutput[j] = local_dL_doutput;
        coords_out[j] = coords_in[j];
    }
    // assert(rgb_ray.isApprox(rgb_ray2));
}
py::tuple ngp_compute_loss(py::capsule numsteps_jax,
                           py::capsule ray_indices_jax, py::capsule coords_jax,
                           py::capsule pixel_jax, py::capsule bg_jax,
                           py::capsule network_output_jax, py::capsule loss_jax,
                           py::capsule dloss_doutput_jax,
                           py::capsule compacted_coords_jax,
                           const uint32_t n_rays, const uint32_t n_rays_total,
                           const uint32_t max_samples_compacted) {
    uint32_t *numsteps_in = dltensor_data<uint32_t>(numsteps_jax);
    uint32_t *ray_indices_in = dltensor_data<uint32_t>(ray_indices_jax);
    NerfCoordinate *coords_in = dltensor_data<NerfCoordinate>(coords_jax);
    Array<uint8_t, 4, 1> *pixel =
        dltensor_data<Array<uint8_t, 4, 1>>(pixel_jax);
    Array<float, 3, 1> *bg = dltensor_data<Array<float, 3, 1>>(bg_jax);
    Array4f *network_output = dltensor_data<Array4f>(network_output_jax);
    float *loss_output = dltensor_data<float>(loss_jax);
    Array4f *dloss_doutput = dltensor_data<Array4f>(dloss_doutput_jax);
    NerfCoordinate *coords_out =
        dltensor_data<NerfCoordinate>(compacted_coords_jax);

    cudaStream_t stream = nullptr;

    uint32_t *counter;
    CUDA_CHECK_THROW(cudaMallocAsync(&counter, sizeof(uint32_t) * 2, stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(counter, 0, sizeof(uint32_t) * 2, stream));

    tcnn::linear_kernel(ngp_compute_loss_kernel, 0, stream, n_rays,
                        n_rays_total, max_samples_compacted,
                        ENerfActivation::Logistic, ENerfActivation::Exponential,
                        ELossType::Huber, numsteps_in, ray_indices_in,
                        coords_in, pixel, bg, network_output, loss_output,
                        dloss_doutput, counter, counter + 1, coords_out);

    uint32_t padded_output_width = 4;
    uint32_t floats_per_coord = 7;

    tcnn::fill_rollover_and_rescale<float>
        <<<tcnn::n_blocks_linear(max_samples_compacted * padded_output_width),
           tcnn::n_threads_linear, 0, stream>>>(max_samples_compacted,
                                                padded_output_width, counter,
                                                (float *)dloss_doutput);
    tcnn::fill_rollover<float>
        <<<tcnn::n_blocks_linear(max_samples_compacted * floats_per_coord),
           tcnn::n_threads_linear, 0, stream>>>(max_samples_compacted,
                                                floats_per_coord, counter,
                                                (float *)coords_out);

    uint32_t ray_counter[2];
    CUDA_CHECK_THROW(cudaMemcpyAsync(ray_counter, counter, sizeof(uint32_t) * 2,
                                     cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

    return py::make_tuple(ray_counter[0], ray_counter[1]);
}

__global__ void advance_pos_nerf(const uint32_t n_elements,
                                 BoundingBox render_aabb,
                                 const Vector3f *__restrict__ ray,
                                 const uint8_t *__restrict__ density_grid,
                                 bool *__restrict__ alives,
                                 float *__restrict__ ts,
                                 NerfCoordinate *__restrict__ coords_out) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements)
        return;

    if (!alives[i]) {
        return;
    }

    ray += i * 2;

    Vector3f origin = ray[0];
    Vector3f dir = ray[1];
    Vector3f idir = dir.cwiseInverse();

    float t = ts[i];
    Vector3f pos;

    if (t == 0) {
        Vector2f tminmax = render_aabb.ray_intersect(origin, dir);

        // The near distance prevents learning of camera-specific fudge right in
        // front of the camera
        tminmax.x() = fmaxf(tminmax.x(), 0.0f);

        float startt = tminmax.x();
        startt += calc_dt(startt) * 0.1; // prevent numerical issue
        t = startt;
    }

    while (1) {
        pos = origin + dir * t;
        if (!render_aabb.contains(pos)) {
            alives[i] = false;
            break;
        }

        float dt = calc_dt(t);

        uint32_t mip = mip_from_dt(dt, pos);

        if (density_grid_occupied_at(pos, density_grid, mip)) {
            coords_out[i] = {warp_position(pos, render_aabb),
                             warp_direction(dir), warp_dt(dt)};
            t += dt;
            break;
        }

        uint32_t res = NERF_GRIDSIZE() >> mip;
        t = advance_to_next_voxel(t, pos, dir, idir, res);
    }

    ts[i] = t;
}

void ngp_advance_pos_nerf(py::capsule ray_jax, py::capsule density_grid_jax,
                          py::capsule alive_jax, py::capsule t_jax,
                          py::capsule coords_jax,
                          const std::vector<Vector3f> aabb_np,
                          const uint32_t n_rays) {

    cudaStream_t stream = nullptr;

    const Vector3f *ray = dltensor_data<Vector3f>(ray_jax);
    const uint8_t *density_grid = dltensor_data<uint8_t>(density_grid_jax);
    bool *alive = dltensor_data<bool>(alive_jax);
    float *ts = dltensor_data<float>(t_jax);
    NerfCoordinate *coords_out = dltensor_data<NerfCoordinate>(coords_jax);

    assert(aabb_np.size() == 2);
    BoundingBox aabb = {aabb_np[0], aabb_np[1]};

    tcnn::linear_kernel(advance_pos_nerf, 0, stream, n_rays, aabb, ray,
                        density_grid, alive, ts, coords_out);

    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
}
} // namespace jax_nerf