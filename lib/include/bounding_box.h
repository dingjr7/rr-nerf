// copy from instant ngp

#pragma once

#include <Eigen/Core>
#include <tiny-cuda-nn/common.h>

namespace jax_nerf {

struct BoundingBox {
    Eigen::Vector3f min, max;

    BoundingBox() {}

    BoundingBox(const Eigen::Vector3f &a, const Eigen::Vector3f &b)
        : min{a}, max{b} {}

    __device__ Eigen::Vector2f ray_intersect(const Eigen::Vector3f &pos,
                                             const Eigen::Vector3f &dir) const {
        float tmin = (min.x() - pos.x()) / dir.x();
        float tmax = (max.x() - pos.x()) / dir.x();

        if (tmin > tmax) {
            tcnn::host_device_swap(tmin, tmax);
        }

        float tymin = (min.y() - pos.y()) / dir.y();
        float tymax = (max.y() - pos.y()) / dir.y();

        if (tymin > tymax) {
            tcnn::host_device_swap(tymin, tymax);
        }

        if (tmin > tymax || tymin > tmax) {
            return {std::numeric_limits<float>::max(),
                    std::numeric_limits<float>::max()};
        }

        if (tymin > tmin) {
            tmin = tymin;
        }

        if (tymax < tmax) {
            tmax = tymax;
        }

        float tzmin = (min.z() - pos.z()) / dir.z();
        float tzmax = (max.z() - pos.z()) / dir.z();

        if (tzmin > tzmax) {
            tcnn::host_device_swap(tzmin, tzmax);
        }

        if (tmin > tzmax || tzmin > tmax) {
            return {std::numeric_limits<float>::max(),
                    std::numeric_limits<float>::max()};
        }

        if (tzmin > tmin) {
            tmin = tzmin;
        }

        if (tzmax < tmax) {
            tmax = tzmax;
        }

        return {tmin, tmax};
    }

    __device__ bool contains(const Eigen::Vector3f &p) const {
        return p.x() >= min.x() && p.x() <= max.x() && p.y() >= min.y() &&
               p.y() <= max.y() && p.z() >= min.z() && p.z() <= max.z();
    }

    __device__ Eigen::Vector3f diag() const { return max - min; }

    __device__ Eigen::Vector3f relative_pos(const Eigen::Vector3f &pos) const {
        return (pos - min).cwiseQuotient(diag());
    }
};
} // namespace jax_nerf