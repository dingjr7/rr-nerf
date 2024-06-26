#pragma once

#include <pybind11/stl.h>
#include <tuple>
#include <vector>

namespace jax_nerf {
// get 8*8*8 offset table
std::tuple<std::vector<size_t>, size_t>
envr_get_offset_table(std::vector<uint8_t> density_grid);
} // namespace jax_nerf