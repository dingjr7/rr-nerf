#include <Eigen/Dense>
#include <cstddef>
#include <envr.h>
using namespace std;
using namespace Eigen;

namespace jax_nerf {
// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
inline uint32_t expand_bits(uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
inline uint32_t morton3D(uint32_t x, uint32_t y, uint32_t z) {
    uint32_t xx = expand_bits(x);
    uint32_t yy = expand_bits(y);
    uint32_t zz = expand_bits(z);
    return xx | (yy << 1) | (zz << 2);
}

bool is_occupied_8_8(const vector<uint8_t> &density_grid, Array3i region) {
    region = region * 16;
    for (int x = 0; x < 16; x++) {
        for (int y = 0; y < 16; y++) {
            for (int z = 0; z < 16; z++) {
                uint32_t idx =
                    morton3D(x + region.x(), y + region.y(), z + region.z());
                if (density_grid[idx / 8] & (1 << (idx % 8))) {
                    return true;
                }
            }
        }
    }
    return false;
}

std::tuple<std::vector<size_t>, size_t>
envr_get_offset_table(std::vector<uint8_t> density_grid) {
    size_t offset = 0;
    std::vector<size_t> result;

    for (int z = 0; z < 8; z++) {
        for (int y = 0; y < 8; y++) {
            for (int x = 0; x < 8; x++) {
                if (is_occupied_8_8(density_grid, {x, y, z})) {
                    offset++;
                }
            }
        }
    }
    size_t max_offset = offset;
    offset = 0;
    for (int z = 0; z < 8; z++) {
        for (int y = 0; y < 8; y++) {
            for (int x = 0; x < 8; x++) {
                if (is_occupied_8_8(density_grid, {x, y, z})) {
                    result.push_back(offset);
                    offset++;
                } else {
                    result.push_back(max_offset);
                }
            }
        }
    }
    return make_tuple(result, max_offset);
}
} // namespace jax_nerf