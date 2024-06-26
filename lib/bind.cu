#include <envr.h>
#include <ngp.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

PYBIND11_MODULE(jax_nerf, m) {
    m.def("update_density_grid_mean_and_bitfield",
          &jax_nerf::update_density_grid_mean_and_bitfield);
    m.def("ngp_generate_training_sample",
          &jax_nerf::ngp_generate_training_sample);
    m.def("ngp_compute_loss", &jax_nerf::ngp_compute_loss);
    m.def("ngp_advance_pos_nerf", &jax_nerf::ngp_advance_pos_nerf);
    m.def("envr_get_offset_table", &jax_nerf::envr_get_offset_table);
}