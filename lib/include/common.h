#pragma once
#include <dlpack/dlpack.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

namespace jax_nerf {
// get ptr from py capsule
template <typename T> T *dltensor_data(const py::capsule a) {
    DLManagedTensor *a_ptr = a.get_pointer<DLManagedTensor>();

    return (T *)a_ptr->dl_tensor.data;
}
} // namespace jax_nerf
