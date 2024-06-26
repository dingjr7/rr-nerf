set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-mf16c")
list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-Wno-float-conversion")
list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fno-strict-aliasing")
list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fPIC")
list(APPEND CUDA_NVCC_FLAGS "--extended-lambda")
list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")

set(TCNN_BUILD_BENCHMARK OFF)
set(TCNN_BUILD_EXAMPLES OFF)
add_subdirectory(dependencies/tiny-cuda-nn)
set(CMAKE_CUDA_ARCHITECTURES ${TCNN_CUDA_ARCHITECTURES})

find_package(pybind11 REQUIRED CONFIG)
find_package(Python 3.9 COMPONENTS Interpreter Development REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)

# Create an extension module
include_directories(
     "include"
     "dependencies/dlpack/include")

file(GLOB NGP_SRC CONFIGURE_DEPENDS
     "src/*.cc"
     "src/*.cu"
)

pybind11_add_module(jax_nerf "bind.cu" ${NGP_SRC})
set_target_properties(jax_nerf PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(jax_nerf PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
target_link_libraries(jax_nerf PUBLIC Eigen3::Eigen tiny-cuda-nn)
set_target_properties(jax_nerf
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/src"
  LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/src"
)