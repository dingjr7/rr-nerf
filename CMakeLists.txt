cmake_minimum_required(VERSION 3.20)

project(jax-nerf-lib
	VERSION 1.0
	DESCRIPTION "CUDA lib for Jax NeRF"
	LANGUAGES C CXX CUDA
)

add_subdirectory(lib)