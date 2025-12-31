A matrix multiplication FLOPs test for Rust Burn
===============================================

## Usage

flops-test --backend cuda --float-type f16

Backends: CPU(there is a bug [tracel-ai/cubecl#1019](https://github.com/tracel-ai/cubecl/issues/1019)), Ndarray, WebGPU, Vulkan, CUDA. Details see help.

Float types: f16 f32 f64


## Results

(matrix size: 8192)

|           GPU           | Backend | Float Type |   GFLOPs  |       Note       |
|:-----------------------:|:-------:|:----------:|:---------:|:----------------:|
| NVIDIA GeForce RTX 3090 |   CUDA  |     f16    |  69097.44 |  570.133.20/12.4 |
|                         |         |     f32    |  21928.32 |                  |
|                         |         |     f64    |   442.78  |                  |
| NVIDIA GeForce RTX 4090 |   CUDA  |     f16    | 150747.60 |  570.133.20/12.4 |
|                         |         |     f32    |  49955.72 |                  |
|                         |         |     f64    |   494.06  |                  |
|                         |         |     f64    |  1159.81  | matrix-size=1024 |

