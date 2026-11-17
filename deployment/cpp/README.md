# LoMa C++ — local feature matching via ONNX Runtime

A small, reusable C++ library that runs the LoMa ONNX pipeline (detector →
descriptor → matcher) and returns pixel-coordinate correspondences. Designed to be
dropped into other C++ projects (SfM, visual localization, HIL/camera systems) and to
run on NVIDIA Jetson Orin Nano / DGX Spark with the CUDA execution provider.

```cpp
#include "loma/loma.hpp"

loma::Options opt;
opt.detector_path   = "onnx/loma_detector_fast.onnx";
opt.descriptor_path = "onnx/loma_descriptor_dedode_b_fast.onnx";
opt.matcher_path    = "onnx/loma_matcher_B128.onnx";
opt.detector_size = 512; opt.descriptor_size = 512;
opt.num_keypoints = 1024; opt.descriptor_dim = 128;   // B128 = DeDoDe-B
opt.provider = loma::Provider::Auto;                  // CUDA EP -> CPU fallback

loma::LoMa model(opt);
cv::Mat a = cv::imread("a.jpg"), b = cv::imread("b.jpg");
std::vector<loma::Match> matches = model.match(a, b);  // {a, b, score} in pixels
```

## Layout
```
cpp/
  include/loma/loma.hpp     public API (depends only on OpenCV)
  src/loma.cpp              implementation (ONNX Runtime hidden behind pimpl)
  examples/match_example.cpp   draw correspondences
  examples/benchmark.cpp       per-stage latency / FPS
  CMakeLists.txt            builds libloma + examples, installs find_package(LoMa)
```

## Dependencies
- **OpenCV** (core, imgproc, imgcodecs)
- **ONNX Runtime** ≥ 1.19 (CPU works anywhere; **GPU needs a CUDA build** — see below)

## Building ONNX Runtime with CUDA — the platform that matters

`pip install onnxruntime-gpu` does **not** work on aarch64 + CUDA (DGX Spark GB10
sm_121, Jetson Orin). There is no prebuilt aarch64 GPU wheel. Pick one:

### Jetson Orin Nano (JetPack)
JetPack ships an `onnxruntime-gpu` build. Headers/libs are under
`/usr/include/onnxruntime` and `/usr/lib/aarch64-linux-gnu` (or install the NVIDIA
`onnxruntime-gpu` wheel from the [Jetson Zoo](https://elinux.org/Jetson_Zoo)). Then:
```bash
cmake -B build -DONNXRUNTIME_ROOT_DIR=/usr
```

### DGX Spark / GB10 (sm_121, CUDA 13)
No prebuilt wheel exists (as of 2026). Two options:
1. **Prebuilt C++ shared libs** —
   [Albatross1382/onnxruntime-aarch64-cuda-blackwell](https://github.com/Albatross1382/onnxruntime-aarch64-cuda-blackwell)
   provides `libonnxruntime.so.1.24.4` + `libonnxruntime_providers_cuda.so`. Drop them
   in a dir with the v1.24.4 headers and point CMake at it.
2. **Build from source** (≥ v1.24.4 — earlier versions don't support CUDA 13):
   ```bash
   git clone --recursive --branch v1.24.4 --depth 1 \
       https://github.com/microsoft/onnxruntime.git && cd onnxruntime
   ./build.sh --config Release --use_cuda \
       --cuda_home /usr/local/cuda --cudnn_home /usr \
       --build_shared_lib --parallel --skip_tests \
       --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=121
   # headers: include/  libs: build/Linux/Release/
   ```
   The repo's `build_ort_gpu.sh` automates this (and also builds the Python wheel).
   **Use FP32 models** — sm_121 lacks INT8 kernels. (Our exports are FP32. ✓)

## Build LoMa C++
```bash
cd cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DONNXRUNTIME_ROOT_DIR=/path/to/onnxruntime   # dir with include/ + lib/
cmake --build build -j

# run
./build/loma_match  ../onnx  a.jpg b.jpg  fast  out.jpg
./build/loma_bench  ../onnx  a.jpg b.jpg  fast  100
```
If ONNX Runtime libs aren't on the default loader path:
`export LD_LIBRARY_PATH=/path/to/onnxruntime/lib:$LD_LIBRARY_PATH`.

## Use from another CMake project
```cmake
find_package(LoMa REQUIRED)
target_link_libraries(my_app PRIVATE LoMa::loma)
```

## Choosing models (preset = baked resolution + keypoint count)
| preset    | detector | descriptor | keypoints | notes                          |
|-----------|----------|------------|-----------|--------------------------------|
| fast      | 512      | 512        | 1024      | lowest latency — Orin Nano RT  |
| balanced  | 640      | 640        | 1536      | more matches                   |
| quality   | 1024     | 784        | 2048      | most matches, heaviest         |

- `Options.detector_size` / `descriptor_size` / `num_keypoints` **must match** the
  chosen ONNX files (resolution + topk are baked into the graph).
- **Variants:** B128 → `descriptor_dim=128` (DeDoDe-B, lightest, recommended for 8 GB).
  B/R/L/G → `descriptor_dim=256` (DeDoDe-G, includes a 1.3 GB DINOv2 descriptor —
  heavy for Orin Nano). The matcher ONNX (`loma_matcher_<V>.onnx`) is dynamic and
  shared across presets.
- Detector inputs are square here for simplicity; re-export at your camera's aspect
  ratio (`export_jetson.py`) if distortion hurts accuracy.

## API notes
- `match()` preprocesses internally (resize → RGB → [0,1] CHW); detector mean/std
  normalization is inside the ONNX graph.
- `extract()` / `matchFeatures()` expose the stages for one-to-many matching (SfM).
- The matcher applies its mutual-NN + threshold filter inside the graph; `match()`
  returns only surviving correspondences.
- `lastTimings()` reports per-call latency; `provider()` reports the EP actually used.
