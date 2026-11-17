// LoMa C++ — local feature matching via ONNX Runtime.
//
// A LoMa model is a 3-stage ONNX pipeline:
//   detector   (DaD)            image            -> keypoints (normalized), probs
//   descriptor (DeDoDe B / G)   image, keypoints -> descriptions
//   matcher    (transformer)    kpts/desc x2     -> match indices + scores
//
// Detector & descriptor are exported at a FIXED square resolution and a FIXED
// keypoint count (baked into the graph). The matcher is dynamic. Pick the ONNX
// trio that matches your latency/accuracy budget (see the `*_fast` / `*_balanced`
// / full-res exports) and set Options accordingly.
//
// This header is dependency-light (only OpenCV in the API). ONNX Runtime is an
// implementation detail hidden behind a pimpl, so downstream projects only need
// to link `loma` and OpenCV.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace loma {

// One accepted correspondence, in pixel coordinates of the original input images.
struct Match {
  cv::Point2f a;   // keypoint in image A
  cv::Point2f b;   // keypoint in image B (the mutual match of a)
  float score;     // matchability score (higher = more confident)
};

// Per-image extracted features. Keypoints are in NORMALIZED coords ([-1, 1]),
// independent of input resolution, so they can be reused across stages.
struct Features {
  std::vector<float> kpts;   // n*2 row-major: (x, y) in [-1, 1]
  std::vector<float> desc;   // n*descriptor_dim row-major
  std::vector<float> probs;  // n keypoint confidences
  int n = 0;                 // number of keypoints
  cv::Size image_size;       // original image size (for pixel conversion)
};

struct Timings {
  double detect_ms = 0;     // both images
  double describe_ms = 0;   // both images
  double match_ms = 0;
  double total_ms = 0;
};

enum class Provider { Auto, Cuda, Cpu };  // Auto = CUDA if available, else CPU

struct Options {
  // ---- model paths (required) ----
  std::string detector_path;
  std::string descriptor_path;
  std::string matcher_path;

  // ---- must match the exported ONNX trio ----
  int detector_size = 512;     // detector input height the ONNX expects
  int detector_width = 0;      // detector input width; 0 → square (= detector_size).
                               // set both for landscape presets (e.g. wide = 512×1024)
  int descriptor_size = 512;   // square H=W the descriptor ONNX expects
  int num_keypoints = 1024;    // baked into the detector graph (informational)
  int descriptor_dim = 128;    // 128 = DeDoDe-B (B128); 256 = DeDoDe-G (B/R/L/G)

  // ---- runtime ----
  Provider provider = Provider::Auto;
  int device_id = 0;
  int intra_op_threads = 0;    // 0 = ORT default
  bool verbose = true;         // log provider selection / model load
};

class LoMa {
 public:
  explicit LoMa(const Options& opt);
  ~LoMa();
  LoMa(LoMa&&) noexcept;
  LoMa& operator=(LoMa&&) noexcept;
  LoMa(const LoMa&) = delete;
  LoMa& operator=(const LoMa&) = delete;

  // Full pipeline on a BGR (OpenCV-native) image pair.
  std::vector<Match> match(const cv::Mat& imgA, const cv::Mat& imgB);

  // Lower-level stages, for reuse in SfM / visual-localization pipelines where
  // features of one image are matched against many.
  Features extract(const cv::Mat& imgBGR);
  std::vector<Match> matchFeatures(const Features& fa, const Features& fb);

  // Timings of the most recent match()/matchFeatures() call.
  const Timings& lastTimings() const;

  // Which execution provider actually got used ("CUDAExecutionProvider" / "CPU...").
  const std::string& provider() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace loma
