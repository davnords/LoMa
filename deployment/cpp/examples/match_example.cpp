// Minimal LoMa C++ usage: match two images and draw the correspondences.
//
//   ./loma_match <onnx_dir> <imgA> <imgB> [preset] [out.jpg]
//   preset = fast | balanced | quality   (default: fast)
//
// `onnx_dir` must contain (for the chosen preset / variant):
//   loma_detector_<preset>.onnx
//   loma_descriptor_dedode_b_<preset>.onnx   (B128, descriptor_dim=128)
//   loma_matcher_B128.onnx
#include <iostream>
#include <map>
#include <string>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "loma/loma.hpp"

struct Preset { int detH, detW, desc, kpts; };  // detW==detH unless landscape
static const std::map<std::string, Preset> kPresets = {
    {"pico",     {256, 256, 256, 512}},
    {"nano",     {256, 256, 256, 1024}},
    {"turbo",    {384, 384, 384, 1024}},
    {"fast",     {512, 512, 512, 1024}},
    {"balanced", {640, 640, 640, 1536}},
    {"quality",  {1024, 1024, 784, 2048}},
    {"wide",     {512, 1024, 512, 2048}},
};

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "usage: " << argv[0] << " <onnx_dir> <imgA> <imgB>"
              << " [pico|nano|turbo|fast|balanced|quality|wide] [out.jpg]\n";
    return 1;
  }
  std::string dir = argv[1], pa = argv[2], pb = argv[3];
  std::string preset = argc > 4 ? argv[4] : "fast";
  std::string out = argc > 5 ? argv[5] : "loma_matches.jpg";
  if (!kPresets.count(preset)) { std::cerr << "bad preset\n"; return 1; }
  const Preset p = kPresets.at(preset);

  loma::Options opt;
  opt.detector_path   = dir + "/loma_detector_" + preset + ".onnx";
  opt.descriptor_path = dir + "/loma_descriptor_dedode_b_" + preset + ".onnx";
  opt.matcher_path    = dir + "/loma_matcher_B128.onnx";
  opt.detector_size = p.detH;
  opt.detector_width = p.detW;
  opt.descriptor_size = p.desc;
  opt.num_keypoints = p.kpts;
  opt.descriptor_dim = 128;            // B128 -> DeDoDe-B
  opt.provider = loma::Provider::Auto; // CUDA if available, else CPU

  cv::Mat A = cv::imread(pa, cv::IMREAD_COLOR);
  cv::Mat B = cv::imread(pb, cv::IMREAD_COLOR);
  if (A.empty() || B.empty()) { std::cerr << "could not read images\n"; return 1; }

  try {
    loma::LoMa model(opt);
    auto matches = model.match(A, B);   // warm-up + timed run below
    matches = model.match(A, B);
    const auto& t = model.lastTimings();
    std::cout << "matches: " << matches.size()
              << " | extract(2 imgs): " << t.describe_ms << " ms"
              << " | match: " << t.match_ms << " ms"
              << " | total: " << t.total_ms << " ms"
              << " | provider: " << model.provider() << "\n";

    // draw
    cv::Mat canvas(std::max(A.rows, B.rows), A.cols + B.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    A.copyTo(canvas(cv::Rect(0, 0, A.cols, A.rows)));
    B.copyTo(canvas(cv::Rect(A.cols, 0, B.cols, B.rows)));
    cv::RNG rng(0);
    for (const auto& m : matches) {
      cv::Scalar c(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
      cv::line(canvas, m.a, cv::Point2f(m.b.x + A.cols, m.b.y), c, 1, cv::LINE_AA);
    }
    cv::imwrite(out, canvas);
    std::cout << "wrote " << out << "\n";
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 2;
  }
  return 0;
}
