// LoMa C++ benchmark: per-stage latency + throughput + match quality.
//
//   ./loma_bench <onnx_dir> <imgA> <imgB> [preset] [iters]
//
// Reports: provider, extract/match/total latency (mean, p50, p95 over `iters`),
// FPS, and #matches. Pair with the Python reference numbers for a cross-check.
#include <algorithm>
#include <chrono>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "loma/loma.hpp"

struct Preset { int detH, detW, desc, kpts; };
static const std::map<std::string, Preset> kPresets = {
    {"pico",     {256, 256, 256, 512}},
    {"nano",     {256, 256, 256, 1024}},
    {"turbo",    {384, 384, 384, 1024}},
    {"fast",     {512, 512, 512, 1024}},
    {"balanced", {640, 640, 640, 1536}},
    {"quality",  {1024, 1024, 784, 2048}},
    {"wide",     {512, 1024, 512, 2048}},
};

static double pct(std::vector<double> v, double q) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  size_t i = static_cast<size_t>(q * (v.size() - 1));
  return v[i];
}
static double mean(const std::vector<double>& v) {
  return v.empty() ? 0 : std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "usage: " << argv[0] << " <onnx_dir> <imgA> <imgB>"
              << " [pico|nano|turbo|fast|balanced|quality|wide] [iters]\n";
    return 1;
  }
  std::string dir = argv[1], pa = argv[2], pb = argv[3];
  std::string preset = argc > 4 ? argv[4] : "fast";
  int iters = argc > 5 ? std::stoi(argv[5]) : 50;
  const Preset p = kPresets.at(preset);

  loma::Options opt;
  opt.detector_path   = dir + "/loma_detector_" + preset + ".onnx";
  opt.descriptor_path = dir + "/loma_descriptor_dedode_b_" + preset + ".onnx";
  opt.matcher_path    = dir + "/loma_matcher_B128.onnx";
  opt.detector_size = p.detH; opt.detector_width = p.detW;
  opt.descriptor_size = p.desc;
  opt.num_keypoints = p.kpts; opt.descriptor_dim = 128;
  opt.provider = loma::Provider::Auto;
  opt.verbose = true;

  cv::Mat A = cv::imread(pa, cv::IMREAD_COLOR);
  cv::Mat B = cv::imread(pb, cv::IMREAD_COLOR);
  if (A.empty() || B.empty()) { std::cerr << "bad images\n"; return 1; }

  loma::LoMa model(opt);

  // warm-up (CUDA kernel/engine init, allocator growth)
  size_t nmatch = 0;
  for (int i = 0; i < 5; ++i) nmatch = model.match(A, B).size();

  std::vector<double> ext, mat, tot;
  for (int i = 0; i < iters; ++i) {
    auto m = model.match(A, B);
    nmatch = m.size();
    const auto& t = model.lastTimings();
    ext.push_back(t.describe_ms);
    mat.push_back(t.match_ms);
    tot.push_back(t.total_ms);
  }

  std::cout << "\n==== LoMa C++ benchmark ====\n";
  std::cout << "preset       : " << preset << " (det " << p.detH << "x" << p.detW
            << ", desc " << p.desc << ", " << p.kpts << " kpts)\n";
  std::cout << "provider     : " << model.provider() << "\n";
  std::cout << "iters        : " << iters << "\n";
  std::cout << "#matches     : " << nmatch << "\n";
  auto row = [](const char* n, std::vector<double>& v) {
    std::cout << "  " << n << "  mean " << mean(v) << " ms | p50 " << pct(v, 0.5)
              << " | p95 " << pct(v, 0.95) << " ms\n";
  };
  std::cout << "latency:\n";
  row("extract(x2)", ext);
  row("match      ", mat);
  row("TOTAL      ", tot);
  std::cout << "throughput   : " << (1000.0 / mean(tot)) << " pairs/s\n";
  return 0;
}
