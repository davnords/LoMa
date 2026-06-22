#include "loma/loma.hpp"

#include <onnxruntime_cxx_api.h>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace loma {
namespace {

using Clock = std::chrono::high_resolution_clock;
inline double ms_since(const Clock::time_point& t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// Resize a BGR image to H×W and pack as float CHW RGB in [0, 1].
std::vector<float> preprocess(const cv::Mat& bgr, int H, int W) {
  cv::Mat resized, rgb;
  cv::resize(bgr, resized, cv::Size(W, H), 0, 0, cv::INTER_LINEAR);  // cv::Size is (w,h)
  cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
  rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);
  std::vector<float> chw(static_cast<size_t>(3) * H * W);
  std::vector<cv::Mat> ch(3);
  // point each channel Mat at the right slice of `chw` so split writes CHW directly
  for (int c = 0; c < 3; ++c)
    ch[c] = cv::Mat(H, W, CV_32F, chw.data() + static_cast<size_t>(c) * H * W);
  cv::split(rgb, ch);
  return chw;
}

inline cv::Point2f to_pixel(float nx, float ny, const cv::Size& s) {
  return {s.width * (nx + 1.f) * 0.5f, s.height * (ny + 1.f) * 0.5f};
}

}  // namespace

struct LoMa::Impl {
  Options opt;
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "loma"};
  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::string used_provider = "CPUExecutionProvider";
  Timings timings;

  struct Net {
    std::unique_ptr<Ort::Session> session;
    std::vector<std::string> in_names, out_names;
    std::vector<const char*> in_c, out_c;
    void buildC() {
      in_c.clear(); out_c.clear();
      for (auto& s : in_names) in_c.push_back(s.c_str());
      for (auto& s : out_names) out_c.push_back(s.c_str());
    }
  } detector, descriptor, matcher;

  explicit Impl(const Options& o) : opt(o) {
    Ort::SessionOptions so = makeSessionOptions();
    load(detector, opt.detector_path, so);
    load(descriptor, opt.descriptor_path, so);
    load(matcher, opt.matcher_path, so);
    if (opt.verbose)
      std::cerr << "[loma] provider=" << used_provider
                << " detector=" << opt.detector_size << "px"
                << " descriptor=" << opt.descriptor_size << "px"
                << " kpts=" << opt.num_keypoints
                << " desc_dim=" << opt.descriptor_dim << "\n";
  }

  Ort::SessionOptions makeSessionOptions() {
    Ort::SessionOptions so;
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (opt.intra_op_threads > 0) so.SetIntraOpNumThreads(opt.intra_op_threads);

    const bool want_cuda =
        opt.provider == Provider::Cuda || opt.provider == Provider::Auto;
    if (want_cuda) {
      try {
        OrtCUDAProviderOptions cuda{};
        cuda.device_id = opt.device_id;
        so.AppendExecutionProvider_CUDA(cuda);
        used_provider = "CUDAExecutionProvider";
        return so;
      } catch (const std::exception& e) {
        if (opt.provider == Provider::Cuda)
          throw std::runtime_error(std::string("CUDA EP requested but unavailable: ") + e.what());
        if (opt.verbose)
          std::cerr << "[loma] CUDA EP unavailable, falling back to CPU: " << e.what() << "\n";
        // rebuild a clean CPU SessionOptions (the failed one may be tainted)
        Ort::SessionOptions cpu;
        cpu.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        if (opt.intra_op_threads > 0) cpu.SetIntraOpNumThreads(opt.intra_op_threads);
        used_provider = "CPUExecutionProvider";
        return cpu;
      }
    }
    used_provider = "CPUExecutionProvider";
    return so;
  }

  void load(Net& net, const std::string& path, Ort::SessionOptions& so) {
    if (path.empty()) throw std::runtime_error("[loma] empty model path");
    net.session = std::make_unique<Ort::Session>(env, path.c_str(), so);
    Ort::AllocatorWithDefaultOptions alloc;
    for (size_t i = 0; i < net.session->GetInputCount(); ++i)
      net.in_names.emplace_back(net.session->GetInputNameAllocated(i, alloc).get());
    for (size_t i = 0; i < net.session->GetOutputCount(); ++i)
      net.out_names.emplace_back(net.session->GetOutputNameAllocated(i, alloc).get());
    net.buildC();
  }

  Ort::Value tensor(std::vector<float>& data, std::vector<int64_t> shape) {
    return Ort::Value::CreateTensor<float>(mem, data.data(), data.size(),
                                           shape.data(), shape.size());
  }

  int outIndex(const Net& net, const std::string& name) const {
    for (size_t i = 0; i < net.out_names.size(); ++i)
      if (net.out_names[i] == name) return static_cast<int>(i);
    throw std::runtime_error("[loma] missing output: " + name);
  }

  // ---- stages ----
  Features extract(const cv::Mat& bgr) {
    Features f;
    f.image_size = bgr.size();
    const int N = opt.num_keypoints, D = opt.descriptor_dim;

    // detector (square unless detector_width is set, e.g. wide = 512×1024)
    const int detH = opt.detector_size;
    const int detW = opt.detector_width > 0 ? opt.detector_width : opt.detector_size;
    auto det_in = preprocess(bgr, detH, detW);
    std::vector<int64_t> det_shape{1, 3, detH, detW};
    Ort::Value dv = tensor(det_in, det_shape);
    auto det_out = detector.session->Run(Ort::RunOptions{nullptr}, detector.in_c.data(),
                                         &dv, 1, detector.out_c.data(),
                                         detector.out_names.size());
    int ki = outIndex(detector, "keypoints");
    int pi = outIndex(detector, "keypoint_probs");
    const float* kp = det_out[ki].GetTensorData<float>();
    const float* pr = det_out[pi].GetTensorData<float>();
    f.n = N;
    f.kpts.assign(kp, kp + static_cast<size_t>(N) * 2);
    f.probs.assign(pr, pr + N);

    // descriptor (image + the detector's keypoints); always square
    auto dsc_in = preprocess(bgr, opt.descriptor_size, opt.descriptor_size);
    std::vector<int64_t> dsc_shape{1, 3, opt.descriptor_size, opt.descriptor_size};
    std::vector<int64_t> kpt_shape{1, N, 2};
    std::vector<Ort::Value> dsc_inputs;
    dsc_inputs.push_back(tensor(dsc_in, dsc_shape));
    dsc_inputs.push_back(tensor(f.kpts, kpt_shape));
    // input order must follow descriptor.in_names ("image", "keypoints")
    std::vector<Ort::Value> ordered;
    for (auto& nm : descriptor.in_names)
      ordered.push_back(nm == "image" ? std::move(dsc_inputs[0]) : std::move(dsc_inputs[1]));
    auto dsc_out = descriptor.session->Run(Ort::RunOptions{nullptr}, descriptor.in_c.data(),
                                           ordered.data(), ordered.size(),
                                           descriptor.out_c.data(), descriptor.out_names.size());
    const float* ds = dsc_out[0].GetTensorData<float>();
    f.desc.assign(ds, ds + static_cast<size_t>(N) * D);
    return f;
  }

  std::vector<Match> matchFeatures(const Features& fa, const Features& fb) {
    const int Na = fa.n, Nb = fb.n, D = opt.descriptor_dim;
    std::vector<float> k0 = fa.kpts, k1 = fb.kpts, d0 = fa.desc, d1 = fb.desc;
    std::vector<int64_t> k0s{1, Na, 2}, k1s{1, Nb, 2}, d0s{1, Na, D}, d1s{1, Nb, D};

    // build inputs in the matcher's declared name order
    std::vector<Ort::Value> in;
    for (auto& nm : matcher.in_names) {
      if (nm == "kpts0") in.push_back(tensor(k0, k0s));
      else if (nm == "kpts1") in.push_back(tensor(k1, k1s));
      else if (nm == "desc0") in.push_back(tensor(d0, d0s));
      else if (nm == "desc1") in.push_back(tensor(d1, d1s));
      else throw std::runtime_error("[loma] unexpected matcher input: " + nm);
    }
    auto out = matcher.session->Run(Ort::RunOptions{nullptr}, matcher.in_c.data(),
                                    in.data(), in.size(), matcher.out_c.data(),
                                    matcher.out_names.size());
    const int64_t* m0 = out[outIndex(matcher, "m0")].GetTensorData<int64_t>();
    const float* ms0 = out[outIndex(matcher, "mscores0")].GetTensorData<float>();

    std::vector<Match> matches;
    matches.reserve(Na / 4);
    for (int i = 0; i < Na; ++i) {
      int64_t j = m0[i];
      if (j < 0 || j >= Nb) continue;  // -1 = filtered out by the matcher
      Match mt;
      mt.a = to_pixel(fa.kpts[2 * i], fa.kpts[2 * i + 1], fa.image_size);
      mt.b = to_pixel(fb.kpts[2 * j], fb.kpts[2 * j + 1], fb.image_size);
      mt.score = ms0[i];
      matches.push_back(mt);
    }
    return matches;
  }
};

LoMa::LoMa(const Options& opt) : impl_(std::make_unique<Impl>(opt)) {}
LoMa::~LoMa() = default;
LoMa::LoMa(LoMa&&) noexcept = default;
LoMa& LoMa::operator=(LoMa&&) noexcept = default;

Features LoMa::extract(const cv::Mat& imgBGR) {
  if (imgBGR.empty()) throw std::runtime_error("[loma] empty input image");
  return impl_->extract(imgBGR);
}

std::vector<Match> LoMa::matchFeatures(const Features& fa, const Features& fb) {
  auto t0 = Clock::now();
  auto m = impl_->matchFeatures(fa, fb);
  impl_->timings.match_ms = ms_since(t0);
  return m;
}

std::vector<Match> LoMa::match(const cv::Mat& imgA, const cv::Mat& imgB) {
  if (imgA.empty() || imgB.empty()) throw std::runtime_error("[loma] empty input image");
  auto t_all = Clock::now();
  auto t0 = Clock::now();
  Features fa = impl_->extract(imgA);
  Features fb = impl_->extract(imgB);
  impl_->timings.detect_ms = 0;            // (extract folds detect+describe together)
  impl_->timings.describe_ms = ms_since(t0);
  auto t1 = Clock::now();
  auto matches = impl_->matchFeatures(fa, fb);
  impl_->timings.match_ms = ms_since(t1);
  impl_->timings.total_ms = ms_since(t_all);
  return matches;
}

const Timings& LoMa::lastTimings() const { return impl_->timings; }
const std::string& LoMa::provider() const { return impl_->used_provider; }

}  // namespace loma
