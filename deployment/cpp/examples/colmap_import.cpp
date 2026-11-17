// LoMa -> COLMAP: extract keypoints + match every image pair with LoMa and
// write cameras, images, keypoints and raw matches straight into a COLMAP
// SQLite database, ready for geometric verification + reconstruction.
//
//   ./loma_colmap_import <onnx_dir> <database.db> <img1> <img2> [img3 ...] [--preset fast]
//
// `onnx_dir` must contain the chosen preset's B128 trio:
//   loma_detector_<preset>.onnx
//   loma_descriptor_dedode_b_<preset>.onnx   (descriptor_dim = 128)
//   loma_matcher_B128.onnx
//
// After running this, finish the standard COLMAP flow:
//   # geometric verification of the raw matches we wrote (writes two_view_geometries):
//   colmap matches_importer --database_path database.db \
//          --match_list_path database.db.matches.txt --match_type raw
//   # then reconstruct:
//   mkdir -p sparse && colmap mapper --database_path database.db \
//          --image_path <image_dir> --output_path sparse
//
// LoMa is the matcher (not a descriptor for NN search), so we import matches
// directly and let COLMAP verify them — no SIFT/NN step. We also write a raw
// match list (<db>.matches.txt) for `matches_importer`; the matches table is
// populated too, for pipelines that verify straight from the DB (e.g. pycolmap).
//
// Build requires SQLite3 (libsqlite3-dev); the target is only configured when
// SQLite3 is found (see CMakeLists.txt).
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include <sqlite3.h>

#include <opencv2/imgcodecs.hpp>

#include "loma/loma.hpp"

namespace {

struct Preset { int detH, detW, desc, kpts; };  // detW==detH unless landscape
const std::map<std::string, Preset> kPresets = {
    {"pico",     {256, 256, 256, 512}},
    {"nano",     {256, 256, 256, 1024}},
    {"turbo",    {384, 384, 384, 1024}},
    {"fast",     {512, 512, 512, 1024}},
    {"balanced", {640, 640, 640, 1536}},
    {"quality",  {1024, 1024, 784, 2048}},
    {"wide",     {512, 1024, 512, 2048}},
};

// COLMAP keypoint coordinate, computed exactly as loma::to_pixel so a Match
// point can be looked up back to its keypoint index by bit-identical key.
cv::Point2f toPixel(float nx, float ny, const cv::Size& s) {
  return {s.width * (nx + 1.f) * 0.5f, s.height * (ny + 1.f) * 0.5f};
}

uint64_t pointKey(float x, float y) {
  uint32_t xb, yb;
  std::memcpy(&xb, &x, 4);
  std::memcpy(&yb, &y, 4);
  return (static_cast<uint64_t>(xb) << 32) | yb;
}

// COLMAP image-pair id (database.h): smaller id first, pair_id = a*2^31-1 + b.
constexpr int64_t kMaxNumImages2 = 2147483647;
int64_t pairId(int64_t a, int64_t b) {
  if (a > b) std::swap(a, b);
  return a * kMaxNumImages2 + b;
}

void exec(sqlite3* db, const char* sql) {
  char* err = nullptr;
  if (sqlite3_exec(db, sql, nullptr, nullptr, &err) != SQLITE_OK) {
    std::string m = err ? err : "?";
    sqlite3_free(err);
    throw std::runtime_error("sqlite: " + m);
  }
}

void check(sqlite3* db, int rc, const char* what) {
  if (rc != SQLITE_OK && rc != SQLITE_DONE && rc != SQLITE_ROW)
    throw std::runtime_error(std::string("sqlite ") + what + ": " +
                             sqlite3_errmsg(db));
}

// Minimal subset of COLMAP's schema; harmless `IF NOT EXISTS` against a DB that
// `colmap database_creator` already made.
const char* kSchema = R"sql(
CREATE TABLE IF NOT EXISTS cameras (
  camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
  model INTEGER NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL,
  params BLOB, prior_focal_length INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS images (
  image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
  name TEXT NOT NULL UNIQUE, camera_id INTEGER NOT NULL,
  prior_qw REAL, prior_qx REAL, prior_qy REAL, prior_qz REAL,
  prior_tx REAL, prior_ty REAL, prior_tz REAL,
  CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < 2147483647),
  FOREIGN KEY(camera_id) REFERENCES cameras(camera_id));
CREATE TABLE IF NOT EXISTS keypoints (
  image_id INTEGER PRIMARY KEY NOT NULL, rows INTEGER NOT NULL,
  cols INTEGER NOT NULL, data BLOB,
  FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS descriptors (
  image_id INTEGER PRIMARY KEY NOT NULL, rows INTEGER NOT NULL,
  cols INTEGER NOT NULL, data BLOB,
  FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS matches (
  pair_id INTEGER PRIMARY KEY NOT NULL, rows INTEGER NOT NULL,
  cols INTEGER NOT NULL, data BLOB);
CREATE TABLE IF NOT EXISTS two_view_geometries (
  pair_id INTEGER PRIMARY KEY NOT NULL, rows INTEGER NOT NULL,
  cols INTEGER NOT NULL, data BLOB, config INTEGER NOT NULL,
  F BLOB, E BLOB, H BLOB, qvec BLOB, tvec BLOB);
)sql";

// Per-image record kept around so we can recover match indices afterwards.
struct ImgRec {
  int64_t image_id;
  std::string name;
  std::unordered_map<uint64_t, int> kpt_index;  // pixel key -> keypoint index
};

}  // namespace

int main(int argc, char** argv) {
  std::string preset = "fast";
  std::vector<std::string> pos;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--preset" && i + 1 < argc) preset = argv[++i];
    else pos.push_back(a);
  }
  if (pos.size() < 4) {
    std::cerr << "usage: " << argv[0]
              << " <onnx_dir> <database.db> <img1> <img2> [img3 ...] [--preset fast]\n";
    return 1;
  }
  const std::string dir = pos[0], dbpath = pos[1];
  const std::vector<std::string> images(pos.begin() + 2, pos.end());
  if (!kPresets.count(preset)) { std::cerr << "unknown preset: " << preset << "\n"; return 1; }
  const Preset p = kPresets.at(preset);

  loma::Options opt;
  opt.detector_path   = dir + "/loma_detector_" + preset + ".onnx";
  opt.descriptor_path = dir + "/loma_descriptor_dedode_b_" + preset + ".onnx";
  opt.matcher_path    = dir + "/loma_matcher_B128.onnx";
  opt.detector_size = p.detH; opt.detector_width = p.detW;
  opt.descriptor_size = p.desc; opt.num_keypoints = p.kpts;
  opt.descriptor_dim = 128; opt.provider = loma::Provider::Auto;

  sqlite3* db = nullptr;
  if (sqlite3_open(dbpath.c_str(), &db) != SQLITE_OK) {
    std::cerr << "cannot open " << dbpath << ": " << sqlite3_errmsg(db) << "\n";
    return 2;
  }
  std::ofstream match_list(dbpath + ".matches.txt");

  try {
    exec(db, "PRAGMA foreign_keys=ON;");
    exec(db, kSchema);
    exec(db, "BEGIN;");

    loma::LoMa model(opt);
    std::vector<loma::Features> feats;
    std::vector<ImgRec> recs;

    // ---- per image: extract + write camera/image/keypoints ----
    for (const auto& path : images) {
      cv::Mat bgr = cv::imread(path, cv::IMREAD_COLOR);
      if (bgr.empty()) throw std::runtime_error("could not read image: " + path);
      loma::Features f = model.extract(bgr);

      // SIMPLE_PINHOLE camera (model 0): params = [f, cx, cy].
      const int w = bgr.cols, h = bgr.rows;
      double cam[3] = {1.2 * std::max(w, h), w / 2.0, h / 2.0};
      sqlite3_stmt* cs = nullptr;
      check(db, sqlite3_prepare_v2(db,
            "INSERT INTO cameras(model,width,height,params,prior_focal_length)"
            " VALUES(0,?,?,?,0);", -1, &cs, nullptr), "prep camera");
      sqlite3_bind_int(cs, 1, w);
      sqlite3_bind_int(cs, 2, h);
      sqlite3_bind_blob(cs, 3, cam, sizeof(cam), SQLITE_TRANSIENT);
      check(db, sqlite3_step(cs), "insert camera");
      sqlite3_finalize(cs);
      const int64_t camera_id = sqlite3_last_insert_rowid(db);

      // strip directory for the COLMAP image name (matches --image_path layout)
      std::string name = path.substr(path.find_last_of("/\\") + 1);
      sqlite3_stmt* is = nullptr;
      check(db, sqlite3_prepare_v2(db,
            "INSERT INTO images(name,camera_id) VALUES(?,?);", -1, &is, nullptr),
            "prep image");
      sqlite3_bind_text(is, 1, name.c_str(), -1, SQLITE_TRANSIENT);
      sqlite3_bind_int64(is, 2, camera_id);
      check(db, sqlite3_step(is), "insert image");
      sqlite3_finalize(is);
      const int64_t image_id = sqlite3_last_insert_rowid(db);

      // keypoints blob: rows=N, cols=2 (x,y) float32; build pixel->index map.
      ImgRec rec; rec.image_id = image_id; rec.name = name;
      std::vector<float> kp(static_cast<size_t>(f.n) * 2);
      for (int i = 0; i < f.n; ++i) {
        cv::Point2f px = toPixel(f.kpts[2 * i], f.kpts[2 * i + 1], f.image_size);
        kp[2 * i] = px.x; kp[2 * i + 1] = px.y;
        rec.kpt_index[pointKey(px.x, px.y)] = i;
      }
      sqlite3_stmt* ks = nullptr;
      check(db, sqlite3_prepare_v2(db,
            "INSERT INTO keypoints(image_id,rows,cols,data) VALUES(?,?,2,?);",
            -1, &ks, nullptr), "prep keypoints");
      sqlite3_bind_int64(ks, 1, image_id);
      sqlite3_bind_int(ks, 2, f.n);
      sqlite3_bind_blob(ks, 3, kp.data(),
                        static_cast<int>(kp.size() * sizeof(float)), SQLITE_TRANSIENT);
      check(db, sqlite3_step(ks), "insert keypoints");
      sqlite3_finalize(ks);

      feats.push_back(std::move(f));
      recs.push_back(std::move(rec));
      std::cout << "image " << image_id << "  " << name
                << "  (" << recs.back().kpt_index.size() << " kpts)\n";
    }

    // ---- every pair: match + write matches table + raw match list ----
    int64_t total = 0;
    for (size_t a = 0; a < images.size(); ++a) {
      for (size_t b = a + 1; b < images.size(); ++b) {
        auto matches = model.matchFeatures(feats[a], feats[b]);
        // recover keypoint indices from the Match pixel points
        std::vector<uint32_t> pairs;  // row-major (idx_in_a, idx_in_b)
        pairs.reserve(matches.size() * 2);
        for (const auto& m : matches) {
          auto ia = recs[a].kpt_index.find(pointKey(m.a.x, m.a.y));
          auto ib = recs[b].kpt_index.find(pointKey(m.b.x, m.b.y));
          if (ia == recs[a].kpt_index.end() || ib == recs[b].kpt_index.end())
            continue;  // should not happen: points come from toPixel
          pairs.push_back(static_cast<uint32_t>(ia->second));
          pairs.push_back(static_cast<uint32_t>(ib->second));
        }
        const uint32_t M = static_cast<uint32_t>(pairs.size() / 2);
        if (M == 0) continue;

        // COLMAP stores columns with the smaller image_id first.
        bool swap = recs[a].image_id > recs[b].image_id;
        if (swap)
          for (uint32_t r = 0; r < M; ++r) std::swap(pairs[2 * r], pairs[2 * r + 1]);

        sqlite3_stmt* ms = nullptr;
        check(db, sqlite3_prepare_v2(db,
              "INSERT OR REPLACE INTO matches(pair_id,rows,cols,data)"
              " VALUES(?,?,2,?);", -1, &ms, nullptr), "prep matches");
        sqlite3_bind_int64(ms, 1, pairId(recs[a].image_id, recs[b].image_id));
        sqlite3_bind_int(ms, 2, static_cast<int>(M));
        sqlite3_bind_blob(ms, 3, pairs.data(),
                          static_cast<int>(pairs.size() * sizeof(uint32_t)),
                          SQLITE_TRANSIENT);
        check(db, sqlite3_step(ms), "insert matches");
        sqlite3_finalize(ms);

        // raw match list (image names + 0-based index pairs) for matches_importer
        match_list << recs[a].name << " " << recs[b].name << "\n";
        for (uint32_t r = 0; r < M; ++r) {
          uint32_t ai = pairs[2 * r], bi = pairs[2 * r + 1];
          if (swap) std::swap(ai, bi);  // write in (a,b) order for readability
          match_list << ai << " " << bi << "\n";
        }
        match_list << "\n";

        total += M;
        std::cout << "pair " << recs[a].name << " <-> " << recs[b].name
                  << "  " << M << " matches\n";
      }
    }

    exec(db, "COMMIT;");
    match_list.close();
    sqlite3_close(db);

    std::cout << "\nwrote " << recs.size() << " images, " << total
              << " matches into " << dbpath << "\n"
              << "raw match list: " << dbpath << ".matches.txt\n\n"
              << "next:\n"
              << "  colmap matches_importer --database_path " << dbpath
              << " --match_list_path " << dbpath << ".matches.txt --match_type raw\n"
              << "  colmap mapper --database_path " << dbpath
              << " --image_path <image_dir> --output_path sparse\n";
  } catch (const std::exception& e) {
    sqlite3_exec(db, "ROLLBACK;", nullptr, nullptr, nullptr);
    sqlite3_close(db);
    std::cerr << "error: " << e.what() << "\n";
    return 3;
  }
  return 0;
}
