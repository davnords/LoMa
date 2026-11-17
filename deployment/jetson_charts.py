"""Render Jetson Orin Nano benchmark charts + a GB10-vs-Orin comparison.

Reads docs/jetson_benchmark.json (Orin Nano, measured on-device) and
docs/benchmark.json (GB10) and writes:
  docs/jetson_latency.png   per-stage latency breakdown on the Orin Nano
  docs/jetson_vs_gb10.png   total-latency + speedup comparison (shared presets)
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = dict(detect="#4C72B0", describe="#DD8452", match="#55A868",
               accent="#C44E52", orin="#76B900", gb10="#414f67",
               grid="#E6E6E6", fg="#2B2B2B")

plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 160, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.25, "font.size": 11.5, "font.family": "DejaVu Sans",
    "axes.edgecolor": "#9aa0a6", "axes.linewidth": 1.0, "axes.grid": True,
    "grid.color": PALETTE["grid"], "grid.linewidth": 0.9, "axes.axisbelow": True,
    "text.color": PALETTE["fg"], "axes.labelcolor": PALETTE["fg"],
    "xtick.color": PALETTE["fg"], "ytick.color": PALETTE["fg"],
    "figure.facecolor": "white", "axes.facecolor": "#FBFBFC", "legend.frameon": False,
})


def titles(ax, main, sub):
    ax.set_title(sub, fontsize=9.5, color="#7a7f87", pad=8)
    ax.annotate(main, xy=(0.0, 1.0), xytext=(0.0, 22), xycoords="axes fraction",
                textcoords="offset points", fontsize=14, fontweight="bold",
                color=PALETTE["fg"], ha="left", va="bottom", annotation_clip=False)


jet = json.load(open("docs/jetson_benchmark.json"))
gb10 = json.load(open("docs/benchmark.json"))
jrows = [r for r in jet["rows"] if not r.get("oom")]
names = [r["name"] for r in jrows]
x = np.arange(len(jrows))
detect = [r["detect_ms"] for r in jrows]
describe = [r["describe_ms"] for r in jrows]
match = [r["match_ms"] for r in jrows]
total = [r["total_ms"] for r in jrows]
fps = [r["fps"] for r in jrows]
sub_j = f"{jet['variant']}  ·  {jet['device']}  ·  {jet['runtime']}"

# ---- chart 1: Orin Nano latency breakdown ----
fig, ax = plt.subplots(figsize=(8.6, 4.8))
ax.bar(x, detect, label="detect ×2", color=PALETTE["detect"], width=0.62)
ax.bar(x, describe, bottom=detect, label="describe ×2", color=PALETTE["describe"], width=0.62)
ax.bar(x, match, bottom=np.add(detect, describe), label="match", color=PALETTE["match"], width=0.62)
for i, t in enumerate(total):
    ax.text(i, t + max(total) * 0.02, f"{t:.0f} ms\n{fps[i]:.1f} fps", ha="center",
            va="bottom", fontsize=8.5, linespacing=1.25)
ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("latency per match() call (ms)")
ax.set_ylim(0, max(total) * 1.2); ax.legend(ncol=3, loc="upper left", fontsize=9.5)
titles(ax, "Jetson Orin Nano — ONNX-CUDA latency breakdown", sub_j)
fig.savefig("docs/jetson_latency.png"); plt.close(fig)

# ---- chart 2: GB10 vs Orin Nano (shared presets) ----
gmap = {r["name"]: r for r in gb10["rows"]}
shared = [n for n in names if n in gmap]
xs = np.arange(len(shared))
j_tot = [next(r for r in jrows if r["name"] == n)["total_ms"] for n in shared]
g_tot = [gmap[n]["total_ms"] for n in shared]
j_fps = [next(r for r in jrows if r["name"] == n)["fps"] for n in shared]
g_fps = [gmap[n]["fps"] for n in shared]
speed = [j / g for j, g in zip(j_tot, g_tot)]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 5))
w = 0.4
axL.bar(xs - w/2, g_tot, w, label="GB10 (DGX Spark)", color=PALETTE["gb10"])
axL.bar(xs + w/2, j_tot, w, label="Orin Nano 8GB", color=PALETTE["orin"])
for i, (g, j, s) in enumerate(zip(g_tot, j_tot, speed)):
    axL.text(i + w/2, j + max(j_tot) * 0.02, f"{j:.0f}ms\n{s:.1f}×", ha="center",
             va="bottom", fontsize=8.5, linespacing=1.2)
    axL.text(i - w/2, g + max(j_tot) * 0.02, f"{g:.0f}", ha="center", va="bottom", fontsize=8.5)
axL.set_xticks(xs); axL.set_xticklabels(shared); axL.set_ylabel("total latency per pair (ms)")
axL.set_ylim(0, max(j_tot) * 1.2); axL.legend(loc="upper left", fontsize=10)
titles(axL, "Latency: GB10 vs Orin Nano", "lower is better  ·  ×N = Orin / GB10 slowdown")

axR.bar(xs - w/2, g_fps, w, label="GB10", color=PALETTE["gb10"])
axR.bar(xs + w/2, j_fps, w, label="Orin Nano", color=PALETTE["orin"])
for i, (g, j) in enumerate(zip(g_fps, j_fps)):
    axR.text(i - w/2, g + max(g_fps) * 0.02, f"{g:.1f}", ha="center", va="bottom", fontsize=8.5)
    axR.text(i + w/2, j + max(g_fps) * 0.02, f"{j:.1f}", ha="center", va="bottom", fontsize=8.5)
axR.set_xticks(xs); axR.set_xticklabels(shared); axR.set_ylabel("throughput (pairs / s)")
axR.set_ylim(0, max(g_fps) * 1.15); axR.legend(loc="upper right", fontsize=10)
titles(axR, "Throughput: GB10 vs Orin Nano", "higher is better")
fig.tight_layout(rect=[0, 0, 1, 0.93], w_pad=3.0)
fig.savefig("docs/jetson_vs_gb10.png"); plt.close(fig)

print("wrote docs/jetson_latency.png + docs/jetson_vs_gb10.png")
