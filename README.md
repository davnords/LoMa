<div align="center">
<h1>LoMa: Local Feature Matching Revisited</h1>


<a href="https://arxiv.org/abs/2604.04931"><img src="https://img.shields.io/badge/arXiv-2604.04931-b31b1b" alt="arXiv"></a>
<a href="https://www.davnords.com/loma"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>

**Chalmers University of Technology**; **Linköping University**; **University of Amsterdam**; **Lund University**

[David Nordström*](https://scholar.google.com/citations?user=-vJPE04AAAAJ), [Johan Edstedt*](https://scholar.google.com/citations?user=Ul-vMR0AAAAJ&hl), [Georg Bökman](https://scholar.google.com/citations?user=FUE3Wd0AAAAJ), [Jonathan Astermark](https://scholar.google.com/citations?user=dsEPAvUAAAAJ), [Anders Heyden](https://scholar.google.com/citations?user=9j-6i_oAAAAJ), [Viktor Larsson](https://scholar.google.com/citations?user=vHeD0TYAAAAJ), [Mårten Wadenbäck](https://scholar.google.com/citations?user=6WRQpCQAAAAJ), [Michael Felsberg](https://scholar.google.com/citations?user=lkWfR08AAAAJ), [Fredrik Kahl](https://scholar.google.com/citations?user=P_w6UgMAAAAJ)
</div>

<p align="center">
    <img src="assets/loma.jpg" alt="example" width=45%>
    <br>
    <em>Performance on a difficult matching pair compared to LightGlue.</em>
</p>

# ⚠️ WARNING: This branch is under development. Use at your own discretion.
Disclaimer: Training reproduction has not been validated but the code closely follows the original training code. To be cleaned up futher.

## Overview
LoMa is a fast and accurate family of local feature matchers. It works similar to [LightGlue](https://github.com/cvg/LightGlue) but significantly improves matching robustness and accuracy across benchmarks, even outperforming [RoMa](https://github.com/Parskatt/RoMa) and [RoMa v2](https://github.com/Parskatt/RoMaV2) on the difficult [WxBS](https://arxiv.org/abs/1504.06603) benchmark. As LoMa leverages local keypoint descriptions, the models are perfect drop-in replacement in e.g. SfM and Visual Localization pipelines.

## Updates
- [April 14, 2026] Rotation invariant LoMa released. The model, which we call LoMa-R, is great at aerial imagery (e.g. [SatAst](https://github.com/georg-bn/satast)). See the paper [Who Handles Orientation?](https://arxiv.org/abs/2604.11809) (CVPRW26) for more information.
- [April 13, 2026] Integration available with [HLoc](https://github.com/davnords/Hierarchical-Localization) and [vismatch](https://github.com/gmberton/vismatch/pull/63).
- [April 6, 2026] LoMa inference code released. 

## How to Use
```python
import cv2
from loma import LoMa, LoMaB

# load pretrained model
model = LoMa(LoMaB())  # also available: LoMaB128, LoMaL, LoMaG, LoMaR
# Define image paths, e.g.
img_A_path, img_B_path = "assets/0015_A.jpg", "assets/0015_B.jpg"
# Extract matching keypoints in image coordinates
kptsA, kptsB = model.match(img_A_path, img_B_path)

# Find a fundamental matrix (or anything else of interest)
F, mask = cv2.findFundamentalMat(
    kptsA, kptsB, ransacReprojThreshold=0.2, method=cv2.USAC_MAGSAC, confidence=0.999999, maxIters=10000
)
```
We provide additional code examples in [demo.py](demo.py), which might help in understanding. To run the demo, use the following API:
```bash
uv run demo.py matcher:loma-b
```

## Setup/Install
In your python environment (tested on Linux python 3.12), run:
```bash
uv pip install -e .
```
or 
```bash
uv sync
```

## Benchmarks
We initially provide code for evaluating on MegaDepth, ScanNet, WxBS and RUBIK. If you do not already have MegaDepth1500 and ScanNet1500, you may run the following to download them:
```bash
source scripts/eval_prep.sh
```
To run a benchmark you need to install the optional dependencies by e.g. `uv sync --extra eval`. Thereafter, you can use the following call signature:
```bash
uv run eval.py matcher:loma-b --benchmark wxbs
```
Use `uv run eval.py --help` to explore the different options. 

### Expected Results
The results are similar to those reported in the paper. For example, running the evaluation for LoMa-B on WxBS gives us `mAA_10px: 0.6876`.

## Sizes
We an array of models: LoMA-{B, B128, L, G, R}. For most usecases LoMa-B, which is the same size as LightGlue, works fine. LoMa-G is significantly heavier but gives the most accurate matches, even surpassing the RoMa-family on e.g. WxBS and IMC22. LoMa-R provides a rotation invariant matcher and descriptor (through data augmentation).

## Training

### Data
We initially provide a training script on MegaDepth. We use the same standard data pre-processing as in [RoMa](https://github.com/Parskatt/RoMa) and [DKM](https://github.com/parskatt/dkm). The easiest way to obtain the data is to follow the instructions in those repos. We expect the data to be in the `data/` folder. If stored elsewhere, symlink it to there.

The full LoMa model is trained on: [ScanNet++](https://github.com/scannetpp/scannetpp), [BlendedMVS](https://github.com/yoyo000/blendedmvs), [Map-Free](https://github.com/nianticlabs/map-free-reloc), [Hypersim](https://github.com/apple/ml-hypersim), [MegaScenes](https://megascenes.github.io/), [MegaDepth](https://github.com/zhengqili/MegaDepth), [AerialMD](https://github.com/kvuong2711/aerial-megadepth), [TartanAir v2](https://tartanair.org/), [Mapillary Planet-scale](https://www.mapillary.com/dataset/depth), [Aria Synthetic Environments](https://www.projectaria.com/datasets/ase/), [CO3Dv2](https://github.com/facebookresearch/co3d), [MegaSynth](https://hwjiang1510.github.io/MegaSynth/), [SpatialVID](https://github.com/NJU-3DV/SpatialVID), [FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html), [UnrealStereo4k](https://huggingface.co/datasets/fabiotosi92/UnrealStereo4K), and [Virtual KITTI 2](https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/).

### Start
LoMa is trained in two stages: (i) descriptor and (ii) matcher. To finetune LoMa, only the matcher needs to be trained. To train LoMa from scratch you should first train the descriptor:
```bash
uv run torchrun --nproc_per_node=8 train/descriptor.py --name loma-desc
```
Thereafter, you train the matcher on-top of your already trained descriptor:
```bash
uv run torchrun --nproc_per_node=8 train/matcher.py --name loma-match --model.descriptor_run_path <path to your descriptor run>
```
You can use `--help` on the training scripts to find the customizable parameters. The defaults are what we used to train LoMa-B. The training scripts support distributed training (including multi-node). Non-distributed training also works, simply run `uv run train/<script.py>`. However, this will not reproduce our results.

To train LoMa-R you can simply turn on rotation data augmentation by adding the flag `--transform.rot360`. To train other sizes of LoMa simply change the embed dimension and number of heads. We use a constant head dimension of 64 and embed dimensions of 256, 512, and 1024 for LoMa-B, LoMa-L, and LoMa-G, respectively.

## Checklist
- [x] Publish the inference code.
- [x] Release rotation invariant matcher.
- [x] Integrate with [HLoc](https://github.com/cvg/Hierarchical-Localization?tab=readme-ov-file). See this [fork](https://github.com/davnords/Hierarchical-Localization).
- [x] Integrate with [vismatch](https://github.com/gmberton/vismatch). See this [PR](https://github.com/gmberton/vismatch/pull/63).
- [x] Provide training code.
- [ ] Release a lightweight descriptor.
- [ ] Release HardMatch.

## License
All our code except the matcher, which inherits its license from LightGlue, is MIT license. LightGlue has an [Apache-2.0](https://github.com/cvg/LightGlue/blob/main/LICENSE) license.

## Acknowledgement
Thanks to [Parskatt](https://github.com/Parskatt) for writing most of the code. Our codebase structure is mainly based on [RoMaV2](https://github.com/Parskatt/RoMaV2) and our architectures build on [LightGlue](https://github.com/cvg/lightglue), [DeDoDe](https://github.com/Parskatt/DeDoDe), and [DaD](https://github.com/Parskatt/dad). 

## BibTeX
If you find our models useful, please consider citing our papers!
```bibtex
@misc{nordström2026lomalocalfeaturematching,
      title={LoMa: Local Feature Matching Revisited}, 
      author={David Nordström and Johan Edstedt and Georg Bökman and Jonathan Astermark and Anders Heyden and Viktor Larsson and Mårten Wadenbäck and Michael Felsberg and Fredrik Kahl},
      year={2026},
      eprint={2604.04931},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.04931}, 
}

@inproceedings{nordstrom2026who,
  title={Who Handles Orientation? Investigating Invariance in Feature Matching},
  author={David Nordström and Johan Edstedt and Georg Bökman and Fredrik Kahl},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  year={2026}
}
```
