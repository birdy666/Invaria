<div align="center">
<h1>Invaria: Learning Scale and Density Invariance in Point Clouds via Next-Resolution Prediction</h1>

[**Chun-Peng Chang**](https://chunpeng-chang.github.io/)<sup>1</sup>, 
[**Shaoxiang Wang**](https://shaoxiang777.github.io/)<sup>2</sup>, 
[**Alain Pagani**](https://www.dfki.de/en/web/about-us/employee/person/alpa02)<sup>2</sup>,
[**Dariu Gavrila**](https://www.tudelft.nl/en/staff/d.m.gavrila/)<sup>1</sup>,
[**Holger Caesar**](https://sites.google.com/it-caesar.de/homepage/)<sup>1</sup>

<sup>1</sup>TU Delft,
<sup>2</sup>DFKI-AV

**NeurIPS 2026**


<a href="https://arxiv.org/abs/2605.15923"><img src="https://img.shields.io/badge/arXiv-2605.15923-red?logo=arxiv" alt="arXiv"></a>
<a href="https://birdy666.github.io/projects/invaria/"><img src="https://img.shields.io/badge/Project_Page-Invaria-green" alt="Project Page"></a>
<a href="https://huggingface.co/birdy666/Invaria"><img src="https://img.shields.io/badge/Hugging_Face-Checkpoints-yellow?logo=huggingface" alt="HF Model"></a>



</div>



## News
- **2026-09-24:** Invaria is accepted to NeurIPS 2026!

## Installation & Data Preparation
Invaria is built on top of [Pointcept](https://github.com/Pointcept/Pointcept). Please follow their [installation guide](https://github.com/Pointcept/Pointcept#installation) to set up the environment, and their [data preparation guide](https://github.com/Pointcept/Pointcept#data-preparation) for dataset preprocessing; we use the same preprocessing pipeline.

## Training & Evaluation
Configs are in `configs/invaria/`. Outputs are saved to `exp/invaria/<exp_name>/`.

**Training**
```bash
# ScanNet
sh scripts/train.sh -g 4 -d invaria -c semseg-invaria-single-stage-scannet -n invaria-scannet
# nuScenes
sh scripts/train.sh -g 4 -d invaria -c semseg-invaria-single-stage-nuscenes -n invaria-nuscenes
```
Add `-r true` to resume an interrupted run.

**Evaluation**
```bash
# evaluate a run trained with the commands above
sh scripts/test.sh -g 1 -d invaria -n invaria-scannet -w model_best
sh scripts/test.sh -g 1 -d invaria -n invaria-nuscenes -w model_best

# download the released checkpoints from Hugging Face
hf download birdy666/Invaria --local-dir weights

# evaluate a released checkpoint (ScanNet)
PYTHONPATH=./ python tools/test.py \
  --config-file configs/invaria/semseg-invaria-single-stage-scannet.py \
  --options save_path=exp/invaria/scannet-eval weight=weights/scannet/invaria.pth

# evaluate a released checkpoint (nuScenes)
PYTHONPATH=./ python tools/test.py \
  --config-file configs/invaria/semseg-invaria-single-stage-nuscenes.py \
  --options save_path=exp/invaria/nuscenes-eval weight=weights/nuscenes/invaria.pth
```
Checkpoints are hosted on [Hugging Face](https://huggingface.co/birdy666/Invaria).


## Acknowledgment
The project is built partially based on [Pointcept](https://github.com/Pointcept/Pointcept). We thank them for their great work!

## Citation

If you find this project useful, please cite:
```
@article{chang2026invaria,
  title={Invaria: Learning Scale and Density Invariance in Point Clouds via Next-Resolution Prediction},
  author={Chang, Chun-Peng and Wang, Shaoxiang and Pagani, Alain and Gavrila, Dariu and Caesar, Holger},
  journal={arXiv preprint arXiv:2605.15923},
  year={2026}
}
```
