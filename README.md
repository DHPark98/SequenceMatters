<div align="center">
<h2>Sequence Matters: Harnessing Video Models in 3D Super-Resolution (AAAI 2025)</h2>

<div>    
    <a href='https://scholar.google.co.kr/citations?user=lsi-8-QAAAAJ&hl=ko&oi=ao' target='_blank'>Hyun-kyu Ko</a><sup>1*</sup>&nbsp;
    <a href='https://scholar.google.co.kr/citations?user=UUtpFKgAAAAJ&hl=ko&oi=ao' target='_blank'>Dongheok Park</a><sup>1*</sup>&nbsp;
    <a target='_blank'>Youngin Park</a><sup>2</sup>&nbsp;
    <a href='https://scholar.google.co.kr/citations?user=_PhPccYAAAAJ&hl=ko&oi=ao' target='_blank'>Byeonghyeon Lee</a><sup>1</sup>&nbsp;
    <a target='_blank'>Juhee Han</a><sup>1</sup>&nbsp;
    <a href='https://silverbottlep.github.io/' target='_blank'>Eunbyung Park</a><sup>1†</sup>
</div>
    <div>
        <sup>*</sup>Equal contribution, <sup>†</sup>Corresponding author
    </div>
    <br>
<div>
    <sup>1</sup>Sungkyunkwan University, South Korea
</div>
<div>
    <sup>2</sup>Samsung Electronics, South Korea
</div>

<div>
    <h4 align="center">
        <a href="https://ko-lani.github.io/Sequence-Matters/index.html" target='_blank'>
        <img src="https://img.shields.io/badge/🍀-Project%20Page-green">
        </a>
        <a href="https://arxiv.org/abs/2401.03707" target='_blank'>
        <img src="https://img.shields.io/badge/arXiv-2401.03707-b31b1b.svg">
        </a>
        <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/DHPark98/SequenceMatters">
    </h4>
</div>

---

<div align="center">
    <h4>
        This repository is the official PyTorch implementation of "Sequence Matters: Harnessing Video Models in 3D Super-Resolution"
    </h4>
</div>
</div>

![Teaser](assets/video_comparison/bicubic_ours_chair.gif)

## :rocket: News
- **2024.12.16**: *SequenceMatters* [project page](https://ko-lani.github.io/Sequence-Matters/) release!
- **2024.12.15**: *SequenceMatters* code release!
- **2024.12.09**: *SequenceMatters* accepted to [AAAI 2025](https://aaai.org/conference/aaai/aaai-25/)!

## ✏️ Abstract
> 3D super-resolution aims to reconstruct high-fidelity 3D models from low-resolution (LR) multi-view images. Early studies primarily focused on single-image super-resolution (SISR) models to upsample LR images into high-resolution images. However, these methods often lack view consistency because they operate independently on each image. Although various post-processing techniques have been extensively explored to mitigate these inconsistencies, they have yet to fully resolve the issues. In this paper, we perform a comprehensive study of 3D super-resolution by leveraging video super-resolution (VSR) models. By utilizing VSR models, we ensure a higher degree of spatial consistency and can reference surrounding spatial information, leading to more accurate and detailed reconstructions. Our findings reveal that VSR models can perform remarkably well even on sequences that lack precise spatial alignment. Given this observation, we propose a simple yet practical approach to align LR images without involving fine-tuning or generating `smooth' trajectory from the trained 3D models over LR images. The experimental results show that the surprisingly simple algorithms can achieve the state-of-the-art results of 3D super-resolution tasks on standard benchmark datasets, such as the NeRF-synthetic and MipNeRF-360 datasets.
<p align="center">
  <img src="assets/figures/main_figure.png">
</p>


## ⚙️ Environment Setup
### Clone Git Repository
```Shell
git clone https://github.com/DHPark98/SequenceMatters.git --recursive
```

### Hardware / Software Requirements
- NVIDIA RTX3090.
- Ubuntu 18.04
- PyTorch 1.12.1 + CUDA 11.3
  
We also checked that the code run successfully with PyTorch 2.0.1 + CUDA 11.8 on Ubuntu 20.04.

### Create the Conda Environment
```Shell
conda create -n seqmat python=3.8 -y
conda activate seqmat
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

### Install Submodules and Other Dependecies
```Shell
cd SequenceMatters
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install -r requirements.txt
```

## 🔥 Run ALS
### Prepare Datasets and Pre-trained VSR Model Weights
Download the [NeRF Synthetic dataset](https://www.matthewtancik.com/nerf) or [Mip-NeRF 360 dataset](https://jonbarron.info/mipnerf360/) from their project pages, and revise ```hr_source_dir```to the dataset path, which is in the configuration file (```configs/blender.yml``` or ```configs/mip360.yml```). Download the pre-trained weights of vsr model from [PSRT](https://github.com/XPixelGroup/RethinkVSRAlignment/blob/main/README.md#training) github repository, and place them under the path below:
```
SequenceMatters
  ├─ (…)
  └─ vsr
      └─ psrt
          ├─ arch
          └─ experiments
              └─ pretrained_models
                  ├─ flownet
                  |   └─ spynet_sintel_final-3d2a1287.pth
                  ├─ PSRT_REDS.pth
                  └─ PSRT_Vimeo.pth

```

### Quick Running
You can simply excute the whole process on entire dataset of Blender or Mip-NeRF 360 Dataset.
```Shell
# Run Blender Dataset
bash scripts/run_blender.sh

# Run Mip-NeRF 360 Dataset
bash scripts/run_mip360.sh
```

### Training a Single Object / Scene
First, downsample the dataset to create LR dataset (You can use ```scripts/downscale_dataset_blender.py``` or ```scripts/downscale_dataset_mip.py```). Then, revise configuration file (```configs/blender.yml``` or ```configs/mip360.yml```), and run the code below:
```Shell
# Training a single object of Blender Dataset
python train.py \
-m <output path> --eval \
--config <path to the revised configuration file>

# Run on Mip-NeRF 360 Dataset
python train.py \
-m <output path> \
-i "images_vsr" --eval -r 1 \
--config <path to the revised configuration file>
```

<details>
<summary><span style="font-weight: bold;">Primary Command Line Arguments for Network Viewer</span></summary>

  ```hr_source_dir```
  path to directory of the HR dataset of Blender / Mip-NeRF 360 dataset.
  
  ```lr_source_dir```
  path to directory of the LR dataset to be saved.
  
  ```save_dir```
  path to directory of the LR dataset to be saved.
  
  ```downscale_factor```
  default : 4 for NeRF / 8 for Mip-NeRF 360

  ```upscale_factor```
  only defined on Mip-NeRF 360 dataset (default : 2)

  ```als```
  true : adaptive length sequences (ALS) / false : simple greedy algorithmm (S)

  ```num_images_in_sequence```
  length of sequence which are input in one vsr inference (reduce the value if you meet VRAM OOM Error)

  ```similarity```
  similarity to order sequences ( option : ['pose', 'feature'] )

  ```thres_values```
  threshold to stop generating sub-sequences in ALS


  ```subpixel```
  subpixel loss ( option : ['bicubic', 'avg'] )
  
  ```lambda_tex```:
  loss weight of 3dgs loss ( 1 - loss weight of subpixel loss )
</details>
<br>


## 📖 Reference
If you find our work useful, please consider citing:
```BibTeX
@article{ko2024sequence,
  title={Sequence Matters: Harnessing Video Models in Super-Resolution},
  author={Ko, Hyun-kyu and Park, Dongheok and Park, Youngin and Lee, Byeonghyeon and Han, Juhee and Park, Eunbyung},
  journal={arXiv preprint arXiv:2412.11525},
  year={2024}
}
```


## Acknowledgement
We express our gratitude to the following contributors, whose code provided a foundation for our work:

[3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [VRT](https://github.com/JingyunLiang/VRT), [PSRT](https://github.com/alttch/psrt), [IART](https://github.com/kai422/IART), [BasicSR](https://github.com/SwinTransformer/Video-Swin-Transformer), [mmeditin](https://github.com/open-mmlab/mmagic)
