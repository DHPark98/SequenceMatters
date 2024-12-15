# Sequence Matters: Harnessing Video Model in 3D Super-Resolution

[![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://github.com/DHPark98/SequenceMatters/)
[![arXiv](https://img.shields.io/badge/arXiv-2311.16973-b31b1b.svg)](https://github.com/DHPark98/SequenceMatters)

Official github for "Sequence Matters: Harnessing Video Model in 3D Super-Resolution"

<img src="assetsfigures/main.jpg">


## Abstract
3D super-resolution aims to reconstruct high-fidelity 3D models from low-resolution (LR) multi-view images. Early studies primarily focused on single-image super-resolution (SISR) models to upsample LR images into high-resolution images. However, these methods often lack view consistency because they operate independently on each image. Although various post-processing techniques have been extensively explored to mitigate these inconsistencies, they have yet to fully resolve the issues. In this paper, we perform a comprehensive study of 3D super-resolution by leveraging video super-resolution (VSR) models. By utilizing VSR models, we ensure a higher degree of spatial consistency and can reference surrounding spatial information, leading to more accurate and detailed reconstructions. Our findings reveal that VSR models can perform remarkably well even on sequences that lack precise spatial alignment. Given this observation, we propose a simple yet practical approach to align LR images without involving fine-tuning or generating `smooth' trajectory from the trained 3D models over LR images. The experimental results show that the surprisingly simple algorithms can achieve the state-of-the-art results of 3D super-resolution tasks on standard benchmark datasets, such as the NeRF-synthetic and MipNeRF-360 datasets.

## Environment Setup
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
Download the [NeRF dataset](https://www.matthewtancik.com/nerf) or [Mip-NeRF 360 dataset](https://jonbarron.info/mipnerf360/) from their project pages, and revise ```hr_source_dir```to the dataset path, which is in the configuration file (```configs/blender.yml``` or ```configs/mip360.yml```). Download the pre-trained weights of vsr model from [PSRT](https://github.com/XPixelGroup/RethinkVSRAlignment/blob/main/README.md#training) github repository, and place them under the path below:
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
                  └─ [**PSRT_Vimeo.pth**](#)

```

### How to use
You can easily import the DiffuseHighSDXLPipeline from our provided code below.

For example, you can generate 2K image via below code.
```Python
from pipeline_diffusehigh_sdxl import DiffuseHighSDXLPipeline
pipeline = DiffuseHighSDXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16,
).to("cuda")

negative_prompt = "blurry, ugly, duplicate, poorly drawn, deformed, mosaic"
prompt = "A baby bunny sitting on a stack of pancakes."

image = pipeline(
        prompt,
        negative_prompt=negative_prompt,
        target_height=[1536, 2048],
        target_width=[1536, 2048],
        enable_dwt=True,
        dwt_steps=5,
        enable_sharpening=True,
        sharpness_factor=1.0,
    ).images[0]

image.save("sample_bunny_2K.png")
```
result:
<img src="figures/sample_bunny_2K.png">


For 4K image generation, try below code!
```Python
from pipeline_diffusehigh_sdxl import DiffuseHighSDXLPipeline
pipeline = DiffuseHighSDXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16,
).to("cuda")

negative_prompt = "blurry, ugly, duplicate, poorly drawn, deformed, mosaic"
prompt = "Cinematic photo of delicious chocolate icecream."

image = pipeline(
        prompt,
        negative_prompt=negative_prompt,
        target_height=[2048, 3072, 4096],
        target_width=[2048, 3072, 4096],
        enable_dwt=True,
        dwt_steps=5,
        enable_sharpening=True,
        sharpness_factor=1.0,
    ).images[0]

image.save("sample_icecream_4K.png")
```

result:
<img src="figures/sample_icecream_4K.png">

Also try with "DSLR shot of" or "Photorealistic picture of" -based prompts for photorealistic samples!
```Python
from pipeline_diffusehigh_sdxl import DiffuseHighSDXLPipeline
pipeline = DiffuseHighSDXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16,
).to("cuda")

negative_prompt = "blurry, ugly, duplicate, poorly drawn, deformed, mosaic"
prompt = "A DSLR shot of fresh strawberries in a ceramic bowl, with tiny water droplets on the fruit, highly detailed, sharp focus, photo-realistic, 8K."

image = pipeline(
        prompt,
        negative_prompt=negative_prompt,
        target_height=[2048, 3072, 4096],
        target_width=[2048, 3072, 4096],
        enable_dwt=True,
        dwt_steps=5,
        enable_sharpening=True,
        sharpness_factor=1.0,
    ).images[0]

image.save("sample_DSLR_strawberry_4K.png")
```

result:
<img src="figures/sample_DSLR_strawberry_4K.png">

If the result image has undesirable structural properties, you can adjust `dwt_steps` argument to little more higher value, e.g., `dwt_steps=7`. If the result image still seems blurry, try higher `sharpness_factor` argument value, e.g., `sharpness_factor=2.0`.
