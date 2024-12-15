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
### HyperParameters
- `target_height` (type: `List[int]` or `int`, default: `[2048, 3072, 4096]`): The height of the image being generated. If list is given, the pipeline generates corresponding intermediate resolution images in a progressive manner.
- `target_width` (type: `List[int]` or `int`, default: `[2048, 3072, 4096]`): The width of the image being generated. If list is given, the pipeline generates corresponding intermediate resolution images in a progressive manner.
- `guidance_image` (type: `torch.FloatTensor` or `PIL.Image.Image` or `np.ndarray`, default: `None`): If the guidance image is given, *DiffuseHigh* pipeline obtains structure properties from the given image, and generates desired high-resolution image.
- `noising_steps` (type: `int`, default: `15`): The number of noising steps being used in *DiffuseHigh* pipeline.
- `enable_dwt` (type: `bool`, default: `True`): Whether to use DWT-based structural guidance.
- `dwt_steps` (type: `int`, default: `5`): The number of structural guidance steps during the denoising process. Typically, we found that 5 ~ 7 steps works well.
- `dwt_level` (type: `int`, default: `1`): The DWT level of our proposed structural guidance.
- `dwt_wave` (type: `str`, default: `'db4'`): Which wavelet to use for the DWT.
- `dwt_mode` (type: `str`, default: `'symmetric'`): Padding scheme for the DWT.
- `enable_sharpening` (type: `bool`, default: `True`): Whether to use sharpening operation in *DiffuseHigh* pipeline.
- `sharpening_kernel_size` (type: `int`, default: `3`): Kernel size for the Gaussian blur involved in sharpening operation.
- `sharpening_simga` (type: `tuple` or `float`, default: `(0.1, 2.0)`): Standard deviation to be used for creating kernel to perform blurring. If float, sigma is fixed. If it is tuple of float (min, max), sigma is chosen uniformly at random to lie in the given range.
- `sharpening_alpha` (type: `float`, default: `1.0`): The sharpeness factor for controling the strength of the sharpening operation.

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
