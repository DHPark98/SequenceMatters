import os
import shutil
import numpy as np
import cv2
from tqdm import tqdm
import yaml
from argparse import ArgumentParser, Namespace
import torch
import torch.nn.functional as F


def main(args):
    if args.white_background:
        background_color = (255, 255, 255) 
    else:
        background_color = (0, 0, 0)
    
    downscale_factor = args.downscale_factor
    input_base_dir = args.hr_source_dir
    output_base_dir = args.lr_source_dir

    scenes = ['chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship']
    splits = ["train", "test", "val"]

    # Iterate through all subdirectories and process images
    for scene in scenes:
        input_dir = os.path.join(input_base_dir, scene)
        output_dir = os.path.join(output_base_dir, scene)

        # Create subdirectory if it does not exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for split in splits:
            split_input_dir = os.path.join(input_dir, split)
            split_output_dir = os.path.join(output_dir, split)

            if not os.path.exists(split_input_dir):
                print(f"Directory not found: {split_input_dir}")
                continue

            # Create output directory if it does not exist
            if not os.path.exists(split_output_dir):
                os.makedirs(split_output_dir)

            # Process image files in the input directory
            for filename in tqdm(os.listdir(split_input_dir), desc=f"Processing {scene}/{split}"):
                input_path = os.path.join(split_input_dir, filename)
                output_path = os.path.join(split_output_dir, filename)

                try:
                    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
                    img = img / 255.0

                    if img.shape[2] == 4:
                        alpha = img[..., 3]
                        img = alpha[..., None] * img[..., :3] + (1. - alpha)[..., None] * background_color

                    # Resize the image with antialiasing
                    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()  # (H, W, C) -> (1, C, H, W)
                    resized_img = F.interpolate(img, scale_factor=1/downscale_factor, mode="bicubic", align_corners=False, antialias=True)
                    resized_img = (resized_img.squeeze(0).permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

                    # Save the resized image
                    cv2.imwrite(output_path, resized_img)

                except Exception as e:
                    print(f"Error processing {input_path}: {e}")

        # Copy files that are not part of sub-subdirectories
        for filename in os.listdir(input_dir):
            input_file_path = os.path.join(input_dir, filename)
            output_file_path = os.path.join(output_dir, filename)

            if os.path.isfile(input_file_path) and filename not in splits:
                try:
                    shutil.copy2(input_file_path, output_file_path)
                except Exception as e:
                    print(f"Error copying file {input_file_path}: {e}")



if __name__ == "__main__":
    parser = ArgumentParser(description="Downscale HR dataset to LR dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration YAML file")
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    args = Namespace(**vars(args), **config)

    main(args)


'''

python /hdd/leao8869/aaai2025_sequence/SequenceMatters/scripts/downscale_dataset_blender.py \
--config /hdd/leao8869/aaai2025_sequence/SequenceMatters/configs/blender.yml

'''