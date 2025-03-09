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
    input_base_dir = args.hr_source_dir
    output_base_dir = args.lr_source_dir
    downscale_factor = args.downscale_factor
    upscale_factor = args.upscale_factor

    scenes = ["bicycle", "bonsai", "counter", "garden", "kitchen", "room", "stump", "flowers", "treehill"]
    scenes = ["flowers", "treehill"]
    splits = ["images"]

    # Iterate through all subdirectories and process images
    for scene in scenes:
        input_dir = os.path.join(input_base_dir, scene)
        output_dir = os.path.join(output_base_dir, scene)

        shutil.copytree(os.path.join(input_dir, "sparse"), os.path.join(output_dir, "sparse"), dirs_exist_ok=True)
        shutil.copy2(os.path.join(input_dir, "poses_bounds.npy"), os.path.join(output_dir, "poses_bounds.npy"))

        for split in splits:
            split_input_dir = os.path.join(input_dir, split)
            images_down_dir = os.path.join(output_dir, "images_down")
            images_gt_dir = os.path.join(output_dir, "images_gt")

            # Create output directories if they do not exist
            if not os.path.exists(images_down_dir):
                os.makedirs(images_down_dir)
            if not os.path.exists(images_gt_dir):
                os.makedirs(images_gt_dir)

            if not os.path.exists(split_input_dir):
                print(f"Directory not found: {split_input_dir}")
                continue

            # Process image files in the input directory
            for filename in tqdm(os.listdir(split_input_dir), desc=f"Processing {scene}/{split}"):
                input_path = os.path.join(split_input_dir, filename)
                output_path_down = os.path.join(images_down_dir, os.path.splitext(filename)[0] + ".png")
                output_path_gt = os.path.join(images_gt_dir, os.path.splitext(filename)[0] + ".png")

                try:
                    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
                    img = img / 255.0  # Normalize to [0, 1]
                    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()  # (H, W, C) -> (1, C, H, W)
                    hr_h, hr_w = img.shape[2:]  # Get height and width

                    # Resize the image with downscale_factor (W//8, H//8)
                    lr_x8_h = round(hr_h / 8)
                    lr_x8_w = round(hr_w / 8)

                    resized_img_down = F.interpolate(img, size=(lr_x8_h, lr_x8_w), mode="bicubic", align_corners=False, antialias=True)
                    resized_img_down = (resized_img_down.squeeze(0).permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                    cv2.imwrite(output_path_down, resized_img_down)

                    # Resize the image further to (W//8 * 4, H//8 * 4)
                    lr_x2_h = lr_x8_h * 4
                    lr_x2_w = lr_x8_w * 4

                    resized_img_gt = F.interpolate(img, size=(lr_x2_h, lr_x2_w), mode="bicubic", align_corners=False, antialias=True)
                    resized_img_gt = (resized_img_gt.squeeze(0).permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                    cv2.imwrite(output_path_gt, resized_img_gt)

                except Exception as e:
                    print(f"Error processing {input_path}: {e}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Downscale HR dataset to LR dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration YAML file")
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    args = Namespace(**vars(args), **config)


    main(args)