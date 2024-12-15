import os
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
import yaml
from argparse import ArgumentParser, Namespace



def main(args):
    input_base_dir = args.hr_source_dir
    output_base_dir = args.lr_source_dir
    downscale_factor = args.downscale_factor
    upscale_factor = args.upscale_factor

    scenes = ["bicycle", "bonsai", "counter", "garden", "kitchen", "room", "stump", "flowers", "room"]
    scenes = ["bicycle"]
    splits = ["images"]

    # Iterate through all subdirectories and process images
    for scene in scenes:
        input_dir = os.path.join(input_base_dir, scene)
        output_dir = os.path.join(output_base_dir, scene)

        shutil.copytree(os.path.join(input_dir, "sparse"), os.path.join(output_dir, "sparse"), dirs_exist_ok=True)
        shutil.copy2(os.path.join(input_dir, "poses_bounds.npy"), os.path.join(output_dir, "poses_bounds.npy"))

        # for split in splits:
        #     split_input_dir = os.path.join(input_dir, split)
        #     images_down_dir = os.path.join(output_dir, "images_down")
        #     images_gt_dir = os.path.join(output_dir, "images_gt")

        #     # Create output directories if they do not exist
        #     if not os.path.exists(images_down_dir):
        #         os.makedirs(images_down_dir)
        #     if not os.path.exists(images_gt_dir):
        #         os.makedirs(images_gt_dir)

        #     if not os.path.exists(split_input_dir):
        #         print(f"Directory not found: {split_input_dir}")
        #         continue

        #     # Process image files in the input directory
        #     for filename in tqdm(os.listdir(split_input_dir), desc=f"Processing {scene}/{split}"):
        #         input_path = os.path.join(split_input_dir, filename)
        #         output_path_down = os.path.join(images_down_dir, os.path.splitext(filename)[0] + ".png")
        #         output_path_gt = os.path.join(images_gt_dir, os.path.splitext(filename)[0] + ".png")

        #         try:
        #             with Image.open(input_path) as img:
        #                 # Handle images with alpha channel
        #                 W, H = img.size
        #                 # Resize the image with downscale_factor (W//8, H//8)
        #                 W_lr, H_lr = round(W / downscale_factor), round(H / downscale_factor)
        #                 resized_img_down = img.resize((W_lr, H_lr), Image.BICUBIC)
        #                 resized_img_down.save(output_path_down, format="PNG")

        #                 # Resize the image further to (W//8 * 4, H//8 * 4)
        #                 resized_img_gt = img.resize(
        #                     (W_lr * upscale_factor, H_lr * upscale_factor), 
        #                     Image.BICUBIC
        #                 )
        #                 resized_img_gt.save(output_path_gt, format="PNG")

        #         except Exception as e:
        #             print(f"Error processing {input_path}: {e}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Downscale HR dataset to LR dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration YAML file")
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    args = Namespace(**vars(args), **config)


    main(args)