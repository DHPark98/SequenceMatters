import os
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
import yaml
from argparse import ArgumentParser, Namespace


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
                    with Image.open(input_path) as img:
                        # Handle images with alpha channel
                        if img.mode == "RGBA":
                            # Convert RGBA to Numpy Array
                            rgba = np.array(img)
                            rgb = np.zeros((rgba.shape[0], rgba.shape[1], 3), dtype=np.uint8)

                            # Separate alpha channel
                            r, g, b, a = rgba[..., 0], rgba[..., 1], rgba[..., 2], rgba[..., 3]

                            # Perform alpha blending
                            rgb[..., 0] = r * (a / 255) + background_color[0] * (1 - a / 255)
                            rgb[..., 1] = g * (a / 255) + background_color[1] * (1 - a / 255)
                            rgb[..., 2] = b * (a / 255) + background_color[2] * (1 - a / 255)

                            # Convert RGB array back to an image
                            img = Image.fromarray(rgb, mode="RGB")

                        W, H = img.size
                        # Resize the image with antialiasing
                        resized_img = img.resize((W//downscale_factor, H//downscale_factor), Image.BICUBIC)

                        # Save the resized image
                        resized_img.save(output_path)

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