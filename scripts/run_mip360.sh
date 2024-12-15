GPU=0


# Preprocess (downscale datasets)
CUDA_VISIBLE_DEVICES=$GPU python scripts/downscale_dataset_mip360.py \
--config configs/mip360.yml


# Run Gaussian Splatting
scenes=(bicycle bonsai counter garden kitchen room stump flowers treehill)
for scene in "${scenes[@]}";
do
    # Training
    CUDA_VISIBLE_DEVICES=$GPU python train.py \
        -m "output/mip360/${scene}" \
        -i "images_vsr" --eval -r 1 \
        --config configs/mip360.yml

    # Rendering 
    CUDA_VISIBLE_DEVICES=$GPU python render_samename.py \
        -m "output/mip360/${scene}" \
        -i "images_gt" --eval -r 1 --skip_train \
        --config configs/mip360.yml

    # Metric
    CUDA_VISIBLE_DEVICES=$GPU python metrics.py \
        -m "output/mip360/${scene}"
done

