GPU=0


# Preprocess (downscale datasets)
CUDA_VISIBLE_DEVICES=$GPU python scripts/downscale_dataset_blender.py \
    --config configs/blender.yml

# Training
scenes=("chair" "drums" "ficus" "hotdog" "lego" "materials" "mic" "ship")   
for scene in "${scenes[@]}";
do
    # Training
    CUDA_VISIBLE_DEVICES=$GPU python train.py \
        -m "output/blender/${scene}" --eval \
        --config configs/blender.yml

    # Rendering 
    CUDA_VISIBLE_DEVICES=$GPU python render_samename.py \
        -m "output/blender/${scene}" \
        --skip_train \
        --config configs/blender.yml

    # Metric
    CUDA_VISIBLE_DEVICES=$GPU python metrics.py \
        -m "output/blender/${scene}"
done

