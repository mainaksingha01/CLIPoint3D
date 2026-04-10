export CUDA_VISIBLE_DEVICES=0

source_domains=("shapenet" "modelnet" "scannet")
target_domains=("shapenet" "modelnet" "scannet")

for source in "${source_domains[@]}"; do
    for target in "${target_domains[@]}"; do
        if [[ "$source" != "$target" ]]; then
            dataset_config="configs/datasets/pointda_${source}_${target}.yaml"
            python train.py --config-file configs/trainers/trainer_200.yaml \
                            --output-dir "test_uncertain_without_target_test_a100" \
                            --dataset-config-file "$dataset_config" --seed 42 \
                            --use_sinkhorn_loss --use_entropy_loss  --use_confidence_sampling
    done
done