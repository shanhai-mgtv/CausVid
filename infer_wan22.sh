cd /mnt/cfs/shanhai/jyutong/hyc/causvid
source /mnt/cfs/shanhai/conda/bin/activate forcing 

python minimal_inference/bidirectional_inference.py \
    --config_path configs/wan22_bidirectional_dmd.yaml  \
    --output_folder ./test_results   \
    --prompt_file_path ./train_data/test.txt \
    --checkpoint_folder /mnt/cfs/shanhai/jyutong/hyc/causvid/results/2025-11-28-10-15-05.438275_seed9320884/checkpoint_model_000400
