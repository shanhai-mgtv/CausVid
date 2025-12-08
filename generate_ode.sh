cd /mnt/shanhai-ai/shanhai-workspace/jyutong/hyc/CausVid
source /opt/conda/bin/activate forcing 
export PYTHONPATH="/mnt/shanhai-ai/shanhai-workspace/jyutong/hyc/CausVid:$PYTHONPATH"

MASTER_ADDR="10.119.16.39"  # 主节点IP
MASTER_PORT="12345"
NODE_NUMS=$1
NODE_RANK=$2
logs_path=$3

torchrun --nnodes=$NODE_NUMS \
    --nproc_per_node=8 \
    --node_rank=$NODE_RANK \
    --rdzv_id=5235 \
    --rdzv_backend=c10d \
    --rdzv_endpoint $MASTER_ADDR \
    causvid/models/wan22/generate_ode_pairs.py \
    --output_folder ./ode_data \
    --caption_path train_data/vidprom_filtered_extended.txt > $logs_path 2>&1

# torchrun --nnodes=1 \
#     --nproc_per_node=1 \
#     --node_rank=0 \
#     --rdzv_id=5235 \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint localhost \
#     causvid/models/wan22/generate_ode_pairs.py \
#     --output_folder ./ode_data \
#     --caption_path train_data/vidprom_filtered_extended.txt 