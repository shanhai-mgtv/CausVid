export NCCL_TIMEOUT=3600
export NCCL_RETRIES=10
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TOKENIZERS_PARALLELISM=false
export WANDB_HOST="https://api.wandb.ai"
export WANDB_KEY="f3d8d5b28440d6f54efc163fa508c2747239b997"

MASTER_ADDR="10.192.0.13"  # 主节点IP
MASTER_PORT="12345"
NODE_NUMS=$1
NODE_RANK=$2
logs_path=$3

cd /mnt/cfs/shanhai/jyutong/hyc/causvid
source /mnt/cfs/shanhai/conda/bin/activate forcing 

torchrun --nnodes $NODE_NUMS \
    --nproc_per_node=8 \
    --node_rank=$NODE_RANK \
    --rdzv_id=5235 \
    --rdzv_backend=c10d \
    --rdzv_endpoint $MASTER_ADDR \
    causvid/train_distillation.py \
    --config_path  configs/wan22_bidirectional_dmd.yaml > $logs_path 2>&1




