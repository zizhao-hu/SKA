#!/bin/bash
NUM_PROC=2
RDZV_HOST=$(hostname)
RDZV_PORT=29401

shift 
srun --export=ALL torchrun \
    --nproc_per_node=$NUM_PROC --nnodes=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$RDZV_HOST:$RDZV_PORT" train.py "$@"
    

