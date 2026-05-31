export OMP_NUM_THREADS=4,3,2

if [ -n "$PYTORCH_CUDA_ALLOC_CONF" ] && [ -z "$PYTORCH_ALLOC_CONF" ]; then
  export PYTORCH_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3
# Avoid NCCL P2P probe overhead (X79 platform, no Resizable BAR)
export NCCL_P2P_DISABLE=1
export NCCL_CUMEM_ENABLE=0

torchrun --nproc_per_node=4 ./train.py --task go2_spark_biped --headless
