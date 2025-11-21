export OMP_NUM_THREADS=4,3,2

if [ -n "$PYTORCH_CUDA_ALLOC_CONF" ] && [ -z "$PYTORCH_ALLOC_CONF" ]; then
  export PYTORCH_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"
fi

torchrun --nproc_per_node=2 ./train.py --task go2_spark_biped --headless
