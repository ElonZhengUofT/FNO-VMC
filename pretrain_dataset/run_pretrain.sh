#!/usr/bin/env bash

ssh -X shizhao@131.215.142.161 << 'EOF'
  source ~/miniconda/etc/profile.d/conda.sh
  conda activate fnovmc
  cd Projects/FNO-VMC
  git pull
  export XLA_FLAGS="--xla_gpu_autotune_level=2"
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7
  # export XLA_FLAGS=--xla_gpu_enable_command_buffer=false

  python -m src.scripts.pretrain_generate \
    --outdir pretrain_dataset \
    --logfile logs/fno_run.log

EOF

ssh -X shizhao@131.215.142.161