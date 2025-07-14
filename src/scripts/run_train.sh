#!/usr/bin/env bash

ssh -X shizhao@131.215.142.161 << 'EOF'
  source ~/miniconda/etc/profile.d/conda.sh
  conda activate fnovmc
  cd Projects/FNO-VMC
  git pull
  export XLA_FLAGS="--xla_gpu_autotune_level=2"
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7
  # export XLA_FLAGS=--xla_gpu_enable_command_buffer=false

  # 用 cProfile 收集整个脚本的性能数据，输出到 prof.out
  # python -m cProfile -o prof.out src/scripts/train.py \
  python -m src.scripts.train \
    --ansatz backflow \
    --config configs/one_dim_hubbard_fno_16.yaml \
    --outdir resul/fno_run \
    --logfile logs/fno_run.log

  # （可选）把 prof.out 打包一下方便下载
  # tar czf prof.tgz prof.out
EOF

ssh -X shizhao@131.215.142.161