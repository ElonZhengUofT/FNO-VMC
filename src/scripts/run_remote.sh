#!/usr/bin/env bash

ssh -X r04vufj4hus63j-64411b66@ssh.runpod.io -i ~/.ssh/id_ed25519 << 'EOF'
  # source ~/miniconda/etc/profile.d/conda.sh
  # conda activate fnovmc
  cd home
  cd FNO-VMC
  git pull
  export XLA_FLAGS="--xla_gpu_autotune_level=2"
  python -m src.scripts.train \
    --ansatz backflow \
    --config configs/one_dim_hubbard_fno_16.yaml \
    --outdir resul/fno_run \
    --logfile logs/fno_run.log
EOF

ssh -X shizhao@131.215.142.161