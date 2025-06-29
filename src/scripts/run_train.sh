#!/usr/bin/env bash

ssh -X shizhao@131.215.142.161 << 'EOF'
  source ~/miniconda/etc/profile.d/conda.sh
  conda activate fnovmc
  cd Projects
  cd FNO-VMC
  git pull
  python -m src.scripts.train \
    --ansatz fno \
    --config configs/two_dim_hubbard_fno_4.yaml \
    --outdir resul/fno_run \
    --logfile logs/fno_run.log
EOF

ssh -X shizhao@131.215.142.161