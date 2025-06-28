#!/usr/bin/env bash

ssh -X shizhao@131.215.142.161 << 'EOF'
  cd Projects
  cd FNO-VMC
  git pull                    # 拉最新代码
  python -m src.scripts.train \
    --ansatz Slater \
    --config configs/two_dim_hubbard_fno_4.yaml \
    --outdir resul/fno_run \
    --logfile logs/fno_run.log
EOF

ssh -X shizhao@131.215.142.161