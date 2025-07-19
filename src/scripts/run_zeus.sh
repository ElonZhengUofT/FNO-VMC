#!/usr/bin/env bash
# ssh -i ~/.ssh/id_ed25519_remote shizhao@zeus.dgp.toronto.edu
# ssh zeus.dgp.toronto.edu
# ssh zeus

ssh belle << 'EOF'
  # source ~/miniconda/etc/profile.d/conda.sh
  # conda activate fnovmc
  cd FNO-VMC
  git pull
  python3 -m src.scripts.run_memory_check
  export XLA_FLAGS="--xla_gpu_autotune_level=2"
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7

  python3 -m src.scripts.pretrain_generate \
    --outdir pretrain_dataset \
    --logfile logs/fno_run.log
EOF

ssh belle