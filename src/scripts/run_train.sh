#!/usr/bin/env bash
ssh -X shizhao@131.215.142.161 << 'EOF'
  git pull
  source ~/miniconda/etc/profile.d/conda.sh
  conda activate fnovmc
  cd Projects
  cd FNO-VMC




ssh -X shizhao@131.215.142.161 << 'EOF'
  source ~/miniconda/etc/profile.d/conda.sh
  conda activate fnovmc
  cd Projects
  cd FNO-VMC
  git pull
  export XLA_FLAGS="--xla_gpu_autotune_level=2"
  export XLA_PYTHON_CLIENT_MEM_FRACTION=false
  python -m src.scripts.train \
    --ansatz backflow \
    --config configs/one_dim_hubbard_fno_16.yaml \
    --outdir resul/fno_run \
    --logfile logs/fno_run.log
EOF