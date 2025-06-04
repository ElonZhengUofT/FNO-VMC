#!/usr/bin/env bash
REMOTE_USER="user"
REMOTE_HOST="remote.server.com"
REMOTE_PROJECT_DIR="~/projects/FNO-VMC"
VENV_PATH="~/.venvs/fno-vmc"

#
WANDB_API_KEY="your_real_wandb_api_key_here"
#

ANSATZ="fno"
CONFIG_PATH="configs/fno_config.yaml"
OUTDIR="results/fno_run1"
WANDB_PROJECT="FNO-VMC"

ssh ${REMOTE_USER}@${REMOTE_HOST} << EOF
  cd ${REMOTE_PROJECT_DIR} || exit
  source ${VENV_PATH}/bin/activate
  export WANDB_API_KEY=${WANDB_API_KEY}
  nohup python scripts/train.py \
    --ansatz ${ANSATZ} \
    --config ${CONFIG_PATH} \
    --outdir ${OUTDIR} \
    --wandb_project ${WANDB_PROJECT} \
    > logs/${ANSATZ}-run.log 2>&1 &
EOF

echo "Already started remote training task: ${ANSATZ} (outdir=${OUTDIR}), logs saved to logs/${ANSATZ}-run.log"
