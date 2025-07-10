#!/usr/bin/env bash
ssh -X shizhao@131.215.142.161 << 'EOF'
  source ~/miniconda/etc/profile.d/conda.sh
  conda activate fnovmc
  cd Projects
  cd FNO-VMC


