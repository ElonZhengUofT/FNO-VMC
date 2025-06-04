#!/usr/bin/env bash
#
# run_on_pod.sh
# This script is designed to run train.py in the /workspace/FNO-VMC directory of a Docker container,

export PYTHONUNBUFFERED=1

#!/usr/bin/env bash
export PYTHONPATH=/workspace/FNO-VMC:$PYTHONPATH
# 2. Change to the project root directory (make sure this path matches your actual path in the Pod)
# cd /workspace/FNO-VMC || {
  #  echo "Error: 无法切换到 /workspace/FNO-VMC，请确认路径是否正确"
  #  exit 1
  #}

# 3. （可选）如果你的镜像里有 conda 环境，需要先激活：
#    例如镜像自带名为 'base' 或者 'pytorch' 的 conda 环境，就写：
# source /opt/conda/bin/activate pytorch
# 或者：
# source ~/.bashrc
# conda activate pytorch
#
# 如果镜像里没有 conda，或者你打算用系统预装的 python3，跳过这一行。

# 4. 用 Python3 运行 train.py
#    - 如果容器里 /usr/bin/python3 就是你要的解释器，可以直接用 python3。
#    - 如果需要指定特定路径，比如 /opt/conda/bin/python，则把下面 python3 换成该路径。

python3 src/scripts/train.py jlts/fno_run \
    --logfile logs/fno_run.log \
    --wandb_project FNO-VMC

# 如果想后台运行，请改成下面这一行（并注释掉上面那个命令）：
# nohup python3 src/scripts/train.py jlts/fno_run \
#      --logfile logs/fno_run.log \
#      --wandb_project FNO-VMC \
#      > nohup.out 2>&1 &
