#!/usr/bin/env bash
#
# run_train.sh
# 这个脚本对应 PyCharm Run Configuration “train” 的全部设置：
#   • 工作目录：/Users/zhengshizhao/PycharmProjects/FNO-VMC
#   • 解释器：   /opt/anaconda3/bin/python
#   • 环境变量： PYTHONUNBUFFERED=1
#   • 脚本路径： src/scripts/train.py
#   • 脚本参数： jlts/fno_run --logfile logs/fno_run.log --wandb_project FNO-VMC
#

# 1. 设置环境变量，保证 Python 输出不做缓冲
export PYTHONUNBUFFERED=1

# 2. 切换到项目根目录
cd /Users/zhengshizhao/PycharmProjects/FNO-VMC || {
  echo "Error: 无法切换到工作目录 /Users/zhengshizhao/PycharmProjects/FNO-VMC"
  exit 1
}

# 3. 使用 Anaconda 的 Python 解释器来运行 train.py
#    如果你本地 PATH 已经指向了正确的 Python，也可直接写 `python3` 或 `python`。
#    下面示例将路径写死成 PyCharm 中配置的解释器路径：
/opt/anaconda3/bin/python src/scripts/train.py jlts/fno_run \
  --logfile logs/fno_run.log \
  --wandb_project FNO-VMC

# 如果前面的 python 命令报错，你可以：
#  1) 确认 /opt/anaconda3/bin/python 确实存在，或者替换成 `python3`
#  2) 确认 src/scripts/train.py 文件可执行且路径无误
