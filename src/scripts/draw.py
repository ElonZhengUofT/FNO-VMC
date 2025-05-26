import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Plot energy mean and variance from VMC JSON log")
    parser.add_argument("--logfile", type=str, required=True,
                        help="Path to the JSON log file (e.g. results/fno_run/fno.log)")
    args = parser.parse_args()

    # 读取 JSON
    with open(args.logfile, 'r') as f:
        log = json.load(f)

    # 提取迭代次数、能量均值和方差
    # 假设 log["Energy"]["iters"] 对应迭代列表，
    # log["Energy"]["Mean"] 对应能量均值列表，
    # log["Energy"]["Variance"] 对应能量方差列表。
    iters     = np.array(log["Energy"]["iters"])
    energy    = np.array(log["Energy"]["Mean"])
    variance  = np.array(log["Energy"]["Variance"])

    # 绘图
    plt.figure(figsize=(8,5))
    plt.plot(iters, energy,   label='Energy Mean')
    plt.plot(iters, variance, label='Energy Variance')
    plt.xlabel('Iteration')
    plt.ylabel('Expectation')
    plt.title('VMC Training (from JSON log)')
    plt.legend()
    plt.grid(True)

    # 保存
    fig_path = os.path.splitext(args.logfile)[0] + '_energy_var.png'
    plt.savefig(fig_path, dpi=300)
    print(f"Plot saved to {fig_path}")

if __name__ == '__main__':
    main()
