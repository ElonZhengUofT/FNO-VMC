import torch
import inspect
import numpy as np

import matplotlib.pyplot as plt
import os

def mmm(tag=None):
    "cuda memory check"
    print(f"GPU Memory Usage----: {tag}:")
    allocated_memory=torch.cuda.memory_allocated()
    print(f"Allocated: {allocated_memory/1024/1024:.2f} MB")
    print(f"Reserved: {torch.cuda.memory_reserved()/1024/1024:.2f} MB")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_properties = torch.cuda.get_device_properties(device)
    free_memory = device_properties.total_memory - allocated_memory
    print(f"Free: {free_memory/ 1024 / 1024:.2f} MB")
    peak=torch.cuda.max_memory_allocated()
    # stats = torch.cuda.memory_stats(torch.device('cuda'))
    peak_memory_usage = peak
    print(f"Peak memory usage: {peak_memory_usage / (1024 * 1024):.2f} MB")
    torch.cuda.reset_peak_memory_stats()
    print()

if __name__ == "__main__":
    mmm()