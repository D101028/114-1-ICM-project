import numpy as np
import matplotlib.pyplot as plt

def plot1():
    # 不同方法對應的耗時列表，列表內分別是 case01~case05 的測試數據，None 表示缺失 (可考慮直接丟棄有缺失的 case04)
    data = {
        "L2 Pixelwise": [0.047, 1.518, 2.610, None, 0.509], 
        "Hausdorff": [0.358, 16.484, 28.050, None, 5.335], 
        "Chamfer": [0.401, 24.838, 35.455, 447.639, 6.619], 
        "L2 Pixelwise + Hausdorff": [0.0513, 1.839, 2.636, None, 0.578], 
        "L2 Pixelwise + Chamfer": [0.054, 1.814, 2.546, None, 0.579]
    }

    title = "Time Consumption of Different Similarity Functions"

    # Prepare data (exclude Case04)
    cases = ["Case01", "Case02", "Case03", "Case05"]
    case_indices = [0, 1, 2, 4]
    methods = list(data.keys())
    x = np.arange(len(cases))
    width = 0.15

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot bars for each method
    for i, method in enumerate(methods):
        values = [data[method][j] for j in case_indices]
        offset = (i - 2) * width
        bars = ax.bar(x + offset, values, width, label=method)
        
        # Add value labels on bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Test Case', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(cases)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot2():
    data = {
        "L2 Pixelwise": [1/1, 44/66, 42/66, None, 14/14], 
        "Hausdorff": [1/1, 30/66, 42/66, None, 12/14], 
        "Chamfer": [1/1, 48/66, 44/66, None, 12/14], 
        "L2 Pixelwise + Hausdorff": [1/1, 55/66, 53/66, None, 14/14], 
        "L2 Pixelwise + Chamfer": [1/1, 55/66, 55/66, None, 14/14]
    }

    title = "Accuracy of Different Similarity Functions"

    # Prepare data (exclude Case04)
    cases = ["Case01", "Case02", "Case03", "Case05"]
    case_indices = [0, 1, 2, 4]
    methods = list(data.keys())
    x = np.arange(len(cases))
    width = 0.15

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot bars for each method
    for i, method in enumerate(methods):
        values = [data[method][j] for j in case_indices]
        offset = (i - 2) * width
        bars = ax.bar(x + offset, values, width, label=method)
        
        # Add value labels on bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Test Case', fontsize=12)
    ax.set_ylabel('Accuracy (Correct Symbols / Total Symbols)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(cases)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot3():
    data = {
        "PC (Intel i5-10400)": [0.054, 1.814, 2.546, None, 0.579], 
        "Phone (Samsung Exynos 1280 SoC)": [0.125, 4.767, 7.917, None, 1.459]
    }

    title = "Time Consumption on Different Devices (L2 Pixelwise + Chamfer)"

    # Prepare data (exclude Case04)
    cases = ["Case01", "Case02", "Case03", "Case05"]
    case_indices = [0, 1, 2, 4]
    methods = list(data.keys())
    x = np.arange(len(cases))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot bars for each method
    for i, method in enumerate(methods):
        values = [data[method][j] for j in case_indices]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, values, width, label=method)
        
        # Add value labels on bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Test Case', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(cases)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()

plot3()