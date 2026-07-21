import argparse
import os
import re
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser(description='Parse a training log and plot metrics.')
parser.add_argument('log_file_path', help='Path to the log file to parse')
parser.add_argument('-o', '--output', default='metrics_plot.png', help='Output image file path')
args = parser.parse_args()

# 1. 定义你的日志文件路径
log_file_path = os.path.expanduser(args.log_file_path)
output_image = os.path.expanduser(args.output)

# 2. 初始化一个字典来存储所有提取的指标
data = {
    'iteration': [],
    'learning rate': [],
    'loss': [],
    'action_loss': [],
    'grad norm': []
}

# 3. 正则表达式：用于精确匹配迭代步数和关键指标
# 匹配类似 "iteration     29999/" 的数字
iter_pattern = re.compile(r'iteration\s+(\d+)\s*/')
# 匹配类似 "metric_name: 对应数值" 的结构（支持科学计数法）
metrics_to_extract = ['learning rate', 'loss', 'action_loss', 'grad norm']

print("开始解析日志...")

if not os.path.isfile(log_file_path):
    raise FileNotFoundError(f'日志文件未找到: {log_file_path}')

with open(log_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        # 寻找包含 iteration 的有效数据行
        iter_match = iter_pattern.search(line)
        if iter_match:
            current_iter = int(iter_match.group(1))
            data['iteration'].append(current_iter)
            
            # 提取其他指标
            for metric in metrics_to_extract:
                # 正则解释：匹配指标名 + 冒号 + 空格 + 整数/浮点数/科学计数法数字
                metric_pattern = rf'{metric}:\s*([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)'
                match = re.search(metric_pattern, line)
                if match:
                    data[metric].append(float(match.group(1)))
                else:
                    data[metric].append(None) # 如果某行缺了该指标，用 None 占位

# 转为 DataFrame 方便处理
df = pd.DataFrame(data).dropna()
print(f"成功解析了 {len(df)} 条迭代记录！")

# 4. 开始绘图
# 我们创建一个 2x2 的画布，把 4 个主要指标分离开，避免因为量级不同挤在一起
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
axes = axes.flatten()

metrics_list = ['learning rate', 'loss', 'action_loss', 'grad norm']
colors = ['blue', 'red', 'orange', 'purple']

for i, metric in enumerate(metrics_list):
    ax = axes[i]
    ax.plot(df['iteration'], df[metric], label=metric, color=colors[i], linewidth=1.5)
    ax.set_title(f'{metric.upper()} vs Iteration')
    ax.set_ylabel(metric)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 如果是学习率或者 loss 这种极小的值，用对数坐标轴显示会更清晰
    if metric in ['learning rate', 'loss']:
        ax.set_yscale('log')
        ax.set_title(f'{metric.upper()} vs Iteration (Log Scale)')

# 设置最底部的 X 轴标签
axes[2].set_xlabel('Iteration')
axes[3].set_xlabel('Iteration')

plt.tight_layout()

# 5. 保存并展示图片
plt.savefig(output_image, dpi=300)
print(f"图表已成功保存至: {output_image}")
plt.show()