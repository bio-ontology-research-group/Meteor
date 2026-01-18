# /ibex/scratch/projects/c2014/kexin/funcarve/optresult/deepec2/v5_preds_probs_scores_GCF_017350745_v5_t5k1g05_EXRM0.7_seed17.pkl

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import ast
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

file_path='data/fig2_seedcombi.txt'

all_results = []

# 定义正则表达式提取 EXRM, seed 和 结果字典
metric_pattern = re.compile(r"Results \(pre, rec, f1, roc, acc\): (\{.*\})")
params_pattern = re.compile(r"EXRM([\d\.]+)_seed(\d+)")

with open(file_path, 'r') as f:
    content = f.read()
    # 按分隔符切分块
    blocks = content.split('------------------------------------------------------')
    
    for block in blocks:
        if not block.strip(): continue
        
        # 提取结果字典
        metric_match = metric_pattern.search(block)
        # 提取参数
        params_match = params_pattern.search(block)
        
        if metric_match and params_match:
            res_dict = ast.literal_eval(metric_match.group(1))
            exrm = params_match.group(1)
            seed = params_match.group(2)
            
            # 提取 orig 和 opt 的 5 个指标
            for mode in ['orig', 'opt']:
                metrics = res_dict[mode]
                # all_results.append({
                #     'EXRM': float(exrm),
                #     'Seed': int(seed),
                #     'Mode': mode,
                #     'Precision': metrics[0],
                #     'Recall': metrics[1],
                #     'F1': metrics[2],
                #     'ROC': metrics[3],
                #     'Accuracy': metrics[4]
                # })
                all_results.append({
                    'Remove Ratio': float(exrm),
                    'Mode': 'Original' if mode == 'orig' else 'Optimized',
                    'Precision': metrics[0], 'Recall': metrics[1],
                    'F1': metrics[2], 'ROC': metrics[3]
                })

# 转换为 DataFrame
df = pd.DataFrame(all_results)

avg_df = df.groupby(['Remove Ratio', 'Mode']).mean(numeric_only=True).reset_index()

plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'font.family': 'serif', # 衬线字体，专业感强
    'figure.dpi': 300
})

# 假设 avg_df 已经准备好，包含指标：Precision, Recall, F1, ROC
metrics_groups = [['Precision', 'Recall'], ['F1', 'ROC']]
colors = {"Original": "#95a5a6", "Optimized": "#2980b9"}
markers = {"Precision": "o", "Recall": "s", "F1": "^", "ROC": "D"}

# 2. 创建 2x1 垂直布局，共享 X 轴
# 宽度 3.5 inch (适合单栏)，高度 6.0 inch (提供充足的纵向展示空间)
fig, axes = plt.subplots(2, 1, figsize=(4, 5.0), sharex=True)

for i, group in enumerate(metrics_groups):
    ax = axes[i]
    orig_data = avg_df[avg_df['Mode'] == 'Original'].sort_values('Remove Ratio')
    opt_data = avg_df[avg_df['Mode'] == 'Optimized'].sort_values('Remove Ratio')
    
    for metric in group:
        # 绘制 Original (灰色虚线)
        ax.plot(orig_data['Remove Ratio'], orig_data[metric], 
                color=colors['Original'], linestyle='--', linewidth=1.0, 
                marker=markers[metric], markersize=3, alpha=0.7)
        
        # 绘制 Optimized (蓝色实线)
        ax.plot(opt_data['Remove Ratio'], opt_data[metric], 
                color=colors['Optimized'], linestyle='-', linewidth=1.2, 
                marker=markers[metric], markersize=4)
        
        # 绘制增益阴影 (Gain Area) - 提升视觉信息密度
        ax.fill_between(orig_data['Remove Ratio'], orig_data[metric], opt_data[metric], 
                        color=colors['Optimized'], alpha=0.08)

    # 子图美化
    ax.set_title(f'({chr(65+i)}) {" & ".join(group)} Performance', fontweight='bold', pad=8)
    ax.set_ylabel('Score Value', fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.4)
    sns.despine(ax=ax) # 移除多余边框，保持简约

    # 添加子图内图例
    metric_handles = [Line2D([0], [0], color='black', marker=markers[m], 
                             linestyle='-', label=m, markersize=4) for m in group]
    mode_handles = [
        Line2D([0], [0], color=colors['Optimized'], linestyle='-', label='Optimized'),
        Line2D([0], [0], color=colors['Original'], linestyle='--', label='Original')
    ]
    ax.legend(handles=metric_handles + mode_handles, frameon=False, fontsize=10, loc='best', ncol=2)

# 统一设置底部 X 轴标签
axes[1].set_xlabel('Genome Remove Ratio', fontweight='bold')

plt.tight_layout()
plt.savefig('fig/fig2.pdf', bbox_inches='tight')
plt.savefig('fig/fig2.png', dpi=300, bbox_inches='tight')
