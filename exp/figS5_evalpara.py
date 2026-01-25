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

# file_path='data/para.comb.txt'
file_path='data/para.txt'

all_results = []

metric_pattern = re.compile(r"Results \(pre, rec, f1, roc, acc\): (\{.*\})")
params_pattern = re.compile(r"v5_t([\d\.]+)_eps([\d\.]+)_delta([\d\.]+)")


with open(file_path, 'r') as f:
    content = f.read()
    blocks = content.split('------------------------------------------------------')
    
    for block in blocks:
        if not block.strip(): continue
        metric_match = metric_pattern.search(block)

        params_match = params_pattern.search(block)
        
        if metric_match and params_match:
            res_dict = ast.literal_eval(metric_match.group(1))
            thr = params_match.group(1)
            eps = params_match.group(2)
            delta = params_match.group(3).strip('.')

            for mode in ['orig', 'opt']:
                metrics = res_dict[mode]
                all_results.append({
                    'Threshold': float(thr),
                    'Epsilon': float(eps),
                    'Delta': float(delta),
                    'Mode': 'Original' if mode == 'orig' else 'Optimized',
                    'Precision': metrics[0], 'Recall': metrics[1],
                    'F1': metrics[2], 'ROC': metrics[3]
                })

# 转换为 DataFrame
df = pd.DataFrame(all_results)


avg_df = df.groupby(['Threshold', 'Epsilon', 'Delta', 'Mode']).mean(numeric_only=True).reset_index()

plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'font.family': 'serif', # 衬线字体，专业感强
    'figure.dpi': 300
})

opt_df = avg_df[avg_df['Mode'] == 'Optimized']

# for target_thr in [0.1, 0.3,0.4, 0.5, 0.9]:
for target_thr in [0.5]:
    plot_df = avg_df[(avg_df['Threshold'] == target_thr) & (avg_df['Mode'] == 'Optimized')]

    if plot_df.empty:
        print(f"error: No data available for Threshold = {target_thr}")
    else:
        pivot_f1 = plot_df.pivot(index='Delta', columns='Epsilon', values='F1')
        
        plt.figure(figsize=(4, 4))
        sns.heatmap(pivot_f1, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={'label': 'F1 Score'})
        
        plt.title(f'METEOR Performance Sensitivity Analysis')
        plt.text(1.0,0.3,f'($\\theta = {target_thr}$)', ha='right', style='italic',fontsize=7 )
        plt.xlabel('Lambda ($\lambda$)')
        plt.ylabel('Epsilon ($\epsilon$)') 
        
        plt.tight_layout()
        plt.savefig(f'fig/S5_{target_thr}.pdf')
        print("Done")
        plt.close()
        

