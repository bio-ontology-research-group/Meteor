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

file_path='data/seedcombi.txt'

all_results = []


metric_pattern = re.compile(r"Results \(pre, rec, f1, roc, acc\): (\{.*\})")
params_pattern = re.compile(r"EXRM([\d\.]+)_seed(\d+)")

with open(file_path, 'r') as f:
    content = f.read()
    blocks = content.split('------------------------------------------------------')
    for block in blocks:
        if not block.strip(): continue
        metric_match = metric_pattern.search(block)
        params_match = params_pattern.search(block)
        
        if metric_match and params_match:
            res_dict = ast.literal_eval(metric_match.group(1))
            exrm = params_match.group(1)
            seed = params_match.group(2)
            
            for mode in ['orig', 'opt']:
                metrics = res_dict[mode]
                all_results.append({
                    'Remove Ratio': float(exrm),
                    'Mode': 'Original' if mode == 'orig' else 'Optimized',
                    'Precision': metrics[0], 'Recall': metrics[1],
                    'F1': metrics[2], 'ROC': metrics[3]
                })

df = pd.DataFrame(all_results)

avg_df = df.groupby(['Remove Ratio', 'Mode']).mean(numeric_only=True).reset_index()

plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'font.family': 'serif', 
    'figure.dpi': 300
})


metrics_groups = [['Precision', 'Recall'], ['F1', 'ROC']]
colors = {"Original": "#95a5a6", "Optimized": "#2980b9"}
markers = {"Precision": "o", "Recall": "s", "F1": "^", "ROC": "D"}


fig, axes = plt.subplots(2, 1, figsize=(4, 5.0), sharex=True)

for i, group in enumerate(metrics_groups):
    ax = axes[i]
    orig_data = avg_df[avg_df['Mode'] == 'Original'].sort_values('Remove Ratio')
    opt_data = avg_df[avg_df['Mode'] == 'Optimized'].sort_values('Remove Ratio')
    
    for metric in group:
        ax.plot(orig_data['Remove Ratio'], orig_data[metric], 
                color=colors['Original'], linestyle='--', linewidth=1.0, 
                marker=markers[metric], markersize=3, alpha=0.7)
        

        ax.plot(opt_data['Remove Ratio'], opt_data[metric], 
                color=colors['Optimized'], linestyle='-', linewidth=1.2, 
                marker=markers[metric], markersize=4)
        

        ax.fill_between(orig_data['Remove Ratio'], orig_data[metric], opt_data[metric], 
                        color=colors['Optimized'], alpha=0.08)


    ax.set_title(f'({chr(65+i)}) {" & ".join(group)} Performance', fontweight='bold', pad=8)
    ax.set_ylabel('Score Value', fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.4)
    sns.despine(ax=ax)


    metric_handles = [Line2D([0], [0], color='black', marker=markers[m], 
                             linestyle='-', label=m, markersize=4) for m in group]
    mode_handles = [
        Line2D([0], [0], color=colors['Optimized'], linestyle='-', label='Sequence-only'),
        Line2D([0], [0], color=colors['Original'], linestyle='--', label='System-aware')
    ]
    ax.legend(handles=metric_handles + mode_handles, frameon=False, fontsize=10, loc='best', ncol=2)


axes[1].set_xlabel('Genome Remove Ratio', fontweight='bold')

plt.tight_layout()
plt.savefig('fig/fig2.pdf', bbox_inches='tight')
plt.savefig('fig/fig2.png', dpi=300, bbox_inches='tight')
