import json
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D # 用于创建手动图例


## load preprocess data
df = pd.read_csv('data/evalopt_MEMOTE_summary.csv')

sns.set_theme(style="white")

df['Connectivity'] = 1 - df['Blocked Reactions Fraction']

plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'font.family': 'sans-serif',
    'figure.dpi': 300
})


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(3.5, 8.5))


sns.ecdfplot(data=df, x='Total Score', ax=ax1, label='Total', color='#1f77b4', lw=1.5)
sns.ecdfplot(data=df, x='Score_consistency', ax=ax1, label='Consist.', color='#1f77b4', lw=1, ls='--')
ax1.set_title('(A) Cumulative Quality Distribution', fontweight='bold')#, x=-0.5, ha='left')
ax1.set_ylabel('Proportion', fontweight='bold')
ax1.set_xlabel('')
ax1.set_xlim(0, 1)
ax1.legend(frameon=False, loc='upper left', fontsize=7)
ax1.axvline(0.6, color='gray', lw=1.0, ls=':')



palette_colors = {"positive": '#1f77b4', "negative": '#a50f15'}


sns.scatterplot(data=df, x='Connectivity', y='Total Score', hue='Gram', 
                palette=palette_colors, s=3, alpha=0.06, ax=ax2, edgecolors=None, legend=False)


sns.kdeplot(data=df, x='Connectivity', y='Total Score', hue='Gram', 
            palette=palette_colors, 
            levels=4,              
            linewidths=0.8,        
            ax=ax2, 
            alpha=0.8,             
            legend=False)          


ax2.set_title('(B) Connectivity & Integrity', fontweight='bold') #, x=0, ha='left')
ax2.set_ylabel('Total Score', fontweight='bold')
ax2.set_xlabel('Connectivity (1 - Blocked)', fontweight='bold')
ax2.set_ylim(0, 1.1)
ax2.set_xlim(0, 1.05)


legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Positive',
           markerfacecolor=palette_colors['positive'], markersize=6),
    Line2D([0], [0], marker='o', color='w', label='Negative',
           markerfacecolor=palette_colors['negative'], markersize=6)
]
ax2.legend(handles=legend_elements, loc='lower left', frameon=True, 
           framealpha=0.8, edgecolor='none', fontsize=7.5)


sns.histplot(data=df, x='Growth Rate (Default)', ax=ax3, color='#1f77b4', 
             kde=True, element="step", alpha=0.15, linewidth=1.2)
ax3.set_title('(C) Growth Rate Distribution', fontweight='bold') #, x=0, ha='left')
ax3.set_ylabel('Count', fontweight='bold')
ax3.set_xlabel('Growth Rate ($h^{-1}$)', fontweight='bold')
median_gr = df['Growth Rate (Default)'].median()
ax3.axvline(median_gr, color='gray', lw=0.8, ls='-.')
ax3.text(median_gr*1.1, ax3.get_ylim()[1]*0.8, f'Med: {median_gr:.2f}', color='gray', fontsize=7)


for ax in [ax1, ax2, ax3]:
    sns.despine(ax=ax)
    ax.yaxis.set_label_coords(-0.16, 0.5)

plt.subplots_adjust(hspace=0.55, left=0.18, right=0.95, top=0.95, bottom=0.08)
plt.savefig('fig/S4_memote_eval.pdf')

