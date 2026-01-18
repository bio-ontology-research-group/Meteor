import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# 1. 读取数据
# 确保此文件是由你之前的汇总脚本生成的
df = pd.read_csv('data/fig1_memote_detailed_comparison.csv')

# 2. 建立名称映射 (与论文一致)
name_map = {
    'deepec_newg': 'DeepEC',
    'deepec2_newg': 'DeepProZyme',
    'enzbert_newg': 'EnzBert',
    'ecrecer_newg': 'Ecrecer',
    'recongfmodels': 'Reconstructor (GF)',
    'reconmodels': 'Reconstructor (Draft)'
}
df['Display Method'] = df['Method'].map(name_map)

plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'font.family': 'sans-serif',
    # 'font.family': 'serif',
    'figure.dpi': 300
})

# 宽度 3.5 inch (单栏)，高度压缩至 4.5 inch
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(4, 4))

plot_order = ['DeepEC', 'EnzBert', 'Ecrecer', 'DeepProZyme', 'Reconstructor (GF)', 'Reconstructor (Draft)']

# 子图 (A): 生长速率
sns.barplot(x='Display Method', y='Growth Rate (Default)', data=df, ax=ax1, 
            palette='Blues_d', order=plot_order)
ax1.set_title('(A) Growth Rate ($h^{-1}$)', fontweight='bold')
ax1.set_ylabel('') # 移除 Y 轴标签，由标题传达信息，更简洁
ax1.set_xlabel('')
# 在索引 3.5 处画垂直虚线
ax1.axvline(x=3.5, color='gray', linestyle='--', linewidth=1.2, alpha=0.8)
# 子图 (B): 阻塞前体
sns.barplot(x='Display Method', y='Blocked Precursors (default)', data=df, ax=ax2, 
            palette='Reds_d', order=plot_order)
ax2.set_title('(B) Blocked Precursors (Gaps)', fontweight='bold')
ax2.set_ylabel('')
ax2.set_xlabel('')
ax2.tick_params(axis='x', rotation=45) # 仅在底部旋转显示标签
# 在索引 3.5 处画垂直虚线
ax2.axvline(x=3.5, color='gray', linestyle='--', linewidth=1.2, alpha=0.8)
# 调整子图间的间距
plt.tight_layout()

# 保存
plt.savefig('fig/fig1.pdf', bbox_inches='tight')
plt.savefig('fig/fig1.png', bbox_inches='tight', dpi=300)