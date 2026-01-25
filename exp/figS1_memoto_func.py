import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch


df = pd.read_csv('data/memote_detailed_comparison.csv')

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
    'figure.dpi': 300
})


fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(4, 4))


plot_order = ['DeepEC', 'EnzBert', 'DeepProZyme', 'Reconstructor (GF)', 'Reconstructor (Draft)']


sns.barplot(x='Display Method', y='Growth Rate (Default)', data=df, ax=ax1, 
            palette='Blues_d', order=plot_order)
ax1.set_title('(A) Growth Rate ($h^{-1}$)', fontweight='bold')
ax1.set_ylabel('') 
ax1.set_xlabel('')

ax1.axvline(x=3.5, color='gray', linestyle='--', linewidth=1.2, alpha=0.8)

sns.barplot(x='Display Method', y='Blocked Precursors (default)', data=df, ax=ax2, 
            palette='Reds_d', order=plot_order)
ax2.set_title('(B) Blocked Precursors (Gaps)', fontweight='bold')
ax2.set_ylabel('')
ax2.set_xlabel('')
ax2.tick_params(axis='x', rotation=45)

ax2.axvline(x=3.5, color='gray', linestyle='--', linewidth=1.2, alpha=0.8)

plt.tight_layout()


plt.savefig('fig/S1.pdf', bbox_inches='tight')
plt.savefig('fig/S1.png', bbox_inches='tight', dpi=300)