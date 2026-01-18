import json
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D # 用于创建手动图例


def parse_memote(file):
    with open(file, 'r') as f:
        data = json.load(f)
    
    # 1. 基础信息与总分
    res = {
        'Total Score': data['score'].get('total_score', 0)
    }
    
    # 2. 提取各个 Section 的得分 (Consistency, Basic, Annotation 等)
    sections = data['score'].get('sections', [])
    for item in sections:
        res[f"Score_{item['section']}"] = item['score']
    
    # 3. 提取具体测试指标 (Tests)
    tests = data.get('tests', {})
    
    # --- 质量与电荷平衡 ---
    # 使用 score (1 - metric)，代表平衡反应的百分比
    res['Mass Balance %'] = tests.get('test_reaction_mass_balance', {}).get('score', 0)
    res['Charge Balance %'] = tests.get('test_reaction_charge_balance', {}).get('score', 0)
    
    # --- 生物量一致性 (Biomass Consistency) ---
    # 计算偏离 1.0 g/mmol 的平均绝对误差。0.0 代表完美符合。
    bc_metric = tests.get('test_biomass_consistency', {}).get('metric', {})
    if isinstance(bc_metric, dict) and bc_metric:
        res['Biomass Consistency Deviation'] = sum(abs(v - 1.0) for v in bc_metric.values()) / len(bc_metric)
    else:
        res['Biomass Consistency Deviation'] = 0.0
        
    # --- 生长率 (Default Medium) ---
    # 提取默认培养基下的平均生长速率值
    growth_data = tests.get('test_biomass_default_production', {}).get('data', {})
    if isinstance(growth_data, dict) and growth_data:
        res['Growth Rate (Default)'] = np.mean(list(growth_data.values()))
    else:
        res['Growth Rate (Default)'] = 0.0
        
    # --- 阻塞的前体物质 (Blocked Precursors) ---
    # 分别统计在默认培养基和完全培养基下，无法产生的唯一前体数量
    for medium_type in ['default', 'open']:
        key = f'test_biomass_precursors_{medium_type}_production'
        precursor_data = tests.get(key, {}).get('data', {})
        if isinstance(precursor_data, dict) and precursor_data:
            all_blocked = set()
            for precursors in precursor_data.values():
                if isinstance(precursors, list):
                    all_blocked.update(precursors)
            res[f'Blocked Precursors ({medium_type})'] = len(all_blocked)
        else:
            res[f'Blocked Precursors ({medium_type})'] = 0
            
    # --- 不切实际的生长速率 (Unrealistic Growth) ---
    # 统计有多少个生物量反应的生长速率超过了生物学极限 (True 的数量)
    fast_growth_data = tests.get('test_fast_growth_default', {}).get('data', {})
    if isinstance(fast_growth_data, dict) and fast_growth_data:
        res['Unrealistic Growth Count'] = sum(1 for v in fast_growth_data.values() if v is True)
    else:
        res['Unrealistic Growth Count'] = 0
        
    # --- 阻塞反应比例 (Blocked Reactions) ---
    # 网络中无法携带通量的反应比例
    res['Blocked Reactions Fraction'] = tests.get('test_blocked_reactions', {}).get('metric', 0)
    
    return res


##### process memote report
PREDICTION_FOLDER = '/ibex/scratch/projects/c2014/kexin/funcarve/optresult/bacdive_deepec2test1/'
OUTPUT_FILE = 'evaluation_opt_MEMOTE_final_summary.csv'

METAINFO_PATH = '/ibex/user/niuk0a/funcarve/DeepProZyme/bacdive/metainfo.csv'

def load_and_preprocess_metainfo(metainfo_path):
    """
    加载 metainfo 文件，并将其处理成一个方便查找的字典结构。
    结构: { Genome_ID: { EC_number: Phenotype_Label (1 or 0) } }
    """
    print(f"Loading metainfo from: {metainfo_path}...")
    try:
        df_meta = pd.read_csv(metainfo_path)
    except Exception as e:
        print(f"Error loading metainfo: {e}")
        return {}

    # 1. 提取核心 Genome ID (与预测文件名中的 ID 匹配)
    # df_meta['Core_Genome_ID'] = df_meta['ID'].astype(str) + ',' + df_meta['Genome name'].astype(str)
    # df_meta['Core_Genome_ID'] = df_meta['Core_Genome_ID'].apply(lambda x: x.split(',')[1].split('_')[0])
    df_meta['Core_Genome_ID'] = df_meta['Genome name'].apply(lambda x: x.split('.')[0])
    return set(df_meta['Core_Genome_ID'].tolist())



# foldername = os.path.basename(PREDICTION_FOLDER.rstrip('/'))

# files = [f for f in os.listdir(PREDICTION_FOLDER) if f.endswith('.json')]
# genonames = [n.split('.json')[0] for n in files]
# infos = [parse_memote(os.path.join(PREDICTION_FOLDER, x)) for x in files]
# all_results =[]
# for i in range(len(infos)):
#     row = infos[i]
#     row['Genome'] = genonames[i]
#     row['Gram'] = genonames[i].split('_')[-1]
#     all_results.append(row)
    
# df = pd.DataFrame(all_results)
# df.to_csv(OUTPUT_FILE,index=False)

# df = pd.read_csv(OUTPUT_FILE)

# # need only keeps geno in metainfo
# genomes2keep = load_and_preprocess_metainfo(METAINFO_PATH)
# # print(df.columns)
# # print(df.head())
# print(f'KEEPS {len(genomes2keep)}')
# print(f'oridf:{len(df)}')
# df['Core_Genome_ID'] = df['Genome'].apply(lambda x: "_".join(x.split('_')[:-1]))
# df_filter = df[df['Core_Genome_ID'].isin(genomes2keep)]
# print(f'filter df:{len(df_filter)}')

# df.to_csv('evaluation_opt_MEMOTE_final_summary_filter.csv',index=False)

# exit()

## load preprocess data
df = pd.read_csv('S1_evalopt_MEMOTE_summary.csv')
# 设置绘图风格
# Set plotting style
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

# 创建 3x1 画布
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(3.5, 8.5))

# --- (A) ECDF: Quality Scores ---
sns.ecdfplot(data=df, x='Total Score', ax=ax1, label='Total', color='#1f77b4', lw=1.5)
sns.ecdfplot(data=df, x='Score_consistency', ax=ax1, label='Consist.', color='#1f77b4', lw=1, ls='--')
ax1.set_title('(A) Cumulative Quality Distribution', fontweight='bold')#, x=-0.5, ha='left')
ax1.set_ylabel('Proportion', fontweight='bold')
ax1.set_xlabel('')
ax1.set_xlim(0, 1)
ax1.legend(frameon=False, loc='upper left', fontsize=7)
ax1.axvline(0.6, color='gray', lw=1.0, ls=':')

# --- (B) Connectivity & Integrity (重点：散点 + 等高线) ---
# 定义高对比配色
palette_colors = {"positive": '#1f77b4', "negative": '#a50f15'}

# 1. 绘制散点图 (作为低透明度背景)
sns.scatterplot(data=df, x='Connectivity', y='Total Score', hue='Gram', 
                palette=palette_colors, s=3, alpha=0.06, ax=ax2, edgecolors=None, legend=False)

# =================== 新增核心代码 ===================
# 2. 绘制等高线 (KDE Plot)
sns.kdeplot(data=df, x='Connectivity', y='Total Score', hue='Gram', 
            palette=palette_colors, # 使用与散点相同的配色
            levels=4,               # 等高线的层数，4-5层比较清晰
            linewidths=0.8,         # 线条宽度
            ax=ax2, 
            alpha=0.8,              # 线条透明度
            legend=False)           # 关键：关闭自带图例，避免冲突
# ====================================================

ax2.set_title('(B) Connectivity & Integrity', fontweight='bold') #, x=0, ha='left')
ax2.set_ylabel('Total Score', fontweight='bold')
ax2.set_xlabel('Connectivity (1 - Blocked)', fontweight='bold')
ax2.set_ylim(0, 1.1)
ax2.set_xlim(0, 1.05)

# --- 手动创建图例 (保持不变，确保可见) ---
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Positive',
           markerfacecolor=palette_colors['positive'], markersize=6),
    Line2D([0], [0], marker='o', color='w', label='Negative',
           markerfacecolor=palette_colors['negative'], markersize=6)
]
# 放在左下角，带半透明背景
ax2.legend(handles=legend_elements, loc='lower left', frameon=True, 
           framealpha=0.8, edgecolor='none', fontsize=7.5)

# --- (C) Growth Rate Distribution ---
sns.histplot(data=df, x='Growth Rate (Default)', ax=ax3, color='#1f77b4', 
             kde=True, element="step", alpha=0.15, linewidth=1.2)
ax3.set_title('(C) Growth Rate Distribution', fontweight='bold') #, x=0, ha='left')
ax3.set_ylabel('Count', fontweight='bold')
ax3.set_xlabel('Growth Rate ($h^{-1}$)', fontweight='bold')
median_gr = df['Growth Rate (Default)'].median()
ax3.axvline(median_gr, color='gray', lw=0.8, ls='-.')
ax3.text(median_gr*1.1, ax3.get_ylim()[1]*0.8, f'Med: {median_gr:.2f}', color='gray', fontsize=7)

# --- 2. 细节对齐与导出 ---
for ax in [ax1, ax2, ax3]:
    sns.despine(ax=ax)
    ax.yaxis.set_label_coords(-0.16, 0.5)

plt.subplots_adjust(hspace=0.55, left=0.18, right=0.95, top=0.95, bottom=0.08)
plt.savefig('fig/S1_memote_eval.pdf')

