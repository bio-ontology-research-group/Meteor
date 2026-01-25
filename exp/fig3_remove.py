import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import seaborn as sns
import matplotlib as mpl
import ast


df = pd.read_pickle('data/metainfo.pkl')


def safe_eval(val):
    if pd.isna(val) or val == "" or str(val).lower() == 'nan':
        return set()
    if isinstance(val, set): return val
    try:
        s_val = str(val).strip()
        if ':' in s_val: s_val = s_val.split(':', 1)[1].strip()
        return ast.literal_eval(s_val)
    except: return set()

def is_four_digit_ec(ec_str):
    return len(ec_str.split('.')) == 4

def parse_pairs(pairs_set):
    res = {}
    for item in pairs_set:
        parts = item.split('_', 2)
        if len(parts) >= 3: res[parts[1]] = (parts[0], parts[2])
    return res


plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'legend.fontsize': 10,
    'figure.dpi': 300 
})

def plot_ec_fingerprint_clustered_oxford(df, output_root="ec_fingerprints"):
    if not os.path.exists(output_root): os.makedirs(output_root)
    

    df['changed_pairs'] = df['changed_pairs'].apply(safe_eval)
    df['removed_proteins'] = df['removed_proteins'].apply(safe_eval)
    
    palette = sns.color_palette("muted")
    colors = [
        "#ffffff", # 0: Not Updated
        "#A7C7E7", # 1: Consistent Lock 
        "#7FADDC", # 2: Normal Substitution (Muted Yellow)
        "#7A2929", # 3: Substitution Error A_ (Muted Red)
        "#E2AFA7", # 4: Substitution Error A3_ (Pale Red/Coral)
        "#7f8c8d"  # 5: Loss / Inactive 
    ]
    custom_cmap = ListedColormap(colors)

    unique_seeds = df['seed'].unique()
    for seed in unique_seeds:
        seed_dir = os.path.join(output_root, f"seed_{seed}")
        if not os.path.exists(seed_dir): os.makedirs(seed_dir)
        
        df_seed = df[df['seed'] == seed]
        for genome in df_seed['current_genome'].unique():
            df_g = df_seed[df_seed['current_genome'] == genome].sort_values('remove_ratio')
            ratios = sorted(df_g['remove_ratio'].unique())

            first_pairs = parse_pairs(df_g.iloc[0]['changed_pairs'])
            target_ecs_info = {ec: prot for ec, (pre, prot) in first_pairs.items() 
                               if pre in ['A', 'A3'] and is_four_digit_ec(ec)}
            if not target_ecs_info: continue
            target_ecs = list(target_ecs_info.keys())

            matrix = np.zeros((len(target_ecs), len(ratios)))
            for r_idx, r in enumerate(ratios):
                row = df_g[df_g['remove_ratio'] == r]
                curr_pairs = parse_pairs(row['changed_pairs'].iloc[0])
                removed_set = row['removed_proteins'].iloc[0]

                for ec_idx, ec in enumerate(target_ecs):
                    orig_prot = target_ecs_info[ec]
                    is_orig_removed = orig_prot in removed_set
                    if ec not in curr_pairs:
                        matrix[ec_idx, r_idx] = 5 if is_orig_removed else 0
                    else:
                        prefix, current_prot = curr_pairs[ec]
                        if prefix == 'M': matrix[ec_idx, r_idx] = 5
                        elif current_prot == orig_prot: matrix[ec_idx, r_idx] = 1
                        else:
                            if is_orig_removed: matrix[ec_idx, r_idx] = 2
                            else: matrix[ec_idx, r_idx] = 4 if prefix == 'A3' else 3

            def get_sort_key(row):
                first_change = np.where(row != 1)[0]
                first_change_idx = first_change[0] if len(first_change) > 0 else len(row)
                return (first_change_idx, -row[-1])

            sorted_indices = sorted(range(len(target_ecs)), key=lambda i: get_sort_key(matrix[i]), reverse=True)
            matrix = matrix[sorted_indices]

            fig, ax = plt.subplots(figsize=(4, 4)) 
            
   
            im = ax.imshow(matrix, aspect='auto', cmap=custom_cmap, 
                           interpolation='nearest', origin='lower',
                           vmin=0, vmax=5)


            ax.set_title(f'Functional Persistence Fingerprint', 
                         fontweight='bold', loc='center', pad=12)
            

            ax.text(1.0, 1.02, f'Seed: {seed} | Genome: {genome}', 
                    transform=ax.transAxes, ha='right', fontsize=7, style='italic')

            ax.set_ylabel(f'{len(target_ecs)} EC Units', fontweight='bold')
            ax.set_xlabel('Protein Removal Ratio', fontweight='bold')
            

            ax.set_xticks(np.arange(len(ratios)))
            ax.set_xticklabels(ratios, rotation=0)
            
            ax.set_yticks([])
            sns.despine(top=True, right=True)
            
            legend_elements = [
                Patch(facecolor=colors[1], label='Consistent'),
                Patch(facecolor=colors[2], label='Normal Substitution'),
                Patch(facecolor=colors[3], label='Error (X.X.X.X)'),
                Patch(facecolor=colors[4], label='Error (X.X.X.-)'),
                Patch(facecolor=colors[5], label='Inactive'),
                Patch(facecolor=colors[0], label='Not updated')
            ]
            ax.legend(handles=legend_elements, loc='upper center', 
                      bbox_to_anchor=(0.5, -0.18), ncol=3, 
                      frameon=False, prop={'size': 8})

            plt.tight_layout()
            
            base_name = os.path.join(seed_dir, f"{genome}_fingerprint")
            plt.savefig(f"{base_name}.pdf", bbox_inches='tight')
            plt.savefig(f"{base_name}.png", bbox_inches='tight')
            plt.close()
            # print(f"{base_name}.pdf",flush=True )
            

    print(f"[SUCCESS] {len(unique_seeds)} 个 Seed 的指纹图已按照 Oxford 双栏标准生成。")


# plot_ec_fingerprint_clustered_oxford(df,'fig/S2_all')



############################################################


df = df[df['remove_ratio'] > 0]

df_avg = df.groupby('remove_ratio').agg({
    'changed_reactions': ['mean', 'std'],
    'active_ECs': ['mean', 'std'],
    'changed_score': ['mean', 'std'],
    'changed_protein_count': ['mean', 'std'],
    'remainpr_rxnnumb': ['mean', 'std']
}).reset_index()

df_avg.columns = ['remove_ratio',
                  'cr_mean', 'cr_std', 
                  'ae_mean', 'ae_std',
                    'cs_mean', 'cs_std',
                    'cpc_mean', 'cpc_std',
                    'rn_mean', 'rn_std'
                  ]


plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'figure.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

palette = [
            sns.color_palette("Greys")[0],
            sns.color_palette("Blues_d")[0],
            sns.color_palette("Reds_d")[0],]
erropalette = [
                sns.color_palette("Greys")[4],
                sns.color_palette("Blues_d")[4],
                sns.color_palette("Reds_d")[4],]
metrics = [
    ('remainpr_rxnnumb', 'rn_mean', 'rn_std', 'Remain Proteins', 'o'),
    ('active_ECs', 'ae_mean', 'ae_std', 'Refined ECs', 's'),
    ('changed_protein_count', 'cpc_mean', 'cpc_std', 'Refined Proteins', 'D')
]


fig, ax = plt.subplots(figsize=(4, 3.5)) # 适合单栏展示

for i, (raw_col, mean_col, std_col, label, marker) in enumerate(metrics):
    color = palette[i]
    errorcolor = erropalette[i]

    
    # ax.scatter(df['remove_ratio'], df[raw_col], 
    #             alpha=0.2, color=color, rasterized=True)
    

    ax.errorbar(df_avg['remove_ratio'], df_avg[mean_col],# yerr=df_avg[std_col], 
                fmt=marker + '-', color=errorcolor, label=label,
                markersize=2, linewidth=1, capsize=2, elinewidth=0.8)
                # markersize=3.5, linewidth=1, capsize=2, elinewidth=0.8)

# ax.set_ylim(bottom=0, top=1000)
# ax.set_xlim(left=0.8, right=1)
# ax.set_ylim(bottom=0, top=400)
# ax.set_ylim(bottom=0, top=6000)


ax.set_xlabel('Genome Removal Ratio ($R_{remove}$)', fontweight='bold')
ax.set_ylabel('Number Count', fontweight='bold')
ax.set_title('Global Refinements Trends', fontweight='bold', loc='center')


for spine in ax.spines.values():
    spine.set_linewidth(0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.legend(frameon=True, facecolor='white', framealpha=0.8, edgecolor='none')

plt.tight_layout()
plt.savefig('fig/fig3.pdf', bbox_inches='tight')
plt.savefig('fig/fig3.png', bbox_inches='tight', dpi=300)
