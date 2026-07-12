"""Reaction-consistent EC propagation analysis.

For each genome: compare baseline vs METEOR scores for ECs in
active-reaction set (should increase) vs muted set (should decrease).
"""
import pickle, glob, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from baseline_io import resolve_baseline_pkl

METEOR_OUT = "/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v7_run/meteor_out"
V6_DATA = "/ibex/user/niuk0a/funcarve/cobra/v6/data"

all_ecs = [l.strip() for l in open(f"{V6_DATA}/all_ancestors.txt") if l.strip()]

baselines = ["dpz", "clean", "enzbert"]
variants = ["vanilla"]

results = []

for baseline in baselines:
    for variant in variants:
        out_dir = f"{METEOR_OUT}/{baseline}_{variant}"
        meteor_dfs = sorted(glob.glob(f"{out_dir}/meteor_df_*.pkl"))
        for mf in meteor_dfs:
            gca = os.path.basename(mf).replace("meteor_df_", "").replace(".pkl", "")
            preds_f = f"{out_dir}/meteor_preds_{gca}.pkl"
            if not os.path.exists(preds_f):
                continue
            try:
                base_f = resolve_baseline_pkl(baseline, variant, gca)
            except Exception:
                continue
            if not os.path.exists(base_f):
                continue

            df_m = pickle.load(open(mf, "rb"))
            df_b = pickle.load(open(base_f, "rb"))
            preds = pickle.load(open(preds_f, "rb"))

            # Align columns to all_ecs
            df_b.columns = [c.replace("EC:", "") for c in df_b.columns]
            shared_prots = df_m.index.intersection(df_b.index)
            shared_ecs = df_m.columns.intersection(df_b.columns)
            if len(shared_prots) == 0 or len(shared_ecs) == 0:
                continue

            bvals = df_b.loc[shared_prots, shared_ecs].values.astype(float)
            mvals = df_m.loc[shared_prots, shared_ecs].values.astype(float)
            delta = mvals - bvals

            active_ecs = preds["active_ecs"]
            muted_ecs = preds["muted_ecs"]

            shared_ecs_list = list(shared_ecs)
            active_mask = np.array([ec in active_ecs for ec in shared_ecs_list])
            muted_mask = np.array([ec in muted_ecs for ec in shared_ecs_list])

            # Only count cells where baseline > 0 (non-trivial)
            nonzero_b = bvals > 0

            # Active ECs: delta for cells with nonzero baseline
            if active_mask.any():
                act_delta = delta[:, active_mask]
                act_nonzero = nonzero_b[:, active_mask]
                if act_nonzero.any():
                    mean_act = act_delta[act_nonzero].mean()
                    n_act = int(act_nonzero.sum())
                else:
                    mean_act = 0.0; n_act = 0
            else:
                mean_act = 0.0; n_act = 0

            # Muted ECs: delta for cells with nonzero baseline
            if muted_mask.any():
                mut_delta = delta[:, muted_mask]
                mut_nonzero = nonzero_b[:, muted_mask]
                if mut_nonzero.any():
                    mean_mut = mut_delta[mut_nonzero].mean()
                    n_mut = int(mut_nonzero.sum())
                else:
                    mean_mut = 0.0; n_mut = 0
            else:
                mean_mut = 0.0; n_mut = 0

            results.append({
                "baseline": baseline,
                "variant": variant,
                "genome": gca,
                "n_active_ecs": len(active_ecs & set(shared_ecs_list)),
                "n_muted_ecs": len(muted_ecs & set(shared_ecs_list)),
                "mean_delta_active": mean_act,
                "n_cells_active": n_act,
                "mean_delta_muted": mean_mut,
                "n_cells_muted": n_mut,
            })

df = pd.DataFrame(results)
print(f"Total genomes analyzed: {len(df)}")
print()

for b in baselines:
    sub = df[df["baseline"] == b]
    if len(sub) == 0:
        continue
    print(f"=== {b} (n={len(sub)} genomes) ===")
    print(f"  Active ECs: mean Δscore = {sub[mean_delta_active].mean():+.4f}  (per-genome mean)")
    print(f"              median Δ    = {sub[mean_delta_active].median():+.4f}")
    print(f"              genomes with Δ>0: {(sub[mean_delta_active]>0).sum()}/{len(sub)}")
    print(f"  Muted  ECs: mean Δscore = {sub[mean_delta_muted].mean():+.4f}")
    print(f"              median Δ    = {sub[mean_delta_muted].median():+.4f}")
    print(f"              genomes with Δ<0: {(sub[mean_delta_muted]<0).sum()}/{len(sub)}")
    print()

# Save
out_f = "/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v7_run/downstream_results/ec_propagation_analysis.tsv"
os.makedirs(os.path.dirname(out_f), exist_ok=True)
df.to_csv(out_f, sep="\t", index=False)
print(f"Saved to {out_f}")
