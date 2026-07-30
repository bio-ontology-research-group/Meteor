#!/usr/bin/env python3
"""Day1-2 Analysis 1: curated sub-threshold EC recovery with IDENTITIES, denominators and CIs.

Same definition as eval/weakreal.py (EC-level, curated-GEM ground truth,
W = ECs with 0<dpz<0.5 that ARE in the curated GEM), plus:
  - the actual EC identities of W, of W recovered by METEOR, by baseline, and
    of the METEOR-only recoveries (for spot-checking a handful of examples)
  - per-organism Wilson 95% CI on the recovery proportion
  - organism-level bootstrap 95% CI on the 6-organism mean (fixed seed)
  - n_repaired / solver status carried over from the solution pickle
  - --run so the SAME analysis can be run against the pre-fix and the
    post-repair solution set and the two compared.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "src"))
from meteor_v8.utils import data_path, data_dir
import sys, os, re, json, pickle, argparse, numpy as np, pandas as pd, cobra
import warnings, logging; warnings.filterwarnings('ignore'); logging.getLogger('cobra').setLevel(logging.ERROR)

B = '/ibex/scratch/projects/c2014/kexin/funcarve'
ap = argparse.ArgumentParser()
ap.add_argument('--run', default='meteor_v8_evw_p2mu3_run',
                help='run directory under funcarve/ holding meteor_out/dpz_vanilla')
ap.add_argument('--out', default=None, help='output json basename')
ap.add_argument('--boot', type=int, default=10000)
ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()

sys.path.insert(0, f'{B}/meteor_v8/src')
from meteor_v8.utils import (load_universal, extract_fba_matrices, load_tight_bounds, apply_media,
    find_excluded_reactions, build_rxn_ec_mask, extract_pred, load_refmapping, load_ec)
from meteor_v8.repair import grow_support
sys.path.insert(0, f'{B}/meteor_v8/eval')
from baseline_io import resolve_baseline_pkl, BASELINE_SUFFIX

MO = f'{B}/{a.run}/meteor_out/dpz_vanilla'
GEMDIR = f'{B}/meteor_diag/curated_gems'
OUT = f'{B}/meteor_v8/results/toolcompare'
TAG = a.out or ('weakreal_ids_' + a.run.replace('meteor_v8_evw_', '').replace('_run', ''))
FULL = re.compile(r'^\d+\.\d+\.\d+\.\d+$')
if not os.path.isdir(MO): sys.exit(f'no such run dir: {MO}')

universal, allrxns, allmet = load_universal()
seedr2ec, _ = load_refmapping(data_dir()); seedr2ec = {k: v for k, v in seedr2ec.items() if v}
anc = load_ec(data_path('all_ancestors.txt')); mask = build_rxn_ec_mask(allrxns, seedr2ec, anc)
S, lb0, ub0 = extract_fba_matrices(universal, allrxns, reversed_trans=True)
NR = len(allrxns)

_pred_cols = None
def dpz_ec_scores(acc):
    global _pred_cols
    p = extract_pred(resolve_baseline_pkl('dpz', 'vanilla', acc, BASELINE_SUFFIX['dpz']), anc)
    if _pred_cols is None: _pred_cols = [str(c) for c in p.columns]
    sc = p.values.max(axis=0)
    return {str(_pred_cols[j]).split(':')[-1]: float(sc[j])
            for j in range(len(sc)) if FULL.match(str(_pred_cols[j]).split(':')[-1])}, p

def rxn_ecs(j):
    return {str(_pred_cols[e]).split(':')[-1] for e in np.where(mask[j] == 1)[0]
            if e < len(_pred_cols) and FULL.match(str(_pred_cols[e]).split(':')[-1])}

_gc = {}
def setup(gram):
    if gram in _gc: return _gc[gram]
    lt, ut = load_tight_bounds(data_path(f'tight_bounds_v6_{gram}.pkl'))
    lb = np.maximum(lb0, lt); ub = np.minimum(ub0, ut)
    oi = allrxns.index('biomass_GmPos' if gram == 'pos' else 'biomass_GmNeg')
    exc = find_excluded_reactions(S, lb, ub, allrxns, allrxns[oi])
    lb, ub, _, _ = apply_media(['default'], allrxns, lb, ub)
    _gc[gram] = (lb, ub, oi, exc); return _gc[gram]

def meteor_ecs(acc):
    sol = pickle.load(open(f'{MO}/meteor_sol_{acc}.pkl', 'rb'))
    act = np.array(sol['y_vals']) > 0.5
    E = set(); [E.update(rxn_ecs(j)) for j in np.where(act)[0]]
    meta = dict(n_active=int(sol.get('n_active', int(act.sum()))),
                n_repaired=(int(sol['n_repaired']) if 'n_repaired' in sol else None),
                status=sol.get('status'), biomass_flux=sol.get('biomass_flux'))
    return E, meta

def baseline_ecs(acc, gram, pred):
    lb, ub, oi, exc = setup(gram)
    ecmax = pred.values.max(axis=0); hot = set(np.where(ecmax >= 0.5)[0])
    draft = np.zeros(NR, bool)
    for j in range(NR):
        ei = np.where(mask[j] == 1)[0]
        if len(ei) and any(e in hot for e in ei): draft[j] = True
    avail = np.ones(NR, bool); avail[list(exc)] = False; core = draft & avail
    sup = grow_support(S, lb, ub, oi, avail, 0.1, core=core)
    model = core | (sup if sup is not None else np.zeros(NR, bool))
    E = set(); [E.update(rxn_ecs(j)) for j in np.where(model)[0]]
    return E

def gem_ecs(gem):
    m = cobra.io.read_sbml_model(f'{GEMDIR}/{gem}.xml'); ex = set()
    for r in m.reactions:
        an = r.annotation.get('ec-code') if hasattr(r, 'annotation') else None
        if not an: continue
        for x in (an if isinstance(an, list) else [an]):
            if FULL.match(str(x).strip()): ex.add(str(x).strip())
    return ex

def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0: return (float('nan'), float('nan'))
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 3), round(min(1.0, c + h), 3))

GEM2G = {'iML1515': ('GCF_058436375.1', 'neg', 'E.coli'),
         'STM_v1_0': ('GCF_000006945.2', 'neg', 'Salmonella'),
         'iYL1228': ('GCF_058435815.1', 'neg', 'K.pneumoniae'),
         'iJN1463': ('GCF_045571375.1', 'neg', 'P.putida'),
         'iYS854': ('GCF_045348045.1', 'pos', 'S.aureus'),
         'iYO844': ('GCF_058182495.1', 'pos', 'B.subtilis')}

rows = []
print(f'=== curated sub-threshold EC recovery + identities  [run={a.run}] ===')
print(f'{"organism":13s} {"|W|":>4s} {"M":>4s} {"B":>4s}  {"recM":>5s} {"95%CI":>13s}  {"recB":>5s} {"repaired":>8s}')
for gem, (gcf, gram, name) in GEM2G.items():
    scores, pred = dpz_ec_scores(gcf); G = gem_ecs(gem)
    strong = {e for e, s in scores.items() if s >= 0.5}
    weak = {e for e, s in scores.items() if 0 < s < 0.5}
    W = weak & G                                   # curated-reference sub-threshold ECs
    M, meta = meteor_ecs(gcf); Ba = baseline_ecs(gcf, gram, pred)
    mW, bW = M & W, Ba & W
    ci_m = wilson(len(mW), len(W)); ci_b = wilson(len(bW), len(W))
    rep = '-' if meta['n_repaired'] is None else str(meta['n_repaired'])
    print(f'{name:13s} {len(W):4d} {len(mW):4d} {len(bW):4d}  '
          f'{len(mW)/max(1,len(W)):5.3f} [{ci_m[0]:.3f},{ci_m[1]:.3f}]  '
          f'{len(bW)/max(1,len(W)):5.3f} {rep:>8s}')
    rows.append(dict(
        organism=name, gem=gem, gcf=gcf, gram=gram,
        n_strong=len(strong), n_weak=len(weak),
        denominator_n_weak_real=len(W),
        recovered=dict(meteor=len(mW), baseline=len(bW)),
        recovery=dict(meteor=round(len(mW)/max(1,len(W)),4), baseline=round(len(bW)/max(1,len(W)),4)),
        recovery_ci95=dict(meteor=ci_m, baseline=ci_b),
        precision=dict(meteor=round(len(M&G)/max(1,len(M)),4), baseline=round(len(Ba&G)/max(1,len(Ba)),4)),
        n_ec=dict(meteor=len(M), baseline=len(Ba)),
        solution=meta,
        ec_ids=dict(
            weak_real=sorted(W),
            meteor_recovered=sorted(mW),
            baseline_recovered=sorted(bW),
            meteor_only=sorted(mW - bW),
            missed_by_both=sorted(W - mW - bW)),
    ))

# organism-level bootstrap on the 6-organism mean (fixed seed, reproducible)
rng = np.random.default_rng(a.seed)
recM = np.array([r['recovery']['meteor'] for r in rows])
recB = np.array([r['recovery']['baseline'] for r in rows])
preM = np.array([r['precision']['meteor'] for r in rows])
preB = np.array([r['precision']['baseline'] for r in rows])
def boot(x):
    idx = rng.integers(0, len(x), size=(a.boot, len(x)))
    d = x[idx].mean(axis=1)
    return (round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4))
# pooled (sum of numerators / sum of denominators) as well as unweighted mean
num_m = sum(r['recovered']['meteor'] for r in rows); num_b = sum(r['recovered']['baseline'] for r in rows)
den = sum(r['denominator_n_weak_real'] for r in rows)
summary = dict(
    run=a.run, n_organisms=len(rows), bootstrap_draws=a.boot, seed=a.seed,
    mean_recovery=dict(meteor=round(float(recM.mean()),4), baseline=round(float(recB.mean()),4)),
    mean_recovery_boot_ci95=dict(meteor=boot(recM), baseline=boot(recB)),
    pooled_recovery=dict(meteor=round(num_m/max(1,den),4), baseline=round(num_b/max(1,den),4),
                         numerator=dict(meteor=num_m, baseline=num_b), denominator=den),
    pooled_recovery_ci95=dict(meteor=wilson(num_m, den), baseline=wilson(num_b, den)),
    mean_precision=dict(meteor=round(float(preM.mean()),4), baseline=round(float(preB.mean()),4)),
    mean_precision_boot_ci95=dict(meteor=boot(preM), baseline=boot(preB)),
    improved_in_n_organisms=int((recM > recB).sum()),
    any_repair_instrumentation=any(r['solution']['n_repaired'] is not None for r in rows),
)
print('\n--- SUMMARY ---')
print(f"  mean recovery   METEOR={summary['mean_recovery']['meteor']:.4f} "
      f"CI{summary['mean_recovery_boot_ci95']['meteor']}   "
      f"baseline={summary['mean_recovery']['baseline']:.4f} "
      f"CI{summary['mean_recovery_boot_ci95']['baseline']}")
print(f"  pooled recovery METEOR={summary['pooled_recovery']['meteor']:.4f} "
      f"({num_m}/{den})   baseline={summary['pooled_recovery']['baseline']:.4f} ({num_b}/{den})")
print(f"  mean precision  METEOR={summary['mean_precision']['meteor']:.4f} "
      f"baseline={summary['mean_precision']['baseline']:.4f}")
print(f"  improved in {summary['improved_in_n_organisms']}/{len(rows)} organisms")
print(f"  repair instrumentation present: {summary['any_repair_instrumentation']}")

os.makedirs(OUT, exist_ok=True)
json.dump(dict(summary=summary, per_organism=rows), open(f'{OUT}/{TAG}.json', 'w'), indent=1)
print(f'-> results/toolcompare/{TAG}.json')
