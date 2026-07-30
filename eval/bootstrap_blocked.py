#!/usr/bin/env python3
"""Day1-2 Analysis 3: clade-blocked bootstrap CIs for the 108-genome structural results.

A plain genome-level bootstrap treats the 108 assemblies as independent, which
they are not: several belong to the same genus/family, so their structural
records are correlated and the naive interval is too narrow.  This script
reports both intervals side by side:

  boot_ci       resample genomes with replacement (naive)
  blocked_ci    resample CLADES with replacement, taking every genome in a
                sampled clade (blocked / cluster bootstrap)

Blocking label is the deepest named clade in the NCBI lineage that is shared
by more than one panel genome, falling back to the genome itself when it is
the only member of its clade -- i.e. singletons stay singletons and only
genuinely related genomes get blocked together.

Runs on the committed per-genome records; no solver, no cluster needed.
"""
import json, glob, os, csv, argparse, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
ap = argparse.ArgumentParser()
ap.add_argument('--res', default=os.path.join(ROOT, 'results', 'table1'))
ap.add_argument('--panel', default=os.path.join(ROOT, 'cohorts', 'panel108_gram.tsv'))
ap.add_argument('--tax', default=os.path.join(ROOT, 'cohorts', 'gcf_taxonomy_gc.tsv'))
ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'bootstrap_blocked.json'))
ap.add_argument('--boot', type=int, default=10000)
ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()

panel = [l.split('\t')[0].strip() for l in open(a.panel) if l.strip()]
panel_set = set(panel)

# ---- clade labels ------------------------------------------------------
lineage = {}
with open(a.tax) as f:
    for row in csv.DictReader(f, delimiter='\t'):
        g = row['gcf'].strip()
        if g in panel_set:
            lineage[g] = [x.strip() for x in row['lineage'].split(';') if x.strip()]
missing = [g for g in panel if g not in lineage]
if missing: print(f'WARNING: no lineage for {len(missing)} genomes: {missing[:5]}')

# Deepest clade shared by >1 panel genome.  These NCBI lineage strings are
# "<organism name>;cellular organisms;Bacteria;<kingdom>;...;<genus>", i.e.
# element 0 is the organism itself and the ranks that follow run BROAD ->
# NARROW.  So drop element 0 and walk from the tail (narrowest) inwards; the
# first clade with more than one panel member is the tightest real grouping.
counts = {}
for g, ln in lineage.items():
    for tx in set(ln[1:]): counts[tx] = counts.get(tx, 0) + 1
UNINFORMATIVE = {'cellular organisms', 'Bacteria', 'Archaea'}
clade = {}
for g in panel:
    ln = lineage.get(g, [])
    lab = g                                    # fallback: its own block
    for tx in reversed(ln[1:]):
        if tx in UNINFORMATIVE: continue
        if counts.get(tx, 0) > 1: lab = tx; break
    clade[g] = lab

blocks = {}
for g in panel: blocks.setdefault(clade[g], []).append(g)
sizes = sorted((len(v) for v in blocks.values()), reverse=True)
print(f'genomes={len(panel)}  blocks={len(blocks)}  '
      f'largest block sizes={sizes[:8]}  singletons={sum(1 for s in sizes if s == 1)}')

# ---- structural records -----------------------------------------------
CFG = ['baseline_clean', 'baseline_dpz', 'baseline_enzbert',
       'meteor_clean', 'meteor_dpz', 'meteor_enzbert']
KEYS = ['n_selected', 'n_rxn', 'deadends', 'mass_imbal', 'mi_frac', 'fba_growth']
rec = {}
for f in sorted(glob.glob(os.path.join(a.res, 'table1_*.json'))):
    o = json.load(open(f)); g = o['gca']
    if g not in panel_set: continue
    rec[g] = o
have = [g for g in panel if g in rec]
print(f'records found: {len(have)}/{len(panel)}')
if not have: raise SystemExit('no per-genome records')

rng = np.random.default_rng(a.seed)
block_names = sorted(blocks)
gidx = {g: i for i, g in enumerate(have)}

def draws(values_by_genome):
    """(naive_draws, blocked_draws) of the mean, as arrays of length --boot."""
    v = np.array([values_by_genome[g] for g in have], float)
    n = len(v)
    naive = v[rng.integers(0, n, size=(a.boot, n))].mean(axis=1)
    # blocked: resample len(blocks) blocks with replacement
    memb = [[gidx[g] for g in blocks[b] if g in gidx] for b in block_names]
    memb = [m for m in memb if m]
    nb = len(memb)
    out = np.empty(a.boot)
    for t in range(a.boot):
        pick = rng.integers(0, nb, size=nb)
        idx = [i for p in pick for i in memb[p]]
        out[t] = v[idx].mean()
    return naive, out

def ci(d): return [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]

result = {'n_genomes': len(have), 'n_blocks': len(blocks), 'bootstrap_draws': a.boot,
          'seed': a.seed, 'block_size_distribution': sizes,
          'metrics': {}, 'deadend_reduction': {}}

print(f'\n{"config":18s} {"metric":11s} {"mean":>9s} {"naive 95% CI":>21s} {"clade-blocked 95% CI":>23s}')
print('-' * 86)
for c in CFG:
    for k in KEYS:
        vals = {g: rec[g][c][k] for g in have
                if isinstance(rec[g].get(c), dict) and k in rec[g][c]}
        if len(vals) != len(have): continue
        nv, bl = draws(vals)
        m = float(np.mean([vals[g] for g in have]))
        cn, cb = ci(nv), ci(bl)
        result['metrics'].setdefault(c, {})[k] = dict(
            mean=round(m, 4), boot_ci=cn, blocked_ci=cb,
            blocked_width=round(cb[1] - cb[0], 4), naive_width=round(cn[1] - cn[0], 4))
        print(f'{c:18s} {k:11s} {m:9.3f} [{cn[0]:9.3f},{cn[1]:9.3f}] [{cb[0]:9.3f},{cb[1]:9.3f}]')

# paired dead-end reduction, baseline -> METEOR, per predictor
print(f'\n{"predictor":12s} {"reduction%":>11s} {"naive 95% CI":>21s} {"clade-blocked 95% CI":>23s}')
print('-' * 70)
for b in ('clean', 'dpz', 'enzbert'):
    cb_, cm = f'baseline_{b}', f'meteor_{b}'
    gs = [g for g in have if isinstance(rec[g].get(cb_), dict) and isinstance(rec[g].get(cm), dict)
          and 'deadends' in rec[g][cb_] and 'deadends' in rec[g][cm]]
    if len(gs) != len(have): continue
    x = np.array([rec[g][cb_]['deadends'] for g in gs], float)
    y = np.array([rec[g][cm]['deadends'] for g in gs], float)
    def red(idx):
        xb, yb = x[idx].mean(), y[idx].mean()
        return 100.0 * (xb - yb) / xb if xb > 0 else np.nan
    n = len(gs)
    naive = np.array([red(rng.integers(0, n, size=n)) for _ in range(a.boot)])
    memb = [[gidx[g] for g in blocks[bn] if g in gidx] for bn in block_names]
    memb = [m for m in memb if m]; nb = len(memb)
    blk = np.empty(a.boot)
    for t in range(a.boot):
        pick = rng.integers(0, nb, size=nb)
        blk[t] = red(np.array([i for p in pick for i in memb[p]]))
    pt = 100.0 * (x.mean() - y.mean()) / x.mean()
    cn, cbi = ci(naive[~np.isnan(naive)]), ci(blk[~np.isnan(blk)])
    result['deadend_reduction'][b] = dict(
        baseline_mean=round(float(x.mean()), 3), meteor_mean=round(float(y.mean()), 3),
        reduction_pct=round(float(pt), 2), boot_ci=cn, blocked_ci=cbi)
    print(f'{b:12s} {pt:11.2f} [{cn[0]:9.2f},{cn[1]:9.2f}] [{cbi[0]:9.2f},{cbi[1]:9.2f}]')

json.dump(result, open(a.out, 'w'), indent=1)
print(f'\n-> {a.out}')
