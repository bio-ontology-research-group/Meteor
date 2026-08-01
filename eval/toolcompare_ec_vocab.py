#!/usr/bin/env python3
"""Does the EC-vocabulary asymmetry explain METEOR's precision gap vs CarveMe?

The EC-level comparison is namespace-neutral in the sense that EC numbers are
shared, but the three EC SETS have different provenance:

  curated GEM : SBML ec-code annotations  (BiGG lineage)
  CarveMe     : SBML ec-code annotations  (BiGG lineage; its template universe
                is itself built from BiGG models, including these GEMs)
  METEOR      : ModelSEED reaction -> EC table (Reconstructor mapping)

So METEOR can emit ECs that no BiGG-annotated model can express, and those are
counted as false positives by construction rather than by biology. This script
quantifies that, and re-scores every method on the vocabulary INTERSECTION.

Reports the honest answer either way.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "src"))
from meteor_v8.utils import data_path, external_path, data_dir
import sys, os, json, re, pickle, cobra, numpy as np, pandas as pd
import warnings, logging; warnings.filterwarnings('ignore'); logging.getLogger('cobra').setLevel(logging.ERROR)
from meteor_v8.utils import load_refmapping, load_ec

B  = '/ibex/scratch/projects/c2014/kexin/funcarve'
MO = f'{B}/meteor_v8_evw_p2mu3_run/meteor_out/dpz_vanilla'
CV = f'{B}/paperA_2026/results/carveme_gc'
GEMDIR = external_path('curated_gems')
OUT = f'{B}/meteor_v8/results/toolcompare'
R2ECF = external_path('Unique_ModelSEED_Reaction_ECs.txt')
FULL = re.compile(r'^\d+\.\d+\.\d+\.\d+$')

def norm(s):
    s = str(s).strip()
    return s if FULL.match(s) else None

def model_ecs(p):
    m = cobra.io.read_sbml_model(p); ex = set()
    for r in m.reactions:
        e = r.annotation.get('ec-code') if hasattr(r, 'annotation') else None
        if not e: continue
        for x in (e if isinstance(e, list) else [e]):
            n = norm(x)
            if n: ex.add(n)
    return ex

def meteor_ecs(gcf):
    d = pickle.load(open(f'{MO}/meteor_preds_{gcf}.pkl', 'rb'))
    return {n for e in d.get('active_ecs', []) if (n := norm(e))}

cvfiles = os.listdir(CV)
def find_cv(acc):
    num = acc.split('_')[1]
    for f in cvfiles:
        if num in f and f.endswith('.xml'): return os.path.join(CV, f)

GEM2G = {'iML1515': ('GCF_058436375.1', 'E.coli'), 'STM_v1_0': ('GCF_000006945.2', 'Salmonella'),
 'iYL1228': ('GCF_058435815.1', 'K.pneumoniae'), 'iJN1463': ('GCF_045571375.1', 'P.putida'),
 'iYS854': ('GCF_045348045.1', 'S.aureus'), 'iYO844': ('GCF_058182495.1', 'B.subtilis')}

# ---- V_seed: every EC the ModelSEED->EC table can ever produce -------------
V_seed = set()
_e = pd.read_csv(R2ECF, sep='\t')
for _, r in _e.iterrows():
    n = norm(r['External ID'])
    if n: V_seed.add(n)
print(f'V_seed  (ModelSEED->EC table)          : {len(V_seed)} ECs')

# ---- V_bigg: every EC any BiGG-annotated model in this study expresses -----
V_bigg = set()
for gem in GEM2G: V_bigg |= model_ecs(f'{GEMDIR}/{gem}.xml')
for gcf, _ in GEM2G.values(): V_bigg |= model_ecs(find_cv(gcf))
print(f'V_bigg  (curated GEM + CarveMe ec-code): {len(V_bigg)} ECs')
INTER = V_seed & V_bigg
print(f'intersection                           : {len(INTER)} ECs')
print(f'  in V_seed only : {len(V_seed - V_bigg)}   in V_bigg only : {len(V_bigg - V_seed)}')

def metrics(P, G):
    i = len(P & G)
    prec = i / max(1, len(P)); rec = i / max(1, len(G))
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    jac = i / max(1, len(P | G))
    return dict(n=len(P), P=round(prec, 4), R=round(rec, 4),
                F1=round(f1, 4), J=round(jac, 4))

rows = []
print(f'\n{"organism":14s} {"arm":8s} | {"FULL vocabulary":34s} | {"INTERSECTION only":34s}')
print(f'{"":14s} {"":8s} | {"n":>5s} {"P":>6s} {"R":>6s} {"F1":>6s} {"J":>6s} | {"n":>5s} {"P":>6s} {"R":>6s} {"F1":>6s} {"J":>6s}')
for gem, (gcf, name) in GEM2G.items():
    G = model_ecs(f'{GEMDIR}/{gem}.xml'); M = meteor_ecs(gcf); C = model_ecs(find_cv(gcf))
    rec = dict(organism=name, gem=gem, gcf=gcf)
    for arm, S in (('meteor', M), ('carveme', C)):
        full = metrics(S, G)
        rest = metrics(S & INTER, G & INTER)
        rec[arm] = dict(full=full, intersection=rest)
        print(f'{name if arm=="meteor" else "":14s} {arm:8s} | '
              f'{full["n"]:5d} {full["P"]:6.3f} {full["R"]:6.3f} {full["F1"]:6.3f} {full["J"]:6.3f} | '
              f'{rest["n"]:5d} {rest["P"]:6.3f} {rest["R"]:6.3f} {rest["F1"]:6.3f} {rest["J"]:6.3f}')
    # where do METEOR's false positives live?
    fp = M - G
    rec['meteor_fp'] = dict(
        total=len(fp),
        outside_V_bigg=len(fp - V_bigg),          # unreachable for any BiGG model
        inside_V_bigg=len(fp & V_bigg),           # BiGG could express it; GEM did not
        frac_outside=round(len(fp - V_bigg) / max(1, len(fp)), 4))
    rec['gem_ec_outside_V_seed'] = len(G - V_seed)   # ground-truth ECs METEOR cannot emit
    rows.append(rec)

def agg(arm, scope, k):
    v = [r[arm][scope][k] for r in rows]; return np.mean(v), np.std(v, ddof=1)

print('\n=== MEANS (n=6) ===')
print(f'{"":22s} {"FULL vocabulary":>22s}   {"INTERSECTION only":>22s}')
for k in ('n', 'P', 'R', 'F1', 'J'):
    line = f'  {k:<4s}'
    for arm in ('meteor', 'carveme'):
        m, s = agg(arm, 'full', k); m2, s2 = agg(arm, 'intersection', k)
        fmt = '%7.1f±%5.1f' if k == 'n' else '%7.3f±%5.3f'
        line += f'  {arm[:4]}: ' + (fmt % (m, s)) + ' -> ' + (fmt % (m2, s2))
    print(line)

fpo = np.mean([r['meteor_fp']['frac_outside'] for r in rows])
print(f'\nMETEOR false positives outside the BiGG EC vocabulary: {100*fpo:.1f}% (mean)')
print(f'  mean FP total = {np.mean([r["meteor_fp"]["total"] for r in rows]):.0f}, '
      f'of which unreachable for any BiGG model = {np.mean([r["meteor_fp"]["outside_V_bigg"] for r in rows]):.0f}')
print(f'curated-GEM ECs that the ModelSEED table cannot express: '
      f'{np.mean([r["gem_ec_outside_V_seed"] for r in rows]):.0f} per organism (mean)')

dP_full = agg('carveme','full','P')[0] - agg('meteor','full','P')[0]
dP_int  = agg('carveme','intersection','P')[0] - agg('meteor','intersection','P')[0]
print(f'\nPRECISION GAP (CarveMe - METEOR): full {dP_full:.3f}  ->  intersection {dP_int:.3f}'
      f'   ({100*(1-dP_int/dP_full):.0f}% of the gap closes)' if dP_full else '')

json.dump(dict(V_seed=len(V_seed), V_bigg=len(V_bigg), intersection=len(INTER),
               per_organism=rows), open(f'{OUT}/panelB_ec_vocab.json', 'w'), indent=1)
print(f'-> results/toolcompare/panelB_ec_vocab.json')
