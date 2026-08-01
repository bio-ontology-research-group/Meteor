#!/usr/bin/env python3
"""Deposit the two EC vocabularies behind the CarveMe comparison so the
asymmetry claim in main text Section 3.5 is checkable.

Writes:
  ec_vocabulary_map.tsv      one row per EC: which vocabulary contains it
  ec_fp_by_organism.tsv      METEOR's false positives, split by whether the
                             reference vocabulary could express them at all
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "src"))
from meteor_v8.utils import external_path
import os, re, csv, pickle, cobra, pandas as pd
import warnings, logging; warnings.filterwarnings('ignore'); logging.getLogger('cobra').setLevel(logging.ERROR)

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

V_seed = {n for x in pd.read_csv(R2ECF, sep='\t')['External ID'] if (n := norm(x))}
gem_ecs = {g: model_ecs(f'{GEMDIR}/{g}.xml') for g in GEM2G}
cv_ecs  = {n: model_ecs(find_cv(a)) for a, n in GEM2G.values()}
V_gem  = set().union(*gem_ecs.values())
V_cv   = set().union(*cv_ecs.values())
V_bigg = V_gem | V_cv

os.makedirs(OUT, exist_ok=True)
p1 = f'{OUT}/ec_vocabulary_map.tsv'
with open(p1, 'w', newline='') as fh:
    w = csv.writer(fh, delimiter='\t')
    w.writerow(['ec', 'in_modelseed_vocab', 'in_reference_vocab',
                'in_curated_gems', 'in_carveme_outputs'])
    for ec in sorted(V_seed | V_bigg, key=lambda s: [int(t) for t in s.split('.')]):
        w.writerow([ec, int(ec in V_seed), int(ec in V_bigg),
                    int(ec in V_gem), int(ec in V_cv)])
print(f'{p1}: {len(V_seed | V_bigg)} rows')
print(f'  modelseed vocab {len(V_seed)} | reference vocab {len(V_bigg)} '
      f'(curated GEMs {len(V_gem)}, CarveMe {len(V_cv)})')
print(f'  reference \\ modelseed = {len(V_bigg - V_seed)}  '
      f'modelseed \\ reference = {len(V_seed - V_bigg)}')

p2 = f'{OUT}/ec_fp_by_organism.tsv'
with open(p2, 'w', newline='') as fh:
    w = csv.writer(fh, delimiter='\t')
    w.writerow(['organism', 'gem', 'ec', 'expressible_in_reference_vocab'])
    tot = out = 0
    for gem, (gcf, name) in GEM2G.items():
        for ec in sorted(meteor_ecs(gcf) - gem_ecs[gem],
                         key=lambda s: [int(t) for t in s.split('.')]):
            inref = ec in V_bigg
            w.writerow([name, gem, ec, int(inref)])
            tot += 1; out += (not inref)
print(f'{p2}: {tot} rows, {out} ({100*out/tot:.1f}%) outside the reference vocabulary')
