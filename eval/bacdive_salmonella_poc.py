#!/usr/bin/env python3
"""BacDive proof-of-concept on Salmonella (GCF_000006945.2): do METEOR's active ECs match the
enzyme-test phenotypes (+/-)? And are the '+' ECs weak-signal (dpz<0.5) recovered by METEOR?"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "src"))
from meteor_v8.utils import data_path, data_dir
import sys,os,pickle,glob,numpy as np
from meteor_v8.utils import extract_pred,load_ec

from baseline_io import resolve_baseline_pkl,BASELINE_SUFFIX
anc=load_ec(data_path('all_ancestors.txt'))
GCF='GCF_000006945.2'
# curated (test, kind, BacDive ability) -> candidate ECs
TESTS=[
 ('nitrate reduction','+',['1.7.5.1','1.7.99.4','1.9.6.1','1.7.2.1']),
 ('nitrite reduction','+',['1.7.1.4','1.7.2.1','1.7.1.15','1.7.2.2']),
 ('arginine dihydrolase','+',['3.5.3.6','3.5.3.1']),
 ('urea hydrolysis (urease)','-',['3.5.1.5']),
 ('gelatin hydrolysis (gelatinase)','-',['3.4.24.3','3.4.24.-','3.4.21.-']),
]
# v8 METEOR active ecs
mp=f'/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8_evw_p2mu3_run/meteor_out/dpz_vanilla/meteor_preds_{GCF}.pkl'
d=pickle.load(open(mp,'rb')); active=set(str(e).split(':')[-1] for e in d['active_ecs'])
muted=set(str(e).split(':')[-1] for e in d.get('muted_ecs',[]))
# baseline dpz max score per EC
bp=resolve_baseline_pkl('dpz','vanilla',GCF,BASELINE_SUFFIX['dpz'])
pred=extract_pred(bp,anc)
cols=[str(c).split(':')[-1] for c in pred.columns]; mx=pred.values.max(axis=0)
base={cols[j]:float(mx[j]) for j in range(len(cols))}
print(f'Salmonella {GCF}: |active_ecs|={len(active)}\n')
print(f"{'test':32s} {'BacDive':8s} {'EC':10s} {'base':>6s} {'active?':8s} {'verdict'}")
for name,ability,ecs in TESTS:
    hit=False; best=None
    for ec in ecs:
        inact = ec in active
        b = base.get(ec,0.0)
        if inact: hit=True
        # print each candidate EC
        print(f"{name:32s} {ability:8s} {ec:10s} {b:6.3f} {'YES' if inact else 'no':8s}", end='')
        if inact and ability=='+': print('  match(+ present)'+('  [WEAK-RECOVERED]' if 0<b<0.5 else ''))
        elif (not inact) and ability=='-': print('  match(- absent)')
        elif inact and ability=='-': print('  MISMATCH(- but present)')
        else: print('')
    # summary per test
    match = (hit and ability=='+') or ((not hit) and ability=='-')
    print(f"  --> TEST {name}: BacDive {ability}, METEOR {'present' if hit else 'absent'} => {'MATCH' if match else 'MISMATCH'}\n")
