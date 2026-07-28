"""Why does MAPred not gain at >=1 EC and why is TopEC's gain gap-fill?
Look at one Price genome's input score matrix and METEOR output for each method."""
import pickle, glob, os, sys, numpy as np, pandas as pd
F='/ibex/scratch/projects/c2014/kexin/funcarve'
TAX=sys.argv[1] if len(sys.argv)>1 else '1064539'
MOP=f'{F}/meteor_v8_evw_p2mu3_run/meteor_out_price'
print(f'=== Price genome tax_{TAX} ===\n')
hdr=f"{'method':9s} {'prot':>6s} {'ECcol':>6s} {'nz/prot':>8s} {'>=0.5/prot':>11s} {'weak/prot':>10s} {'ECs>=.5':>8s} {'ECs weak':>9s}"
print(hdr); print('-'*len(hdr))
stats={}
for m in ['CLEAN','DPZ','EnzBERT','GraphEC','MAPred','TopEC']:
    f=f'{F}/paperA_2026/baseline_preds/price22_{m}_clean/tax_{TAX}_{m}.pkl'
    if not os.path.exists(f):
        print(f'{m:9s}  MISSING'); continue
    df=pd.read_pickle(f); V=df.values.astype(float)
    nz=(V>0).sum(1); hi=(V>=0.5).sum(1); wk=((V>0)&(V<0.5)).sum(1)
    ec_hi=int(((V>=0.5).any(0)).sum()); ec_wk=int((((V>0)&(V<0.5)).any(0) & ~(V>=0.5).any(0)).sum())
    stats[m]=dict(ec_hi=ec_hi, ec_wk=ec_wk)
    print(f'{m:9s} {V.shape[0]:6d} {V.shape[1]:6d} {nz.mean():8.1f} {hi.mean():11.2f} {wk.mean():10.1f} {ec_hi:8d} {ec_wk:9d}')
print('\n  nz/prot = nonzero EC scores per protein; weak = 0<score<0.5')
print('  ECs>=.5 = distinct ECs the genome calls above threshold')
print('  ECs weak = distinct ECs present ONLY as weak signal (recoverable, never above 0.5)\n')
print('=== METEOR active-EC counts (meteor_out_price) ===')
for m in ['GraphEC','MAPred','TopEC']:
    hits=glob.glob(f'{MOP}/{m.lower()}/*{TAX}*.pkl')+glob.glob(f'{MOP}/{m}/*{TAX}*.pkl')
    if not hits: print(f'{m:9s} no meteor output found under {MOP}'); continue
    o=pickle.load(open(hits[0],'rb'))
    ae=o.get('active_ecs') if isinstance(o,dict) else None
    print(f'{m:9s} {os.path.basename(hits[0])}: keys={sorted(o.keys())[:8] if isinstance(o,dict) else type(o)}'
          + (f'  n_active_ecs={len(ae)}' if ae is not None else ''))
