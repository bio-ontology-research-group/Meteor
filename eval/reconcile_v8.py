import pickle,glob,os,numpy as np,csv
ROOT='/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8_evw_p2mu3_run/meteor_out'
TOL=1e-6
rows=[]
configs=sorted(glob.glob(f'{ROOT}/*/'))
for cd in configs:
    cfg=os.path.basename(cd.rstrip('/'))
    for p in sorted(glob.glob(f'{cd}/meteor_sol_*.pkl')):
        gca=os.path.basename(p)[len('meteor_sol_'):-4]
        try: o=pickle.load(open(p,'rb'))
        except Exception as e: print('ERR',p,e); continue
        yv=np.asarray(o['y_vals']); v=np.asarray(o['v_vals'])
        ya=yv>0.5; fl=(np.abs(v)>TOL)&(~ya); recon=ya|(np.abs(v)>TOL)
        n0=int(ya.sum()); nl=int(fl.sum()); nr=int(recon.sum())
        rows.append(dict(config=cfg,gca=gca,n_active_orig=n0,n_leak=nl,n_active_recon=nr,
                         bio=float(o.get('biomass_flux',0)),status=str(o.get('status',''))))
        # write sidecar with reconciled active set (bool) for downstream growth/posterior
        np.save(p.replace('meteor_sol_','recon_active_').replace('.pkl','.npy'), recon)
outcsv=f'{ROOT}/../reconcile_summary.csv'
with open(outcsv,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print('wrote',outcsv,'n=',len(rows))
import numpy as np
def stats(sel,name):
    r=[x for x in rows if sel(x)]
    if not r: return
    nl=np.array([x['n_leak'] for x in r]); n0=np.array([x['n_active_orig'] for x in r]); nr=np.array([x['n_active_recon'] for x in r])
    frac=nl/np.maximum(n0,1)
    print(f'{name:16s} n={len(r):4d} | leak: mean {nl.mean():6.1f} med {int(np.median(nl)):4d} max {nl.max():4d} | %undercount mean {100*frac.mean():.2f}% | genomes w/ leak>0: {int((nl>0).sum())} ({100*(nl>0).mean():.0f}%) | n_active {n0.mean():.0f}->{nr.mean():.0f}')
print()
for cfg in sorted(set(x['config'] for x in rows)):
    stats(lambda x,c=cfg: x['config']==c, cfg)
print()
stats(lambda x: True, 'ALL')
# worst offenders
worst=sorted(rows,key=lambda x:-x['n_leak'])[:12]
print('\nworst leak genomes:')
for x in worst: print(f"  {x['config']:16s} {x['gca']:18s} leak={x['n_leak']:4d} n_active {x['n_active_orig']}->{x['n_active_recon']} bio={x['bio']:.3f}")
