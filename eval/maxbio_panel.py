#!/usr/bin/env python3
"""Recompute maxbio_active on the STORED selected sets for the whole panel,
using the same code path verify_and_repair uses.  This is the authoritative
check of whether the published selections grow under the MILP's own medium."""
import sys, os, json, pickle, argparse, numpy as np, warnings, logging
warnings.filterwarnings('ignore'); logging.getLogger('cobra').setLevel(logging.ERROR)
sys.path.insert(0, 'src')
from meteor_v8.utils import load_universal, extract_fba_matrices, load_tight_bounds, apply_media, data_path
from meteor_v8.repair import maxbio_active
ap = argparse.ArgumentParser()
ap.add_argument('--run', default='meteor_v8_evw_p2mu3_run')
ap.add_argument('--baseline', default='dpz')
a = ap.parse_args()
B = '/ibex/scratch/projects/c2014/kexin/funcarve'
PANEL = f'{B}/meteor_v7_run/downstream_results/panel108_gram.tsv'
panel = [l.split('\t')[0].strip() for l in open(PANEL) if l.strip()]
gram = {l.split('\t')[0].strip(): l.split('\t')[1].strip() for l in open(PANEL) if l.strip()}
u, allrxns, _ = load_universal()
print('len(allrxns)=%d  len(universal.reactions)=%d  ALIGNED=%s'
      % (len(allrxns), len(u.reactions), len(allrxns) == len(u.reactions)), flush=True)
S, lb0, ub0 = extract_fba_matrices(u, allrxns, reversed_trans=True)
cache = {}
def setup(g):
    if g in cache: return cache[g]
    lt, ut = load_tight_bounds(data_path(f'tight_bounds_v6_{g[:3]}.pkl'))
    lb = np.maximum(lb0, lt); ub = np.minimum(ub0, ut)
    lb, ub, _, _ = apply_media(['default'], allrxns, lb, ub)
    oi = allrxns.index('biomass_GmPos' if g == 'positive' else 'biomass_GmNeg')
    cache[g] = (lb, ub, oi); return cache[g]
rows = []
for i, acc in enumerate(panel):
    f = f'{B}/{a.run}/meteor_out/{a.baseline}_vanilla/meteor_sol_{acc}.pkl'
    if not os.path.exists(f): continue
    s = pickle.load(open(f, 'rb'))
    y = np.asarray(s['y_vals'], float); act = y > 0.5
    lb, ub, oi = setup(gram[acc])
    mb = maxbio_active(S, lb, ub, oi, act)
    rows.append(dict(gca=acc, n_active=int(act.sum()), maxbio_active=round(float(mb), 4),
                     milp_biomass=s.get('biomass_flux'),
                     stored_maxbio_core=s.get('maxbio_core'),
                     n_repaired=s.get('n_repaired')))
    print('[%3d/%d] %-18s n_act=%5d maxbio=%.4f' % (i+1, len(panel), acc, act.sum(), mb), flush=True)
mbs = np.array([r['maxbio_active'] for r in rows])
print('\n=== %s (n=%d) ===' % (a.run, len(rows)))
print('  maxbio_active >= 0.1 (gamma_min): %d/%d' % (int((mbs >= 0.0999).sum()), len(rows)))
print('  maxbio_active  < 1e-6           : %d/%d' % (int((mbs < 1e-6).sum()), len(rows)))
print('  mean=%.4f  median=%.4f  min=%.4f  max=%.4f' % (mbs.mean(), np.median(mbs), mbs.min(), mbs.max()))
bad = [r['gca'] for r in rows if r['maxbio_active'] < 0.0999]
if bad: print('  BELOW gamma_min (%d): %s' % (len(bad), bad))
out = f'{B}/meteor_v8/results/maxbio_panel_{a.run}_{a.baseline}.json'
json.dump(dict(run=a.run, n=len(rows), ge_gamma_min=int((mbs >= 0.0999).sum()),
               mean=round(float(mbs.mean()), 4), min=round(float(mbs.min()), 4),
               below=bad, per_genome=rows), open(out, 'w'), indent=1)
print('-> ' + out)
