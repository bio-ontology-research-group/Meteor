"""Compare the two TopEC prediction sets: tax-named (used by the paper) vs
topec_price_new (GCA-named). Is the 1-EC-per-protein a property of TopEC or of
our conversion?"""
import pandas as pd, numpy as np, glob, os
F='/ibex/scratch/projects/c2014/kexin/funcarve'
def prof(f, tag):
    df = pd.read_pickle(f); V = df.values.astype(float)
    nz = (V > 0).sum(1)
    vals = V[V > 0]
    print(f'{tag}')
    print(f'   file      {os.path.basename(f)}')
    print(f'   shape     {V.shape[0]} proteins x {V.shape[1]} ECs')
    print(f'   nonzero/protein  mean={nz.mean():.2f}  min={nz.min()}  max={nz.max()}')
    print(f'   distinct nonzero counts: {sorted(set(nz.tolist()))[:10]}')
    if vals.size:
        print(f'   nonzero values   min={vals.min():.4f} max={vals.max():.4f} '
              f'mean={vals.mean():.4f}  all==1.0? {bool((vals==1.0).all())}')
        q = np.percentile(vals, [10,50,90])
        print(f'   value deciles    p10={q[0]:.3f} p50={q[1]:.3f} p90={q[2]:.3f}')
    print()

print('=== A. tax-named set used by the paper pipeline ===')
a = sorted(glob.glob(f'{F}/paperA_2026/baseline_preds/price22_TopEC_clean/tax_*_TopEC.pkl'))
print(f'({len(a)} files)')
if a: prof(a[0], 'A[0]')

print('=== B. topec_price_new (GCA-named) ===')
b = sorted(glob.glob(f'{F}/topec_price_new/*_TopEC.pkl'))
print(f'({len(b)} files)')
if b: prof(b[0], 'B[0]')

print('=== C. for reference, MAPred from the same pipeline ===')
c = sorted(glob.glob(f'{F}/paperA_2026/baseline_preds/price22_MAPred_clean/tax_*_MAPred.pkl'))
if c: prof(c[0], 'C[0]')
