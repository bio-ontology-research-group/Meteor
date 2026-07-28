"""
Producer vs non-producer specificity check for the KEGG-derived BGC
indicator-EC analysis, using antiSMASH-detected BGC regions on the SAME
109-GCF genome assemblies as ground truth (replaces the MIBiG-taxon-proxy
ground truth used in eval_bgc_specificity_gc.py).

Why this is stronger evidence than the MIBiG version: MIBiG producer labels
were matched via taxon ID (a different genome of the "same species" may or
may not carry the BGC), whereas antiSMASH was run directly on the exact
109-GCF assembly whose DPZ/METEOR predictions we are testing -- producer
status here is genome-identical, not species-proxied.

Classes tested: pks, nrps, terpene, saccharide, other. Excluded:
  - beta-lactam: antiSMASH found ZERO beta-lactam(betalactam) regions across
    all 108 genomes in this panel (only unrelated "betalactone" hits, a
    different scaffold) -- no producers exist to test against, so
    specificity is undefined for this class in this panel.
  - ribosomal/RiPP: excluded from the KEGG-derived indicator-EC dictionary
    itself (no corresponding KEGG pathway map for RiPP biosynthesis), so
    there is no indicator-EC side to compare against even though antiSMASH
    calls plenty of RiPP-like regions.
"""
import json, pickle
import pandas as pd
from pathlib import Path

W          = Path('/ibex/scratch/projects/c2014/kexin/funcarve/paperA_2026')
DPZ_DIR    = W / 'baseline_preds/dpz_vanilla_genome_collection'
METEOR_DIR = Path('/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8_evw_p2mu3_run/meteor_out/dpz_vanilla')
DICT_PATH  = W / 'results/bgc_class_kegg_ec_dict.json'
LABELS     = W / 'results/antismash_producer_labels_gc.tsv'
OUT        = Path('/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8/results/toolcompare/bgc_specificity_antismash_v8.tsv')

class_ec_dict = json.loads(DICT_PATH.read_text())['class_ec']

labels = pd.read_csv(LABELS, sep='\t', dtype={'gcf': str})
labels = labels.set_index('gcf')

gcf_set = sorted(
    (set(p.stem.replace('_DPZ', '') for p in DPZ_DIR.glob('*_DPZ.pkl'))
     & set(p.stem.replace('meteor_preds_','')
           for p in METEOR_DIR.glob('meteor_preds_*.pkl')))
    & set(labels.index)
)
print(f'GCFs available (DPZ+METEOR+antiSMASH): {len(gcf_set)}')

print('Loading DPZ baseline PKLs...')
dpz_cache = {}
for gcf in gcf_set:
    df = pickle.load(open(DPZ_DIR / f'{gcf}_DPZ.pkl', 'rb'))
    df.columns = [c.replace('EC:', '') for c in df.columns]
    dpz_cache[gcf] = set(df.columns[df.max(axis=0) > 0.5])

print('Loading METEOR preds PKLs...')
meteor_cache = {}
for gcf in gcf_set:
    p = pickle.load(open(METEOR_DIR / f'meteor_preds_{gcf}.pkl', 'rb'))
    meteor_cache[gcf] = set(p['active_ecs'])

CLASS_INDICATOR = {
    'pks': set(class_ec_dict['pks']),
    'nrps': set(class_ec_dict['nrps']),
    'terpene': set(class_ec_dict['terpene']),
    'saccharide': set(class_ec_dict['saccharide']),
    'other': set(class_ec_dict['other']),
}

rows = []
for cls, indicator_ecs in CLASS_INDICATOR.items():
    if not indicator_ecs or cls not in labels.columns:
        continue
    producer_gcfs = [g for g in gcf_set if labels.loc[g, cls] == 1]
    non_producer_gcfs = [g for g in gcf_set if labels.loc[g, cls] == 0]
    n_p, n_np = len(producer_gcfs), len(non_producer_gcfs)
    if n_p == 0 or n_np == 0:
        print(f'  skip {cls}: n_producer={n_p} n_nonproducer={n_np} (need both > 0)')
        continue

    def mean_recall(gcfs, cache):
        vals = [len(indicator_ecs & cache[g]) / len(indicator_ecs) for g in gcfs]
        return sum(vals) / len(vals)

    def any_fpr(gcfs, cache):
        hits = sum(1 for g in gcfs if len(indicator_ecs & cache[g]) > 0)
        return hits / len(gcfs)

    b_prod_recall = mean_recall(producer_gcfs, dpz_cache)
    m_prod_recall = mean_recall(producer_gcfs, meteor_cache)
    b_np_recall = mean_recall(non_producer_gcfs, dpz_cache)
    m_np_recall = mean_recall(non_producer_gcfs, meteor_cache)
    b_fpr = any_fpr(non_producer_gcfs, dpz_cache)
    m_fpr = any_fpr(non_producer_gcfs, meteor_cache)

    rows.append({
        'bgc_class': cls,
        'n_indicator_ecs': len(indicator_ecs),
        'n_producer_gcfs': n_p,
        'n_nonproducer_gcfs': n_np,
        'producer_recall_baseline': round(b_prod_recall, 4),
        'producer_recall_meteor': round(m_prod_recall, 4),
        'producer_recall_delta': round(m_prod_recall - b_prod_recall, 4),
        'nonproducer_mean_recall_baseline': round(b_np_recall, 4),
        'nonproducer_mean_recall_meteor': round(m_np_recall, 4),
        'nonproducer_recall_delta': round(m_np_recall - b_np_recall, 4),
        'nonproducer_any_fpr_baseline': round(b_fpr, 4),
        'nonproducer_any_fpr_meteor': round(m_fpr, 4),
        'specificity_baseline': round(1 - b_fpr, 4),
        'specificity_meteor': round(1 - m_fpr, 4),
        'excess_recall_baseline': round(b_prod_recall - b_np_recall, 4),
        'excess_recall_meteor': round(m_prod_recall - m_np_recall, 4),
    })

out = pd.DataFrame(rows)
out.to_csv(OUT, sep='\t', index=False)
print(out.to_string(index=False))
print(f'\nSaved: {OUT}')
