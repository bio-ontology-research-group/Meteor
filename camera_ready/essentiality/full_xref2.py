import json, pickle

curated_seed_ids_full_report = json.load(open('/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/feasibility_essentiality/results/forensic/enriched_xref_recheck.json'))['coverage_report']

nb = json.load(open('/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/feasibility_essentiality/results/forensic/nobackup_curated_precise.json'))
ctrl = json.load(open('/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/feasibility_essentiality/results/forensic/control_group_meteor_selected.json'))

# rebuild enriched seed sets the same way (script kept them in memory only; recompute quickly is cheap, reuse cached pickle approach)
import sys
sys.path.insert(0, '/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/eval')
from _env import *
import cobra
GEMDIR = '/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/code_snapshot/data/external/curated_gems'
CURATED_GEM = {"salmonella": "STM_v1_0", "kpneumoniae": "iYL1228", "pputida": "iJN1463"}
bigg2seed, kegg2seed = pickle.load(open('/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/feasibility_essentiality/xref/bigg_kegg_to_seed.pkl','rb'))
curated_seed_ids_full = {}
for org, gem in CURATED_GEM.items():
    m = cobra.io.read_sbml_model(f"{GEMDIR}/{gem}.xml")
    seeds = set()
    for r in m.reactions:
        ann = r.annotation or {}
        sid = ann.get("seed.reaction")
        if sid:
            seeds.update(sid if isinstance(sid, list) else [sid])
        bg = ann.get("bigg.reaction")
        if bg:
            for b in (bg if isinstance(bg, list) else [bg]):
                if b in bigg2seed: seeds.update(bigg2seed[b])
        kg = ann.get("kegg.reaction")
        if kg:
            for k in (kg if isinstance(kg, list) else [kg]):
                if k in kegg2seed: seeds.update(kegg2seed[k])
    curated_seed_ids_full[org] = seeds

def recheck_combined(rows, label):
    n = len(rows)
    present = 0
    for r in rows:
        org = r['org']
        sid = r.get('seed_id') or (r['rxn'][:-2] if r['rxn'].endswith(('_c','_e','_p')) else r['rxn'])
        by_enriched_id = sid in curated_seed_ids_full[org]
        by_ec = r.get('in_curated_by_ec') is True
        if by_enriched_id or by_ec:
            present += 1
    print(f"{label}: n={n}  present(enriched_id OR ec)={present} ({100*present/n:.1f}%)  absent={n-present} ({100*(n-present)/n:.1f}%)")
    return present, n

p95, n95 = recheck_combined(nb['all_reactions'], "95 no-backup set")
pctrl, nctrl = recheck_combined(ctrl['rows'], "control group (kept)")

json.dump(dict(nobackup_enriched_combined=dict(n=n95, present=p95, pct_present=round(100*p95/n95,1), pct_absent=round(100*(n95-p95)/n95,1)),
               control_enriched_combined=dict(n=nctrl, present=pctrl, pct_present=round(100*pctrl/nctrl,1), pct_absent=round(100*(nctrl-pctrl)/nctrl,1))),
          open('/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/feasibility_essentiality/results/forensic/enriched_xref_recheck_combined.json','w'), indent=1)
print("done")
