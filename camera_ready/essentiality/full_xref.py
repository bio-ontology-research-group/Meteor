import sys, json, pickle
sys.path.insert(0, '/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/eval')
from _env import *  # noqa
import cobra

GEMDIR = '/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/code_snapshot/data/external/curated_gems'
CURATED_GEM = {"salmonella": "STM_v1_0", "kpneumoniae": "iYL1228", "pputida": "iJN1463"}

bigg2seed, kegg2seed = pickle.load(open('/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/feasibility_essentiality/xref/bigg_kegg_to_seed.pkl','rb'))

curated_seed_ids_full = {}
coverage_report = {}
for org, gem in CURATED_GEM.items():
    m = cobra.io.read_sbml_model(f"{GEMDIR}/{gem}.xml")
    seeds = set()
    n_direct = 0; n_via_bigg = 0; n_via_kegg = 0; n_unresolved = 0
    for r in m.reactions:
        ann = r.annotation or {}
        found_this = False
        sid = ann.get("seed.reaction")
        if sid:
            for s in (sid if isinstance(sid, list) else [sid]):
                seeds.add(s); found_this = True
            if found_this: n_direct += 1
        bg = ann.get("bigg.reaction")
        if bg:
            hit = False
            for b in (bg if isinstance(bg, list) else [bg]):
                if b in bigg2seed:
                    seeds.update(bigg2seed[b]); hit = True
            if hit and not found_this: n_via_bigg += 1; found_this = True
        kg = ann.get("kegg.reaction")
        if kg:
            hit = False
            for k in (kg if isinstance(kg, list) else [kg]):
                if k in kegg2seed:
                    seeds.update(kegg2seed[k]); hit = True
            if hit and not found_this: n_via_kegg += 1; found_this = True
        if not found_this:
            n_unresolved += 1
    curated_seed_ids_full[org] = seeds
    coverage_report[org] = dict(n_rxn=len(m.reactions), n_direct=n_direct, n_via_bigg=n_via_bigg,
                                 n_via_kegg=n_via_kegg, n_unresolved=n_unresolved, n_seed_ids_total=len(seeds))
    print(org, gem, coverage_report[org], flush=True)

# now re-check the 95 "no-backup" reactions and the 25332 control group against the enriched sets
nb = json.load(open('/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/feasibility_essentiality/results/forensic/nobackup_curated_precise.json'))
ctrl = json.load(open('/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/feasibility_essentiality/results/forensic/control_group_meteor_selected.json'))

def recheck(rows, label):
    n = len(rows)
    present = 0
    for r in rows:
        org = r['org']
        sid = r.get('seed_id') or (r['rxn'][:-2] if r['rxn'].endswith(('_c','_e','_p')) else r['rxn'])
        if sid in curated_seed_ids_full[org]:
            present += 1
    print(f"{label}: n={n}  present(enriched)={present} ({100*present/n:.1f}%)  absent(enriched)={n-present} ({100*(n-present)/n:.1f}%)")
    return present, n

p95, n95 = recheck(nb['all_reactions'], "95 no-backup set")
pctrl, nctrl = recheck(ctrl['rows'], "control group (kept)")

json.dump(dict(coverage_report=coverage_report,
               nobackup_enriched=dict(n=n95, present=p95, pct_present=100*p95/n95, pct_absent=100*(n95-p95)/n95),
               control_enriched=dict(n=nctrl, present=pctrl, pct_present=100*pctrl/nctrl, pct_absent=100*(nctrl-pctrl)/nctrl)),
          open('/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/feasibility_essentiality/results/forensic/enriched_xref_recheck.json','w'), indent=1)
print("done")
