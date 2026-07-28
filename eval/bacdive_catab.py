"""BacDive metabolite utilisation as a NETWORK-CONTENT test (no FBA, no medium).

For each (genome, compound) assertion in BacDive, ask whether the reconstructed
network contains machinery that consumes the compound. This avoids the medium
confound of an FBA test: METEOR selects a reaction whenever its evidence makes
the cost negative, whether or not that reaction carries flux under the fixed
minimal medium, so catabolic capability is not pruned merely for being unused.

Arms: METEOR (y > 0.5) vs the threshold baseline (any EC scored >= tau).
"""
import sys, os, csv, glob, json, pickle, argparse, collections
import numpy as np
V6 = "/ibex/user/niuk0a/funcarve/cobra/v6"
sys.path.insert(0, V6); os.chdir(V6)
sys.path.insert(0, "/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8/eval")

ap = argparse.ArgumentParser()
ap.add_argument("--chunk", type=int, required=True)
ap.add_argument("--n_chunks", type=int, default=40)
ap.add_argument("--tau", type=float, default=0.5)
ap.add_argument("--outdir", default="/ibex/scratch/projects/c2014/kexin/funcarve/"
                                    "meteor_v8/results/bacdive_catab")
a = ap.parse_args()
os.makedirs(a.outdir, exist_ok=True)
OUT = os.path.join(a.outdir, "catab_%03d.json" % a.chunk)
if os.path.exists(OUT):
    print("done"); sys.exit(0)

from src.v6utils import (load_universal, extract_fba_matrices, load_refmapping,
                         load_ec, extract_pred, build_rxn_ec_mask)

F = "/ibex/scratch/projects/c2014/kexin/funcarve"
MET = f"{F}/meteor_v8_evw_p2mu3_run/meteor_out_bacdive/dpz_vanilla"
BASE = f"{F}/dpec2_result/result_bacdive"
BD = f"{F}/meteor_v7_run/downstream_results/bacdive_meta"
FBA_KINDS = {"carbon source", "assimilation", "growth"}

universal, allrxns, allmet = load_universal()
S, lb, ub = extract_fba_matrices(universal, allrxns, reversed_trans=True)
seedr2ec, _ = load_refmapping(f"{V6}/data"); seedr2ec = {k: v for k, v in seedr2ec.items() if v}
anc = load_ec(f"{V6}/data/all_ancestors.txt")
mask = build_rxn_ec_mask(allrxns, seedr2ec, anc)

# metabolite id -> row index; a reaction "handles" a compound if it has a nonzero
# stoichiometric coefficient for it and is not the compound's own exchange.
met_ix = {m.id: i for i, m in enumerate(universal.metabolites)}
Scsr = S.tocsr() if hasattr(S, "tocsr") else None

def handlers(cpd):
    """reaction indices touching any compartment form of this SEED compound"""
    rows = [met_ix[m] for m in met_ix if m.startswith(cpd + "_")]
    if not rows: return set()
    idx = set()
    for r in rows:
        row = Scsr.getrow(r) if Scsr is not None else None
        if row is None: continue
        idx.update(int(x) for x in row.indices)
    return {j for j in idx if not allrxns[j].startswith(("EX_", "DM_", "SK_"))}

# BacDive compound -> SEED cpd
c2s = {}
for r in csv.DictReader(open(f"{BD}/bacdive_to_seed_mapping.tsv"), delimiter="\t"):
    if r.get("seed_cpd_id"):
        c2s[r["bacdive_compound"].strip().lower()] = r["seed_cpd_id"].strip()

assert_by_genome = collections.defaultdict(list)
for r in csv.DictReader(open(f"{BD}/bacdive_met_util.tsv"), delimiter="\t"):
    if r["kind"] not in FBA_KINDS: continue
    if r["ability"] not in ("+", "-"): continue
    cpd = c2s.get(r["compound_name"].strip().lower())
    if not cpd: continue
    assert_by_genome[r["genome_accession"].split(".")[0]].append(
        (r["compound_name"], cpd, 1 if r["ability"] == "+" else 0))

sols = {os.path.basename(f)[len("meteor_sol_"):-4]: f
        for f in glob.glob(f"{MET}/meteor_sol_*.pkl")}
genomes = sorted(g for g in sols if g.split(".")[0] in assert_by_genome)
genomes = genomes[a.chunk::a.n_chunks]
print(f"chunk {a.chunk}: {len(genomes)} genomes", flush=True)

hcache = {}
rows = []
for gi, g in enumerate(genomes):
    try:
        yv = np.array(pickle.load(open(sols[g], "rb")).get("y_vals", []))
    except Exception:
        continue
    met_set = set(np.where(yv > 0.5)[0].tolist())
    bp = glob.glob(f"{BASE}/*{g}*.pkl")
    if not bp: continue
    try:
        pred = extract_pred(bp[0], anc)
    except Exception:
        continue
    hit = (pred.values >= a.tau).any(axis=0)
    base_set = {j for j in range(len(allrxns))
                if (mask[j] == 1).any() and hit[mask[j] == 1].any()}
    for cname, cpd, lab in assert_by_genome[g.split(".")[0]]:
        if cpd not in hcache: hcache[cpd] = handlers(cpd)
        H = hcache[cpd]
        if not H: continue
        rows.append(dict(genome=g, compound=cname, cpd=cpd, label=lab,
                         n_h=len(H),
                         meteor=len(met_set & H), base=len(base_set & H)))
    if gi % 20 == 0: print(f"  {gi}/{len(genomes)}", flush=True)

json.dump(rows, open(OUT, "w"))
print(f"written {OUT}: {len(rows)} assertions", flush=True)
