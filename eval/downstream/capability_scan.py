#!/usr/bin/env python3
"""Physiological-capability scan: for each (genome, capability) test whether the
baseline predicts the capability (max indicator-EC confidence > tau) and whether
METEOR's MILP activates >=1 universal reaction implementing it. Reports, per
capability, how many predicted assemblies METEOR metabolically recovers.

Publication-ready, parameterized. IMPORTANT provenance note: the capability->EC
dictionary MUST come from a public, documented source (e.g. KEGG pathway/module
ECs), NOT a hand-curated private list. Pass it via --capability_ec.

Required inputs:
  --capability_ec  JSON {capability_name: [indicator ECs]} (public source)
  --baseline_dir   {gcf}_{suffix}.pkl   ([P x EC] DataFrame)
  --meteor_dir     meteor_sol_{gcf}.pkl     (dict with y_vals = MILP reaction on/off)
  --universal      universal.pickle (cobra model; for EC->reaction index map)
  --seedr2ec       seedr2ec.pkl (reaction->EC mapping)
"""
import argparse, json, pickle
from pathlib import Path
import pandas as pd

SUFFIX = {"clean": "CLEAN_confidence", "dpz": "DPZ", "enzbert": "enzbert"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", choices=list(SUFFIX), required=True)
    ap.add_argument("--variant", default="vanilla")
    ap.add_argument("--baseline_dir", required=True)
    ap.add_argument("--meteor_dir", required=True)
    ap.add_argument("--capability_ec", required=True, help="public {capability: [ECs]} JSON")
    ap.add_argument("--universal", required=True)
    ap.add_argument("--seedr2ec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau", type=float, default=0.5)
    a = ap.parse_args()
    suf = SUFFIX[a.baseline]

    cap_ec = json.loads(Path(a.capability_ec).read_text())
    if "capabilities" in cap_ec:
        cap_ec = cap_ec["capabilities"]
    elif "class_ec" in cap_ec:            # accept the bgc-dict shape too
        cap_ec = cap_ec["class_ec"]

    u = pickle.load(open(a.universal, "rb"))
    allrxns = [r.id for r in u.reactions]
    seedr2ec = {k: v for k, v in pickle.load(open(a.seedr2ec, "rb")).items() if v}
    ec_to_idx = {}
    for idx, rid in enumerate(allrxns):
        base = rid[:-2] if rid.endswith(("_c", "_e", "_p")) else rid
        for ec in seedr2ec.get(base, []):
            ec_to_idx.setdefault(ec, []).append(idx)
    cap_rxns = {name: (ecs, sorted({i for ec in ecs for i in ec_to_idx.get(ec, [])}))
                for name, ecs in cap_ec.items()}

    bdir, mdir = Path(a.baseline_dir), Path(a.meteor_dir)
    sol_paths = sorted(mdir.glob("meteor_sol_*.pkl"))
    print(f"scanning {len(sol_paths)} genomes x {len(cap_rxns)} capabilities")
    rows = []
    for i, sp in enumerate(sol_paths, 1):
        gcf = sp.name[len("meteor_sol_"):-4]
        bp = bdir / f"{gcf}_{suf}.pkl"
        if not bp.exists():
            continue
        pred = pd.read_pickle(bp)
        cols = {str(c).replace("EC:", "").strip(): c for c in pred.columns}
        y = pickle.load(open(sp, "rb"))["y_vals"]
        for name, (ecs, idxs) in cap_rxns.items():
            mc = max((float(pred[cols[ec]].max()) for ec in ecs if ec in cols), default=0.0)
            n_act = sum(1 for j in idxs if y[j] > 0.5)
            rows.append(dict(gcf=gcf, capability=name, max_indicator_conf=round(mc, 4),
                             predicted=mc > a.tau, n_universal_rxns=len(idxs),
                             n_meteor_active=n_act, any_meteor_active=n_act > 0))
    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, sep="\t", index=False)
    print(f"saved {a.out} rows={len(df)}")
    print("\n%-40s %10s %10s %8s" % ("capability", "predicted", "recovered", "missed"))
    for name in cap_rxns:
        sub = df[df.capability == name]
        npred = int(sub.predicted.sum())
        nrec = int((sub.predicted & sub.any_meteor_active).sum())
        if npred:
            print("%-40s %10d %10d %8d" % (name, npred, nrec, npred - nrec))


if __name__ == "__main__":
    main()
