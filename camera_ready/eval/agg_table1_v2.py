"""Aggregate psb_revision/results/table1_v2/table1_{gca}.json over the 108 panel -> table1_v2_meansd.json.

mean, sd (sample, n-1), n per arm for n_selected, n_rxn, deadends, mass_imbal, mi_frac (old, boundary-inclusive),
n_internal, mi_int, mi_frac_int, mi_int_missing_formula, mi_int_genuine, fba_growth; plus n_growing.
Arms: baseline_{clean,dpz,enzbert}, meteor_{clean,dpz,enzbert}, abl_full, abl_skelonly, abl_uniform.
Reimplementation of the laptop agg_table1_sd.py logic (not present on ibex).
"""
import os, sys, json, glob, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env  # noqa: E402

RES = f"{_env.RESULTS}/table1_v2"
OUT = f"{_env.RESULTS}/table1_v2_meansd.json"
ARMS = [f"{p}_{b}" for p in ("baseline", "meteor") for b in ("clean", "dpz", "enzbert")] + \
       ["abl_full", "abl_skelonly", "abl_uniform"]
KEYS = ("n_selected", "n_rxn", "deadends", "mass_imbal", "mi_frac", "n_internal", "mi_int",
        "mi_frac_int", "mi_int_missing_formula", "mi_int_genuine", "fba_growth")


def main():
    panel = [g for g, _ in _env.panel_rows()]
    recs = {}
    for g in panel:
        f = f"{RES}/table1_{g}.json"
        if os.path.exists(f):
            recs[g] = json.load(open(f))
    missing = [g for g in panel if g not in recs]
    out = {"n_panel": len(panel), "n_records": len(recs), "missing_records": missing, "arms": {}}
    for arm in ARMS:
        vals = {k: [] for k in KEYS}
        used = []
        for g, o in recs.items():
            v = o.get(arm)
            if not isinstance(v, dict) or "err" in v:
                continue
            used.append(g)
            for k in KEYS:
                vals[k].append(v[k])
        n = len(used)
        summ = {"n": n, "missing": sorted(set(recs) - set(used))}
        for k in KEYS:
            x = vals[k]
            if not x:
                continue
            summ[k] = {"mean": round(st.mean(x), 5), "sd": round(st.stdev(x), 5) if n > 1 else 0.0,
                       "min": min(x), "max": max(x)}
        if vals["fba_growth"]:
            summ["n_growing"] = sum(1 for x in vals["fba_growth"] if x and x > 1e-6)
        if vals["mi_int"] and sum(vals["mi_int"]):
            summ["missing_formula_share_pooled"] = round(sum(vals["mi_int_missing_formula"]) / sum(vals["mi_int"]), 4)
        out["arms"][arm] = summ
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"records {len(recs)}/{len(panel)}; missing {missing}")
    hdr = f"{'arm':17s} {'n':>3s} {'n_sel':>15s} {'deadends':>13s} {'mi_frac(old)':>15s} {'mi_int':>13s} {'mi_frac_int':>17s} {'missing/genuine':>15s} {'grow':>7s}"
    print(hdr); print("-" * len(hdr))
    for arm, s in out["arms"].items():
        if "n_rxn" not in s:
            print(f"{arm:17s} no data"); continue
        f = lambda k, d: f"{s[k]['mean']:.{d}f} ± {s[k]['sd']:.{d}f}"
        print(f"{arm:17s} {s['n']:3d} {f('n_selected',1):>15s} {f('deadends',2):>13s} {f('mi_frac',4):>15s} "
              f"{f('mi_int',2):>13s} {f('mi_frac_int',5):>17s} "
              f"{s['mi_int_missing_formula']['mean']:.2f}/{s['mi_int_genuine']['mean']:.2f}".rjust(0) +
              f" {s['n_growing']:>3d}/{s['n']}")
    print("written", OUT)


if __name__ == "__main__":
    main()
