"""Gate: v2 record must reproduce published table1_{gca}.json (6 arms) and skeleton_abl JSONs (2 arms) exactly."""
import json, sys, os
R = "/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision"
OLD = "/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8/results/table1"
KEYS6 = ("n_selected", "n_rxn", "deadends", "mass_imbal", "mi_frac", "fba_growth")
KEYS2 = ("deadends", "mi_frac", "n_selected", "n_rxn", "mass_imbal", "fba_growth")
bad = 0
for gca in sys.argv[1:]:
    new = json.load(open(f"{R}/results/table1_v2/table1_{gca}.json"))
    old = json.load(open(f"{OLD}/table1_{gca}.json"))
    for arm in [f"{p}_{b}" for p in ("baseline", "meteor") for b in ("clean", "dpz", "enzbert")]:
        for k in KEYS6:
            if new[arm].get(k) != old[arm].get(k):
                bad += 1; print(f"DRIFT {gca} {arm} {k}: new={new[arm].get(k)} old={old[arm].get(k)}")
    for arm in ("full", "skelonly"):
        p = f"{R}/results/skeleton_abl/{gca}_{arm}.json"
        if not os.path.exists(p): print(f"SKIP {gca} abl_{arm}: no skeleton_abl json"); continue
        ref = json.load(open(p))
        for k in KEYS2:
            if new["abl_" + arm].get(k) != ref.get(k):
                bad += 1; print(f"DRIFT {gca} abl_{arm} {k}: new={new['abl_'+arm].get(k)} ref={ref.get(k)}")
    print(f"{gca}: checked; new fields meteor_dpz -> " + json.dumps({k: new["meteor_dpz"][k] for k in
          ("n_internal", "mi_int", "mi_int_missing_formula", "mi_int_genuine", "mi_frac_int")}))
print("GATE", "FAIL" if bad else "PASS", bad)
sys.exit(1 if bad else 0)
