import json, statistics as st

NEW = "/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/results/table1_v2/carveme_panel108.json"
OLD = "/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/results/table1_v2/carveme_panel108.PRE_BIOMASS_FIX.json.bak"

new = json.load(open(NEW))
old = json.load(open(OLD))

print("n records new:", len(new), "n records old:", len(old))
assert set(new.keys()) == set(old.keys()), "gca set changed between runs"

carveme_gcas = [g for g, r in new.items() if "carveme" in r and r.get("gem") is None or True]
# CarveMe present for all records; curated GEM present only for the 6 curated organisms
carveme_recs = {g: r["carveme"] for g, r in new.items() if "carveme" in r}
print("n CarveMe records:", len(carveme_recs))

# --- assertion 1: every CarveMe record's objective is exactly {"Growth"} ---
bad_obj = {g: r["objective_reaction_ids"] for g, r in carveme_recs.items() if r["objective_reaction_ids"] != ["Growth"]}
print("records where CarveMe objective_reaction_ids != ['Growth']:", len(bad_obj))
if bad_obj:
    print(bad_obj)

# --- assertion 2: for every genome, genuine count dropped by exactly 1, missing-formula unchanged, mi_internal count dropped by 1 ---
mismatches = []
for g in carveme_recs:
    n = new[g]["carveme"]
    o = old[g]["carveme"]
    d_genuine = n["mi_internal_genuine"] - o["mi_internal_genuine"]
    d_noformula = n["mi_internal_due_to_missing_formula"] - o["mi_internal_due_to_missing_formula"]
    d_internal_count = n["mass_imbal_internal"] - o["mass_imbal_internal"]
    d_n_internal = n["n_internal"] - o["n_internal"]  # n_internal should also drop by 1 (Growth now excluded)
    if not (d_genuine == -1 and d_noformula == 0 and d_internal_count == -1 and d_n_internal == -1):
        mismatches.append((g, d_genuine, d_noformula, d_internal_count, d_n_internal))
print("genomes NOT matching expected diff (genuine -1, noformula 0, mi_internal -1, n_internal -1):", len(mismatches))
if mismatches:
    for m in mismatches[:20]:
        print(" ", m)

# --- assertion 3: raw counts, not fractions, for the headline numbers ---
genuine_counts = [carveme_recs[g]["mi_internal_genuine"] for g in carveme_recs]
noformula_counts = [carveme_recs[g]["mi_internal_due_to_missing_formula"] for g in carveme_recs]
n_internal = [carveme_recs[g]["n_internal"] for g in carveme_recs]
mi_frac_internal = [carveme_recs[g]["mi_frac_internal"] for g in carveme_recs]

print()
print("=== CarveMe, 108 panel genomes, AFTER fix (from carveme_panel108.json, computed here, not hand-typed) ===")
print(f"  n = {len(genuine_counts)}")
print(f"  mi_internal_genuine: mean={st.mean(genuine_counts):.4f} sd={st.stdev(genuine_counts):.4f} min={min(genuine_counts)} max={max(genuine_counts)}")
print(f"  mi_internal_due_to_missing_formula: mean={st.mean(noformula_counts):.4f} sd={st.stdev(noformula_counts):.4f} (all should be 0)")
print(f"  n_internal: mean={st.mean(n_internal):.4f} sd={st.stdev(n_internal):.4f}")
print(f"  mi_frac_internal (genuine/n_internal, since noformula=0): mean={st.mean(mi_frac_internal):.4f} sd={st.stdev(mi_frac_internal):.4f}")

old_genuine_counts = [old[g]["carveme"]["mi_internal_genuine"] for g in carveme_recs]
print()
print("=== CarveMe, BEFORE fix (buggy, Growth counted), for comparison ===")
print(f"  mi_internal_genuine: mean={st.mean(old_genuine_counts):.4f} sd={st.stdev(old_genuine_counts):.4f}")

# --- curated GEMs (6 organisms), unaffected by objective-id fix in principle -- verify ---
curated_gcas = [g for g in new if "curated_bigg" in new[g]]
print()
print("=== curated BiGG GEMs (6), objective ids and genuine/missing-formula counts (from JSON) ===")
for g in curated_gcas:
    c = new[g]["curated_bigg"]
    print(f"  {new[g]['organism']:16s} gem={new[g]['gem']:10s} obj={c['objective_reaction_ids']} "
          f"n_internal={c['n_internal']} genuine={c['mi_internal_genuine']} missing_formula={c['mi_internal_due_to_missing_formula']} "
          f"mi_frac_internal={c['mi_frac_internal']}")
    oc = old[g]["curated_bigg"]
    same = (c["mi_internal_genuine"], c["mi_internal_due_to_missing_formula"]) == (oc["mi_internal_genuine"], oc["mi_internal_due_to_missing_formula"])
    print(f"     unchanged vs pre-fix: {same}")
