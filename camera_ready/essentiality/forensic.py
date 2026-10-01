"""Round 10: forensic root-cause analysis of why threshold beats METEOR on
Salmonella/K.pneumoniae/P.putida. PURE post-hoc analysis of already-saved
sgd_*.json files, RBH maps, and the baseline predictor score matrices
(loaded read-only, no MILP re-solve). Writes only under
psb_revision/feasibility_essentiality/results/forensic/.
"""
import json, csv, os, sys
import numpy as np

HERE = "/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/feasibility_essentiality"
R = "/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision"
sys.path.insert(0, f"{R}/eval")
from _env import *  # noqa
from meteor_v8.utils import load_ec, load_refmapping, data_path, data_dir, extract_pred
from baseline_io import resolve_baseline_pkl, BASELINE_SUFFIX

OUT = f"{HERE}/results/forensic"; os.makedirs(OUT, exist_ok=True)
anc = load_ec(data_path("all_ancestors.txt"))

def load_map(path):
    m = {}
    for l in open(path):
        p = l.rstrip("\n").split("\t")
        if len(p) >= 3: m[p[0]] = p[2]
    return m

def load_ref(path, col="gene"):
    r = {}
    for row in csv.DictReader(open(path)):
        r[str(row[col]).strip()] = (row["ess.experimental"] == "yes")
    return r

def load_products(ft_path, prefix=None):
    """locus (old_locus_tag) -> product name, from an NCBI feature table."""
    d = {}
    import re
    for l in open(ft_path):
        p = l.rstrip("\n").split("\t")
        if len(p) < 17 or p[0] != "CDS": continue
        prod = p[13]
        # match to old_locus_tag via the paired gene row is complex; use CDS's own locus_tag col directly if it matches prefix
        loc = p[16]
        d[loc] = prod
    return d

ORGS = {
    "salmonella": dict(gca="GCF_000006945.2", baseline_variant=("dpz","vanilla"), suffix="DPZ",
        map_tsv=f"{HERE}/ref/salmonella_map/lt2_to_sl1344.tsv",
        ref_csv=f"{HERE}/ref/salmonella/salmonella_binary.csv", ref_col="gene",
        products=None),
    "kpneumoniae": dict(gca="GCF_058435815.1", baseline_variant=("dpz","vanilla"), suffix="DPZ",
        map_tsv=f"{HERE}/ref/kpneumoniae_map/kp0179_to_ecl8.tsv",
        ref_csv=f"{HERE}/ref/kpneumoniae/kpneumoniae_ecl8_binary.csv", ref_col="gene",
        products=None),
    "pputida": dict(gca="GCF_045571375.1", baseline_variant=("dpz","vanilla"), suffix="DPZ",
        map_tsv=f"{HERE}/ref/pputida_map/panel_to_pp.tsv",
        ref_csv=f"{HERE}/ref/pputida/pputida_binary.csv", ref_col="gene",
        products=None),
}

# product lookups per organism (locus -> description)
prod_pputida = {}
for row in csv.DictReader(open(f"{HERE}/ref/pputida/pputida_KT2440_essentiality_LB.csv")):
    prod_pputida[row["Gene ID"].strip()] = row["Gene description"].strip()

def prod_from_ft(ft_path):
    d = {}
    import re
    cur_rs2name = {}
    for l in open(ft_path):
        p = l.rstrip("\n").split("\t")
        if len(p) < 17: continue
        if p[0] == "CDS" and p[10]:
            cur_rs2name[p[16]] = p[13]  # RS_locus -> product name
    # now map old_locus_tag -> product via gene rows
    out = {}
    for l in open(ft_path):
        p = l.rstrip("\n").split("\t")
        if len(p) < 17 or p[0] != "gene": continue
        m = re.search(r"old_locus_tag=([A-Za-z0-9_,]+)", l)
        if not m: continue
        rs = p[16]; prod = cur_rs2name.get(rs, "")
        for v in m.group(1).split(","):
            out[v] = prod
    return out

prod_salm = prod_from_ft(f"{HERE}/ref/salmonella_map/sl1344_ft.txt")
prod_kpn = prod_from_ft(f"{HERE}/ref/kpneumoniae_map/ecl8_ft.txt")
ORGS["salmonella"]["products"] = prod_salm
ORGS["kpneumoniae"]["products"] = prod_kpn
ORGS["pputida"]["products"] = prod_pputida

def bin_score(s):
    if s <= 0: return "0"
    if s < 0.1: return "(0,0.1)"
    if s < 0.5: return "[0.1,0.5)"
    return "[0.5,1]"

PREDICTORS = ["clean", "dpz", "enzbert"]
all_results = {}
for org, cfg in ORGS.items():
    p2ref = load_map(cfg["map_tsv"])
    ref_ess = load_ref(cfg["ref_csv"], cfg["ref_col"])
    products = cfg["products"]
    org_summary = {}
    for pred_name in PREDICTORS:
        try:
            meteor = json.load(open(f"{HERE}/results/essround3/sgd_{org}_meteor_{pred_name}.json"))
            thresh = json.load(open(f"{HERE}/results/essround3/sgd_{org}_thresh_{pred_name}.json"))
        except FileNotFoundError:
            continue
        # load this predictor's score matrix once
        suffix = {"clean": "CLEAN_confidence", "dpz": "DPZ", "enzbert": "enzbert"}[pred_name]
        pkl_path = resolve_baseline_pkl(pred_name, "vanilla", cfg["gca"], suffix)
        pred_df = extract_pred(pkl_path, anc)
        max_score = pred_df.max(axis=1)  # per-protein max score across all ECs
        max_score_map = {str(idx).split()[0]: float(v) for idx, v in max_score.items()}

        common = set(meteor) & set(thresh)
        n_common = len(common)
        both_ref = []
        for g in common:
            loc = p2ref.get(g)
            if loc and loc in ref_ess:
                both_ref.append(g)
        n_both_ref = len(both_ref)
        meteor_wrong_thresh_right = []
        meteor_right_thresh_wrong = []
        both_right = both_wrong = 0
        for g in both_ref:
            loc = p2ref[g]; truth = ref_ess[loc]
            m_call = bool(meteor[g]); t_call = bool(thresh[g])
            m_correct = (m_call == truth); t_correct = (t_call == truth)
            if m_correct and t_correct: both_right += 1
            elif (not m_correct) and (not t_correct): both_wrong += 1
            elif t_correct and not m_correct: meteor_wrong_thresh_right.append((g, loc, truth, m_call, t_call))
            elif m_correct and not t_correct: meteor_right_thresh_wrong.append((g, loc, truth, m_call, t_call))

        # orphan check: reference-truth genes present in ONE arm's model but not the other
        meteor_only = [g for g in meteor if g not in thresh]
        thresh_only = [g for g in thresh if g not in meteor]
        meteor_only_ref = sum(1 for g in meteor_only if p2ref.get(g) in ref_ess)
        thresh_only_ref = sum(1 for g in thresh_only if p2ref.get(g) in ref_ess)

        summary = dict(n_meteor_genes=len(meteor), n_thresh_genes=len(thresh), n_common=n_common,
                        n_both_have_ref=n_both_ref, both_right=both_right, both_wrong=both_wrong,
                        n_meteor_wrong_thresh_right=len(meteor_wrong_thresh_right),
                        n_meteor_right_thresh_wrong=len(meteor_right_thresh_wrong),
                        meteor_only_genes=len(meteor_only), meteor_only_with_ref=meteor_only_ref,
                        thresh_only_genes=len(thresh_only), thresh_only_with_ref=thresh_only_ref)
        org_summary[pred_name] = summary
        print(f"{org}/{pred_name}: {summary}")

        if pred_name == cfg["baseline_variant"][0]:
            # deep dive: confidence-zone + pathway clustering for group (a)
            details = []
            for g, loc, truth, m_call, t_call in meteor_wrong_thresh_right:
                sc = max_score_map.get(g)
                prod = products.get(loc, "")
                details.append(dict(protein=g, locus=loc, ref_truth=truth, meteor_call=m_call, thresh_call=t_call,
                                     max_score=sc, score_bin=bin_score(sc) if sc is not None else None, product=prod))
            bincount = {}
            for d in details:
                b = d["score_bin"]; bincount[b] = bincount.get(b, 0) + 1
            print(f"  [{org}/{pred_name} DEEPDIVE] group(a) n={len(details)} score-bin distribution: {bincount}")
            json.dump(dict(group_a=details, score_bin_dist=bincount,
                            group_b=[dict(protein=g, locus=loc, ref_truth=truth, meteor_call=m_call, thresh_call=t_call)
                                     for g, loc, truth, m_call, t_call in meteor_right_thresh_wrong]),
                      open(f"{OUT}/{org}_deepdive_{pred_name}.json", "w"), indent=1)
    all_results[org] = org_summary

json.dump(all_results, open(f"{OUT}/summary_all.json", "w"), indent=1)
print("\n-> results/forensic/summary_all.json and per-org deepdive files")
