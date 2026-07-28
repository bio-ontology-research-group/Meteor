"""
Evaluation script for genome_collection (109 assemblies).
Computes top-1 and Fmax for 117 holdout proteins (4-digit EC + genome-resolved).

Usage:
    python eval_gc.py --out_tsv eval_gc_results.tsv

Inputs (auto-discovered from W):
  - benchmark/swissprot_holdout/holdout227.fasta  (UniProt IDs + EC in header)
  - scripts/genome_collection_manifest.json       (GCF -> UniProt proteins)
  - benchmark/proteomes/organism_proteomes_full/  (old UniProt FASTAs for hash map)
  - benchmark/proteomes/genome_collection/        (new GCF FASTAs, RefSeq IDs)
  - baseline_preds/{method}_genome_collection/    (flat pkls per GCF)
  - meteor_outputs/v6tf_{method}_{cutoff}_gc/     (METEOR output pkls)
"""
import os, json, hashlib, pickle, argparse
import pandas as pd
import numpy as np
from collections import defaultdict
from baseline_io import resolve_baseline_pkl, list_reinfer_overrides

W = "/ibex/scratch/projects/c2014/kexin/funcarve/paperA_2026"

METHODS = {
    "dpz-unifmu3": ("dpz_vanilla_genome_collection", "DPZ", "/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8_mu3e0_run/meteor_out/dpz_vanilla", ""),
    "dpz-evwmu3": ("dpz_vanilla_genome_collection", "DPZ", "/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8_evw_p2mu3_run/meteor_out/dpz_vanilla", ""),
    "dpz-evwmu8": ("dpz_vanilla_genome_collection", "DPZ", "/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8_evw_p2mu8_run/meteor_out/dpz_vanilla", ""),
}

def seq_hash(seq):
    return hashlib.md5(seq.upper().encode()).hexdigest()

def resolve_row(df, pid):
    """Exact match first; fall back to prefix match. Some CLEAN baseline
    pkls index by the full FASTA header (e.g. 'WP_000747555.1 MULTISPECIES:
    ATP-grasp domain-containing protein [Bacillus]') instead of the bare
    accession, causing false misses on an exact-match lookup."""
    if pid in df.index:
        return pid
    for i in df.index:
        if str(i).startswith(pid):
            return i
    return None

def parse_fasta(path):
    records = {}
    cur_id, cur_seq = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if cur_id: records[cur_id] = "".join(cur_seq).upper()
                header = line[1:]
                cur_id = header.split()[0]
                cur_seq = []
            else:
                cur_seq.append(line)
    if cur_id: records[cur_id] = "".join(cur_seq).upper()
    return records

def parse_holdout_ec(fasta_path):
    """Returns {uniprot_id: [ec1, ec2, ...]} from holdout fasta headers."""
    ec_map = {}
    import re
    with open(fasta_path) as f:
        for line in f:
            if not line.startswith(">"): continue
            uid = line[1:].split()[0]
            # EC numbers like 1.2.3.4
            ecs = re.findall(r'\d+\.\d+\.\d+\.\d+', line)
            ec_map[uid] = list(set(ecs))
    return ec_map

def fmax(gt_ecs, pred_scores):
    """Compute Fmax given ground truth EC list and {ec: score} dict."""
    if not gt_ecs: return 0.0
    gt = set(gt_ecs)
    all_scores = sorted(set(pred_scores.values()), reverse=True)
    best = 0.0
    for thresh in all_scores + [0.0]:
        pred = {ec for ec, s in pred_scores.items() if s >= thresh}
        if not pred: continue
        p = len(gt & pred) / len(pred)
        r = len(gt & pred) / len(gt)
        if p + r == 0: continue
        f = 2 * p * r / (p + r)
        best = max(best, f)
    return best

def top1_hit(gt_ecs, pred_scores):
    if not gt_ecs or not pred_scores: return 0
    top1 = max(pred_scores, key=pred_scores.get)
    return 1 if top1 in set(gt_ecs) else 0

def topk_hit(gt_ecs, pred_scores, k):
    if not gt_ecs or not pred_scores: return 0
    topk = sorted(pred_scores, key=pred_scores.get, reverse=True)[:k]
    return 1 if any(e in set(gt_ecs) for e in topk) else 0

# Step 1: load holdout TSV → EC ground truth + sequences
print("Loading holdout TSV (EC + sequences)...")
import re
TSV_PATH = f"{W}/benchmark/swissprot_holdout/swissprot_holdout_sample.tsv"
tsv_df = pd.read_csv(TSV_PATH, sep="\t", dtype=str).fillna("")
holdout_ec = {}       # uid -> [ec, ...]
uid_to_seq = {}       # uid -> sequence (uppercase)
for _, row in tsv_df.iterrows():
    uid = row["Entry"].strip()
    ec_str = row.get("EC number", "").strip()
    ecs = [e.strip() for e in ec_str.split(";") if re.match(r'^\d+\.\d+\.\d+\.\d+$', e.strip())]
    holdout_ec[uid] = ecs
    seq = row.get("Sequence", "").strip().upper().replace(" ", "")
    if seq:
        uid_to_seq[uid] = seq
print(f"  {len(holdout_ec)} holdout proteins loaded, {len(uid_to_seq)} with sequences")
holdout_4d = {uid: ecs for uid, ecs in holdout_ec.items() if ecs}
print(f"  {len(holdout_4d)} proteins with complete 4-digit EC")

# Step 2: build seq_hash → (gcf, RefSeq_ID) AND acc → (gcf, RefSeq_ID) from GCF FASTAs
print("Building seq_hash + accession → RefSeq_ID map from genome_collection...")
GC_DIR = f"{W}/benchmark/proteomes/genome_collection"
hash_to_refseq = {}
acc_to_refseq = {}   # WP_/NP_/YP_ accession -> (gcf, rid)
_ACC_RE = re.compile(r"(WP_\d+\.\d+|NP_\d+\.\d+|YP_\d+\.\d+)")
for fname in os.listdir(GC_DIR):
    if not fname.endswith(".fasta"): continue
    gcf = fname.replace(".fasta", "")
    for rid, seq in parse_fasta(os.path.join(GC_DIR, fname)).items():
        h = seq_hash(seq)
        hash_to_refseq[h] = (gcf, rid)
        m = _ACC_RE.search(rid)
        if m:
            acc_to_refseq[m.group(1)] = (gcf, rid)
print(f"  {len(hash_to_refseq)} RefSeq proteins indexed by seq_hash")
print(f"  {len(acc_to_refseq)} RefSeq proteins indexed by accession")

# Step 3: load manifest → UniProt → GCF
print("Loading manifest...")
manifest = json.load(open(f"{W}/scripts/genome_collection_manifest.json"))
uniprot_to_gcf = {}
proteins_section = manifest.get("assemblies", manifest.get("proteins", manifest))
for gcf, info in proteins_section.items():
    for uid in info.get("proteins", []):
        uniprot_to_gcf[uid] = gcf
print(f"  {len(uniprot_to_gcf)} UniProt→GCF mappings")

# Step 4: build UniProt → (GCF, RefSeq_ID) via seq_hash + accession fallback
print("Building UniProt → (GCF, RefSeq_ID)...")

# Load holdout_ncbi_map.json for accession-based fallback
_ncbi_map_path = f"{W}/benchmark/swissprot_holdout/holdout_ncbi_map.json"
_ACC_RE2 = re.compile(r"(WP_\d+\.\d+|NP_\d+\.\d+|YP_\d+\.\d+)")
uid_to_refseq_acc = {}  # uid -> WP_/NP_/YP_ accession from D2 FASTA
if os.path.exists(_ncbi_map_path):
    ncbi_map = json.load(open(_ncbi_map_path))
    for tax, prots in ncbi_map.items():
        for uid, ncbi_id in prots.items():
            m = _ACC_RE2.search(ncbi_id)
            if m:
                uid_to_refseq_acc[uid] = m.group(1)
    print(f"  Loaded {len(uid_to_refseq_acc)} accession mappings from holdout_ncbi_map.json")

uniprot_to_refseq = {}
n_by_hash = 0
n_by_acc = 0
for uid, seq in uid_to_seq.items():
    if uid not in uniprot_to_gcf:
        continue
    # Primary: seq_hash
    h = seq_hash(seq)
    if h in hash_to_refseq:
        uniprot_to_refseq[uid] = hash_to_refseq[h]
        n_by_hash += 1
        continue
    # Fallback: RefSeq accession from holdout_ncbi_map
    acc = uid_to_refseq_acc.get(uid)
    if acc and acc in acc_to_refseq:
        uniprot_to_refseq[uid] = acc_to_refseq[acc]
        n_by_acc += 1

print(f"  {len(uniprot_to_refseq)}/{len(uniprot_to_gcf)} UniProt IDs resolved "
      f"({n_by_hash} by seq_hash, {n_by_acc} by accession fallback)")

# 117-protein eval set: 4-digit EC ∩ genome-resolved
eval117 = {uid: ecs for uid, ecs in holdout_4d.items() if uid in uniprot_to_refseq}
print(f"  {len(eval117)} proteins in 117-eval set (4-digit EC + genome-resolved)")

# Also 204-protein set (genome-resolved, any EC)
eval204 = {uid: holdout_ec.get(uid, []) for uid in uniprot_to_gcf if uid in uniprot_to_refseq}
print(f"  {len(eval204)} proteins in 204-eval set (genome-resolved)")

# Step 6: evaluate each method — lazy loading per GCF to save memory
print("\nEvaluating...")

# Pre-build: GCF → list of (uid, refseq_id, gt_ecs) for eval117 proteins
from collections import defaultdict
gcf_to_proteins = defaultdict(list)
for uid, gt_ecs in eval117.items():
    gcf, refseq_id = uniprot_to_refseq[uid]
    gcf_to_proteins[gcf].append((uid, refseq_id, gt_ecs))

def safe_mean(lst):
    vals = [x for x in lst if x is not None]
    return round(np.mean(vals), 3) if vals else None
def safe_sum(lst):
    vals = [x for x in lst if x is not None]
    return sum(vals) if vals else None

rows = []
for method, (bdir, bsuffix, mdir, mname_pat) in METHODS.items():
    bpred_dir = f"{W}/baseline_preds/{bdir}"
    mpred_dir = mdir if mdir.startswith("/") else f"{W}/meteor_outputs/{mdir}"

    # Build GCF -> filename index without loading PKLs; prefer reinfer/ (fixed) over genome_collection (some broken)
    gcf_base_path = {}
    if os.path.exists(bpred_dir):
        for fname in os.listdir(bpred_dir):
            if fname.endswith(f"_{bsuffix}.pkl"):
                gcf = fname.replace(f"_{bsuffix}.pkl", "")
                gcf_base_path[gcf] = os.path.join(bpred_dir, fname)
    baseline_name, variant_name = "dpz", "vanilla"
    overrides = list_reinfer_overrides(baseline_name, variant_name)
    for gcf in overrides:
        gcf_base_path[gcf] = resolve_baseline_pkl(baseline_name, variant_name, gcf, bsuffix)
    if overrides:
        print(f"  [{method}] using {len(overrides)} reinfer-fixed pkls (overriding genome_collection)", flush=True)

    gcf_meteor_path = {}
    if os.path.exists(mpred_dir):
        for fname in os.listdir(mpred_dir):
            if fname.startswith("meteor_df_") and fname.endswith(".pkl"):
                inner = fname[len("meteor_df_"):-len(".pkl")]
                for gcf in gcf_base_path:
                    if inner.startswith(gcf):
                        gcf_meteor_path[gcf] = os.path.join(mpred_dir, fname)
                        break

    top1_b_list, top1_m_list = [], []
    top3_b_list, top3_m_list = [], []
    top5_b_list, top5_m_list = [], []
    top10_b_list, top10_m_list = [], []
    fm_b_list, fm_m_list = [], []
    n_eval = 0

    # Process one GCF at a time — load PKL, extract rows, free memory
    uid_order = list(eval117.keys())
    results_by_uid = {}
    for gcf, proteins in gcf_to_proteins.items():
        df_b, df_m = None, None
        if gcf in gcf_base_path:
            try:
                with open(gcf_base_path[gcf], "rb") as f:
                    df_b = pickle.load(f)
            except: pass
        if gcf in gcf_meteor_path:
            try:
                with open(gcf_meteor_path[gcf], "rb") as f:
                    df_m = pickle.load(f)
            except: pass

        for uid, refseq_id, gt_ecs in proteins:
            scores_b, scores_m = {}, {}
            if df_b is not None:
                b_key = resolve_row(df_b, refseq_id)
                if b_key is not None:
                    scores_b = {k.replace("EC:", ""): v for k, v in df_b.loc[b_key].to_dict().items()}
            if df_m is not None:
                m_key = resolve_row(df_m, refseq_id)
                if m_key is not None:
                    scores_m = df_m.loc[m_key].to_dict()
            results_by_uid[uid] = (gt_ecs, scores_b, scores_m)
            if scores_b and scores_m:
                n_eval += 1
        del df_b, df_m

    for uid in uid_order:
        if uid not in results_by_uid:
            top1_b_list.append(None); top1_m_list.append(None)
            top3_b_list.append(None); top3_m_list.append(None)
            top5_b_list.append(None); top5_m_list.append(None)
            top10_b_list.append(None); top10_m_list.append(None)
            fm_b_list.append(None); fm_m_list.append(None)
            continue
        gt_ecs, scores_b, scores_m = results_by_uid[uid]
        both = bool(scores_b) and bool(scores_m)  # fair: only proteins scored by BOTH
        top1_b_list.append(top1_hit(gt_ecs, scores_b) if both else None)
        top1_m_list.append(top1_hit(gt_ecs, scores_m) if both else None)
        top3_b_list.append(topk_hit(gt_ecs, scores_b, 3) if both else None)
        top3_m_list.append(topk_hit(gt_ecs, scores_m, 3) if both else None)
        top5_b_list.append(topk_hit(gt_ecs, scores_b, 5) if both else None)
        top5_m_list.append(topk_hit(gt_ecs, scores_m, 5) if both else None)
        top10_b_list.append(topk_hit(gt_ecs, scores_b, 10) if both else None)
        top10_m_list.append(topk_hit(gt_ecs, scores_m, 10) if both else None)
        fm_b_list.append(fmax(gt_ecs, scores_b) if both else None)
        fm_m_list.append(fmax(gt_ecs, scores_m) if both else None)

    rows.append({
        "method": method,
        "n_eval": n_eval,
        "top1_baseline": safe_sum(top1_b_list),
        "top1_meteor": safe_sum(top1_m_list),
        "top3_baseline": safe_sum(top3_b_list),
        "top3_meteor": safe_sum(top3_m_list),
        "top5_baseline": safe_sum(top5_b_list),
        "top5_meteor": safe_sum(top5_m_list),
        "top10_baseline": safe_sum(top10_b_list),
        "top10_meteor": safe_sum(top10_m_list),
        "fmax_baseline": safe_mean(fm_b_list),
        "fmax_meteor": safe_mean(fm_m_list),
        "base_pkls": len(gcf_base_path),
        "meteor_pkls": len(gcf_meteor_path),
    })
    print(f"  {method}: n={n_eval}, top1={safe_sum(top1_b_list)}/{safe_sum(top1_m_list)}, top3={safe_sum(top3_b_list)}/{safe_sum(top3_m_list)}, top5={safe_sum(top5_b_list)}/{safe_sum(top5_m_list)}, top10={safe_sum(top10_b_list)}/{safe_sum(top10_m_list)}, "
          f"Fmax={safe_mean(fm_b_list)}/{safe_mean(fm_m_list)}", flush=True)
    import pickle as _pk
    _pk.dump(results_by_uid, open(f"/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8_evw_p2mu3_run/downstream_results/holdout_diag_{method}.pkl","wb"))

df_out = pd.DataFrame(rows)
out_path = f"/ibex/scratch/projects/c2014/kexin/funcarve/meteor_diag/eval_evw.tsv"
df_out.to_csv(out_path, sep="\t", index=False)
print(f"\nSaved to {out_path}")
print(df_out.to_string())
