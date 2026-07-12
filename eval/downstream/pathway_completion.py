#!/usr/bin/env python3
"""KEGG canonical-pathway completeness: baseline vs METEOR (v7), on a genome panel.

For each (genome, KEGG metabolism pathway) it compares the fraction of the
pathway's ECs covered by (a) the raw baseline predictor (max per-EC confidence
>= tau) and (b) METEOR's MILP-active EC set. Reports per-pair coverage + a
summary (mean coverage, delta, fraction of pathways >= 50%% complete).

Publication-ready: fully parameterized, no hard-coded per-run paths. Required
public data (documented in eval/downstream/README.md):
  --kegg_cache   KEGG pathway->EC map (fetched once from rest.kegg.jp; cached JSON)
  --manifest     genome list (JSON: {assemblies: {gcf: ...}} or {gcf: ...})
  --baseline_dir dir of raw baseline pkls  {gcf}_{suffix}.pkl  ([P x EC] DataFrame)
  --meteor_dir   dir of METEOR outputs     meteor_preds_{gcf}.pkl  (dict with active_ecs)
"""
import argparse, json, pickle, re, time, urllib.request
from pathlib import Path
import pandas as pd

SUFFIX = {"clean": "CLEAN_confidence", "dpz": "DPZ", "enzbert": "enzbert",
          "graphec": "GraphEC", "mapred": "MAPred", "topec": "TopEC"}


def kegg_get(url, retries=3, delay=0.4):
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return r.read().decode()
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(delay * (i + 1))


def fetch_kegg_pathway_ecs(cache_path):
    cache_path = Path(cache_path)
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    raw = kegg_get("https://rest.kegg.jp/list/pathway")
    pw_names, map_ids = {}, []
    for line in raw.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        pid = parts[0].replace("path:", "")
        if pid.startswith("map"):
            pw_names[pid] = parts[1]
            map_ids.append(pid)
    metab = [p for p in map_ids if re.match(r"map0[0-9]{4}$", p) and int(p[3:]) < 1100]
    pw2ec = {}
    for i, pid in enumerate(metab):
        try:
            raw = kegg_get(f"https://rest.kegg.jp/link/ec/{pid}")
            ecs = {parts[1].replace("ec:", "").strip()
                   for line in raw.strip().split("\n") if line
                   for parts in [line.split("\t")]
                   if len(parts) >= 2 and re.match(r"^\d+\.\d+\.\d+\.\d+$", parts[1].replace("ec:", "").strip())}
            if ecs:
                pw2ec[pid] = sorted(ecs)
        except Exception as e:
            print(f"  warn {pid}: {e}")
        time.sleep(0.3)
    result = {"pathway_ecs": pw2ec, "pathway_names": pw_names,
              "n_pathways": len(pw2ec), "kegg_release": time.strftime("%Y-%m-%d")}
    cache_path.write_text(json.dumps(result, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", choices=list(SUFFIX), required=True)
    ap.add_argument("--variant", default="vanilla")
    ap.add_argument("--baseline_dir", required=True, help="dir of {gcf}_{suffix}.pkl")
    ap.add_argument("--meteor_dir", required=True, help="dir of meteor_preds_{gcf}.pkl")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--kegg_cache", required=True)
    ap.add_argument("--out", required=True, help="output TSV (per gcf x pathway); .json summary alongside")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--min_ec", type=int, default=5,
                     help="drop pathways with fewer than this many four-digit ECs (paper default: 5)")
    a = ap.parse_args()

    suf = SUFFIX[a.baseline]
    kegg = fetch_kegg_pathway_ecs(a.kegg_cache)
    pathway_ecs = {k: set(v) for k, v in kegg["pathway_ecs"].items() if len(v) >= a.min_ec}
    pathway_names = kegg["pathway_names"]
    print(f"metabolism pathways (raw): {len(kegg['pathway_ecs'])}; "
          f"with >={a.min_ec} four-digit ECs: {len(pathway_ecs)}")

    man = json.loads(Path(a.manifest).read_text())
    gcfs = sorted(man.get("assemblies", man).keys())

    rows, n_ok = [], 0
    for gcf in gcfs:
        b_pkl = Path(a.baseline_dir) / f"{gcf}_{suf}.pkl"
        m_pkl = Path(a.meteor_dir) / f"meteor_preds_{gcf}.pkl"
        if not b_pkl.exists() or not m_pkl.exists():
            continue
        b_df = pickle.load(open(b_pkl, "rb"))
        b_df.columns = [str(c).replace("EC:", "") for c in b_df.columns]
        max_conf = b_df.max(axis=0)
        baseline_ecs = set(max_conf[max_conf >= a.tau].index.tolist())
        m_preds = pickle.load(open(m_pkl, "rb"))
        meteor_ecs = set(m_preds.get("active_ecs", set())) if isinstance(m_preds, dict) else set()
        for pw_id, pw_ec in pathway_ecs.items():
            nt = len(pw_ec)
            if nt == 0:
                continue
            b_hit, m_hit = len(baseline_ecs & pw_ec), len(meteor_ecs & pw_ec)
            rows.append(dict(gcf=gcf, pathway_id=pw_id.replace("map", ""),
                             pathway_name=pathway_names.get(pw_id, pw_id), n_pathway_ecs=nt,
                             baseline_hit=b_hit, meteor_hit=m_hit,
                             baseline_coverage=round(b_hit / nt, 6),
                             meteor_coverage=round(m_hit / nt, 6),
                             delta_coverage=round((m_hit - b_hit) / nt, 6)))
        n_ok += 1
    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, sep="\t", index=False)
    print(f"processed {n_ok} genomes, {len(df)} rows -> {a.out}")
    if len(df):
        summ = dict(baseline=f"{a.baseline}-{a.variant}", n_pathways=int(df.pathway_id.nunique()),
                    n_genomes=int(df.gcf.nunique()), n_pairs=len(df),
                    baseline_mean_coverage=round(df.baseline_coverage.mean(), 4),
                    meteor_mean_coverage=round(df.meteor_coverage.mean(), 4),
                    delta_mean=round(df.delta_coverage.mean(), 4),
                    baseline_frac_ge50=round((df.baseline_coverage >= 0.5).mean(), 4),
                    meteor_frac_ge50=round((df.meteor_coverage >= 0.5).mean(), 4),
                    kegg_release=kegg.get("kegg_release", ""))
        Path(a.out).with_suffix(".summary.json").write_text(json.dumps(summ, indent=2))
        print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
