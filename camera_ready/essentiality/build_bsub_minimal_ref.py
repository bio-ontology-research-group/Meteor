"""Build the combined B. subtilis minimal-medium essentiality reference:
minimal_essential(gene) = LB_essential(gene) [Koo2017 Table S3, 257 genes]
                          OR auxotrophic(gene) [Koo2017 Table S4/D, 98 genes]
Both tables from Koo BM et al. 2017 Cell Systems 4:291-305 (manually
downloaded by the user, PMC/ScienceDirect PoW-gated for scripted access).
Join key: gene SYMBOL (matches gess-bsub.csv's gene2 column and our
model's SGD calls, which are keyed by symbol via knb1_to_symbol.tsv).
Multi-name entries like "gcaD (glmU)" are split so BOTH names are
matchable keys pointing to the same locus/essential call.
Writes ref/koo2017/bsub_minimal_binary.csv (same shape as pec_ecoli_binary.csv).
"""
import csv, re

HERE = "/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/feasibility_essentiality"

def split_names(gene_field):
    """'gcaD (glmU)' -> ['gcaD', 'glmU']; 'hisA' -> ['hisA']"""
    m = re.match(r'^\s*([A-Za-z0-9]+)\s*(?:\(\s*([A-Za-z0-9]+)\s*\))?\s*$', gene_field)
    if not m:
        return [gene_field.strip()]
    names = [m.group(1)]
    if m.group(2):
        names.append(m.group(2))
    return names

s3 = list(csv.DictReader(open(f"{HERE}/ref/koo2017/koo2017_table_s3_essential.csv")))
s4 = list(csv.DictReader(open(f"{HERE}/ref/koo2017/koo2017_table_s4_D_auxotrophs.csv")))
print(f"Table S3 (LB-essential): {len(s3)} rows")
print(f"Table S4/D (auxotrophs): {len(s4)} rows")

# gene -> (essential_bool, locus_tag, source)
combined = {}
n_s3_names = 0
for r in s3:
    locus = r["locus tag"].strip()
    for name in split_names(r["gene"]):
        n_s3_names += 1
        combined[name] = dict(ess=True, locus=locus, source="S3_LB_essential")

n_s4_new = 0; n_s4_already = 0
for r in s4:
    locus = r["Locus tag"].strip()
    for name in split_names(r["gene"]):
        if name in combined:
            n_s4_already += 1
            # already essential via S3 -- union stays True, keep record but note both sources
            combined[name]["source"] += "+S4_auxotroph"
        else:
            n_s4_new += 1
            combined[name] = dict(ess=True, locus=locus, source="S4_auxotroph_only")

print(f"S3 contributed {n_s3_names} name-keys (257 genes, {n_s3_names-257} extra from 15 multi-name splits)")
print(f"S4 auxotrophs: {n_s4_already} already essential via S3 (overlap), {n_s4_new} newly added (auxotroph-only)")
print(f"Combined minimal-essential set: {len(combined)} gene-symbol keys, all essential=True (this is a positive-only union list)")

# NOTE: this union gives ONLY the positive (essential) class directly. For a full binary
# reference we still need "non-essential" calls for the remaining genome -- those come from
# treating every OTHER gene symbol appearing in gess-bsub.csv (or the model's own gene set)
# as non-essential BY ABSENCE from this union, exactly mirroring how PEC/gess-bsub.csv work
# (a gene not in the essential set is implicitly non-essential). We do NOT fabricate an
# explicit non-essential row for every B. subtilis gene here (no whole-genome locus list was
# provided); downstream scoring (metrics()) already handles this correctly because it treats
# "not in ref_ess dict" as "no reference data" (excluded from denominator) UNLESS we mark it
# non-essential explicitly. To match the existing pipeline's ref-loading convention
# (load_ref() in rescoring_2x2.py reads a full gene,ess.experimental table with yes/no rows),
# we instead write ONLY yes-rows here and let essentiality_correct_medium.py's own gess-bsub.csv
# fallback logic decide -- see the loader note below.
# --- non-essential ("no") rows ---
# Koo S3/S4 only ever give the POSITIVE (essential) class; neither table lists the
# non-essential remainder explicitly (the raw whole-library fitness sheets, e.g.
# S4 tabs A/B, were not provided -- only the curated positive list, tab D). To get
# a full binary table shaped like pec_ecoli_binary.csv (whole-genome yes/no), we use
# gess-bsub.csv (SubtiWiki/Kobayashi, 844 genes with a real experimental call) as the
# whole-gene-coverage backbone, and apply the SAME union logic to it: a gene gets a
# "no" row here only if (a) gess-bsub.csv has an actual call for it (so we know it is
# a real, tested gene, not fabricated) AND (b) it is NOT in the Koo S3/S4 union. Genes
# gess-bsub.csv calls "yes" (rich-essential) are already covered by minimal_essential=True
# via the superset logic and are skipped here as duplicates of the S3-derived positives
# (not double counted). This is the boolean complement of the stated union rule:
#   minimal_non_essential(g) = NOT LB_essential(g) AND NOT auxotrophic(g)
# using gess-bsub.csv as the source of "NOT LB_essential" ground truth, since Koo's own
# tables do not enumerate the non-essential remainder.
gess_path = "/ibex/scratch/projects/c2014/kexin/funcarve/gapseq_eval/gapseqEval/GeneEssentiality/essentiality.data/gess-bsub.csv"
gess = list(csv.DictReader(open(gess_path)))
n_no_written = 0; n_skipped_already_yes = 0; n_skipped_now_essential_via_koo = 0
out = open(f"{HERE}/ref/koo2017/bsub_minimal_binary.csv", "w", newline="")
w = csv.writer(out)
w.writerow(["gene", "ess.experimental", "locus_tag", "source"])
for name, d in sorted(combined.items()):
    w.writerow([name, "yes", d["locus"], d["source"]])
for r in gess:
    name = r["gene2"].strip()
    if not name: continue
    if name in combined:
        n_skipped_now_essential_via_koo += 1
        continue  # already written as yes above (Koo overrides to essential)
    if r["ess.experimental"] == "yes":
        n_skipped_already_yes += 1
        continue  # gess-bsub says rich-essential but Koo S3 doesn't list it -- ambiguous, drop rather than guess
    w.writerow([name, "no", "", "gess-bsub.csv_non_essential_and_not_in_Koo_union"])
    n_no_written += 1
out.close()
print(f"-> wrote {len(combined)} essential (yes) rows + {n_no_written} non-essential (no) rows")
print(f"   (skipped {n_skipped_now_essential_via_koo} gess-bsub genes already covered by the Koo yes-union;")
print(f"    dropped {n_skipped_already_yes} gess-bsub yes genes not confirmed by Koo S3/S4 -- ambiguous, excluded rather than guessed)")
