# Third-party inputs

These files are not ours. They are reproduced here so that the analysis
scripts run against a checkout, and each remains under the licence of its
source. Cite the sources below, not this repository, when using them.

## reac_xref.tsv

MetaNetX reaction cross-references, mapping BiGG and ModelSEED reaction
identifiers onto MNXR identifiers. `eval/toolcompare_*.py` and
`eval/curated_gradient.py` use it to compare reaction sets across the two
namespaces.

- Source: MetaNetX, https://www.metanetx.org/mnxdoc/mnxref.html
- Licence: CC BY 4.0
- Only rows whose first field begins `bigg.reaction:` or `seed.reaction:`
  are read; the rest of the file is unused.

## Unique_ModelSEED_Reaction_ECs.txt

ModelSEED reaction to EC number mapping, taken from Reconstructor. Used
wherever a ModelSEED reaction has to be resolved to EC numbers.

- Source: Reconstructor, https://github.com/emmamglass/reconstructor
- Cite: Jenior et al., Reconstructor: a COBRApy compatible tool for
  automated genome-scale metabolic network reconstruction with parsimonious
  flux-based gap-filling, Bioinformatics 39 (2023)

## curated_gems/

The six published genome-scale models used as the curated reference set:
iML1515 (*E. coli*), STM_v1_0 (*Salmonella*), iYL1228 (*K. pneumoniae*),
iJN1463 (*P. putida*), iYS854 (*S. aureus*), iYO844 (*B. subtilis*).

- Source: BiGG Models, http://bigg.ucsd.edu/
- Each model has its own publication; see BiGG for the citation attached to
  each identifier.
- We read only the `ec-code` annotations and the reaction identifiers.
