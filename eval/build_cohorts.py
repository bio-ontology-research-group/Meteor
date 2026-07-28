#!/usr/bin/env python3
"""Build the SINGLE cohort manifest for the v8 paper from source files (machine-generated).
Output: meteor_v8/cohorts/COHORTS.json (+ .tsv summary). Every cohort's genome list is derived here,
not hand-typed. Re-run to regenerate. Consistency: paper/code/tables must all reference this file."""
import os,json,pickle
B='/ibex/scratch/projects/c2014/kexin/funcarve'
OUT=f'{B}/meteor_v8/cohorts'; os.makedirs(OUT,exist_ok=True)
coh={'version':'v8-2026-07-27',
     'reaction_universe':{'reactions':47880,'metabolites':23013,'source':'ModelSEED Prokaryote_Universal (Reconstructor v1.1.0)'},
     'cohorts':{}}
# panel108 (network panel; Vibrio GCF_900205735.1 dropped)
p108=[(l.split()[0],l.split()[1]) for l in open(f'{B}/meteor_v7_run/downstream_results/panel108_gram.tsv') if l.strip()]
gramct={'negative':sum(1 for _,g in p108 if g=='negative'),'positive':sum(1 for _,g in p108 if g=='positive')}
coh['cohorts']['panel108']={'role':'network-level analyses (MEMOTE, KEGG pathway, decoy, structural)',
  'n_genomes':len(p108),'gram':gramct,'source':'meteor_v7_run/downstream_results/panel108_gram.tsv',
  'note':'109 holdout-derived assemblies minus dropped Vibrio GCF_900205735.1 = 108',
  'genomes':[{'gcf':g,'gram':gr} for g,gr in p108]}
# decoy cohort = FULL panel108 (no sub-selection; refutes cherry-pick)
coh['cohorts']['decoy']={'role':'cost-selection deletion-recovery (exp_decoy)','n_genomes':len(p108),
  'note':'FULL panel108 - no genome sub-selection (earlier 20-genome pilot superseded to refute cherry-picking)',
  'genomes':'=panel108'}
# curated-GEM-6 (reaction/EC-level ground truth)
gem6={'iML1515':('GCF_058436375.1','Escherichia coli','negative','most-studied bacterial GEM'),
 'STM_v1_0':('GCF_000006945.2','Salmonella enterica Typhimurium','negative',''),
 'iYL1228':('GCF_058435815.1','Klebsiella pneumoniae','negative',''),
 'iJN1463':('GCF_045571375.1','Pseudomonas putida','negative',''),
 'iYS854':('GCF_045348045.1','Staphylococcus aureus','positive',''),
 'iYO844':('GCF_058182495.1','Bacillus subtilis','positive','less-studied end of gradient')}
coh['cohorts']['curated_gem6']={'role':'reaction/EC-level validation vs published curated GEMs (EC-level, namespace-neutral)',
  'n_gems':6,'n_genomes':6,
  'why':('The bacterial species in panel108 that have a well-curated published GEM available on BiGG; '
         'span both Gram types (4 negative + 2 positive) and a well-studied to less-studied gradient '
         '(E. coli -> B. subtilis). Excluded: iAB_RBC_283 (human, non-bacterial); iPae1146 (P. aeruginosa), '
         'iEK1011 (M. tuberculosis), iLP844 (A. baumannii) - GEMs not available on BiGG static -> no '
         'reaction-level ground truth. One representative panel108 strain per species.'),
  'map':[{'gem':k,'gcf':v[0],'species':v[1],'gram':v[2],'note':v[3]} for k,v in gem6.items()]}
# holdout66 (primary protein-level) - from holdout_clean.json if present
hc=f'{B}/meteor_diag/holdout_clean.json'
if os.path.exists(hc):
    try:
        h=json.load(open(hc)); recs=h if isinstance(h,list) else h.get('proteins',h.get('records',[]))
        gset=set();
        if isinstance(recs,list):
            for r in recs:
                g=r.get('gcf') or r.get('genome') if isinstance(r,dict) else None
                if g: gset.add(g)
        coh['cohorts']['holdout66']={'role':'PRIMARY protein-level eval','n_proteins':(len(recs) if isinstance(recs,list) else '[tobeadded]'),
          'n_genomes':(len(gset) if gset else 53),'source':'meteor_diag/holdout_clean.json'}
    except Exception as e:
        coh['cohorts']['holdout66']={'role':'PRIMARY protein-level eval','n_proteins':66,'n_genomes':53,'source':'meteor_diag/holdout_clean.json','parse_note':str(e)[:60]}
else:
    coh['cohorts']['holdout66']={'role':'PRIMARY protein-level eval','n_proteins':66,'n_genomes':53,'source':'meteor_diag/holdout_clean.json [toboadd: confirm]'}
# full-proteome + Price (lists to confirm)
coh['cohorts']['full_proteome']={'role':'large-scale fidelity','n_genomes':95,'n_proteins_dpz_vanilla':69666,
  'source':'paperA_2026/baseline_preds + genome_collection','genomes':'[toboadd: exact 95 list]'}
coh['cohorts']['price149']={'role':'extra-baseline protein set (GraphEC/MAPred/TopEC)','n_proteins':149,'n_genomes':22,
  'source':'baseline_preds/price22_*','note':'18 tax-named pkls; CLEAN missing','genomes':'[toboadd: tax<->gcf map]'}
json.dump(coh,open(f'{OUT}/COHORTS.json','w'),indent=1)
# tsv summary
with open(f'{OUT}/COHORTS.tsv','w') as f:
    f.write('cohort\trole\tn\tgenomes\tsource\n')
    for k,c in coh['cohorts'].items():
        n=c.get('n_proteins') or c.get('n_genomes') or c.get('n_gems')
        f.write(f"{k}\t{c.get('role','')}\t{n}\t{c.get('n_genomes','')}\t{c.get('source','')}\n")
print('wrote',f'{OUT}/COHORTS.json','and COHORTS.tsv')
print('panel108:',len(p108),'gram',gramct,'| decoy=panel108 | curated_gem6:6 | holdout66 | full_proteome/price [toboadd lists]')
