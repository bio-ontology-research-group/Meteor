from collections import defaultdict
import os
import re
import torch
import pickle
import numpy as np
import pandas as pd
from cobra.util import create_stoichiometric_matrix
from scipy.sparse import csr_matrix
import scipy
from pulp import LpStatus, value, PULP_CBC_CMD
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpBinary, PULP_CBC_CMD



def load_ec(file):
    print('load ec from:',file)
    with open(file,'r') as f:
        ecs = f.read().splitlines()
    return ecs

def load_refmapping(datap):
    with open(f'{datap}/seedr2ec.pkl', 'rb') as f:
        seedr2ec = pickle.load(f)
    with open(f'{datap}/seedec2r.pkl', 'rb') as f:
        seedec2r = pickle.load(f)
    print("seedr2ec dictionary loaded successfully.")
    
    return seedr2ec,seedec2r

def load_universal(with_transport=False):
    print('[INFO] load_universal',flush=True)
    
    if with_transport:
        uni_tranf='../data/universal_model_with_transport_rxns.pickle'
        print('[INFO] load universal model from Reconstructor[plus transport reactions]',flush=True)
    else:
        filename = '../data/universal.pickle'
        uni_tranf = filename
        print('[INFO] load universal model from Reconstructor',flush=True)
    print('load universal model from:',uni_tranf)
    if os.path.exists(uni_tranf):
        universal = pickle.load(open(uni_tranf, 'rb'))
        
    allrxns = [rxn.id for rxn in universal.reactions]
    allmet = [met.id for met in universal.metabolites]
    return universal, allrxns, allmet

def _invert_mapping(ec2r):
    rxn_to_ec = defaultdict(list)
    for ec, rxns in ec2r.items():
        for rxn in rxns:
            rxn_to_ec[rxn].append(ec)
    return rxn_to_ec

def maximum_separation(dist_lst, first_grad, use_max_grad):
    opt = 0 if first_grad else -1
    gamma = np.append(dist_lst[1:], np.repeat(dist_lst[-1], 10))
    sep_lst = np.abs(dist_lst - np.mean(gamma))
    sep_grad = np.abs(sep_lst[:-1]-sep_lst[1:])
    if use_max_grad:
        # max separation index determined by largest grad
        max_sep_i = np.argmax(sep_grad)
    else:
        # max separation index determined by first or the last grad
        large_grads = np.where(sep_grad > np.mean(sep_grad))
        if len(large_grads[0]) == 0:
            max_sep_i = 0
        else:
            max_sep_i = large_grads[-1][opt]
    # if no large grad is found, just call first EC
    if max_sep_i >= 5:
        max_sep_i = 0
    return max_sep_i

def maximum_separation_all(dist_lst, first_grad, use_max_grad):
    opt = 0 if first_grad else -1
    gamma = np.append(dist_lst[1:], np.repeat(dist_lst[-1], len(dist_lst)))
    sep_lst = np.abs(dist_lst - np.mean(gamma))
    sep_grad = np.abs(sep_lst[:-1]-sep_lst[1:])
    if use_max_grad:
        # max separation index determined by largest grad
        max_sep_i = np.argmax(sep_grad)
    else:
        # max separation index determined by first or the last grad
        large_grads = np.where(sep_grad > np.mean(sep_grad))
        if len(large_grads[0]) == 0:
            max_sep_i = 0
        else:
            max_sep_i = large_grads[-1][opt]
    # if no large grad is found, just call first EC
    if max_sep_i >= 5:
        max_sep_i = 0
    return max_sep_i

def get_media(mediainput,mediaf='../data/medium.pkl'):
    mediainfo = pd.read_pickle(mediaf)
    with open(mediaf,'rb') as f:
        mediainfo = pickle.load(f)
    media = mediainput[0]  
    if media in  mediainfo.keys():
        cmedia = mediainfo[media]
        media = [x +"_e" for x in cmedia]
    else:
        media = media
    return media
    
def extract_pred(pred_fpath, ancestorsec,remove_ratio=0):
        df = pd.read_pickle(pred_fpath)
        if df.index[0].count('.') == 3:  # if x.x.x.x format x can be any string
            df = df.T
        # reformat columns remove EC: 
        if 'EC:' in df.columns[0]:
            cols = df.columns.tolist()
            new_cols = [col.split('EC:')[-1] if 'EC:' in col else col for col in cols]
            df.columns = new_cols
        df = df.reindex(columns=ancestorsec, fill_value=0.0).astype(np.float16)
        return df

def build_rxn_ec_mask(rxn_ids, rxn_to_ec, allecs): 
    """
    构建 [n_rxns, n_ecs] 的稀疏掩码，1 表示该 reaction 关联该 EC。
    """
    n_rxns = len(rxn_ids)
    n_ecs = len(allecs)
    ec_to_index = {ec: i for i, ec in enumerate(allecs)}
    mask = np.zeros((n_rxns, n_ecs), dtype=np.float32)
    for i, rxn in enumerate(rxn_ids):
        rname = rxn[:-2]  # 去掉 "_c"
        ecs = rxn_to_ec[rname] if rname in rxn_to_ec else []
        # print(f'Processing reaction {rxn} with ECs: {ecs}', flush=True)
        if len(ecs) == 0:
            continue
        for ec in ecs:
            if ec in ec_to_index:
                j = ec_to_index[ec]
                mask[i, j] = 1.0
    return mask

def extract_fba_matrices(model, rxn_ids,reversed_trans=True, device='cpu'):
    if reversed_trans:
        reformate = 'RE'
    svpath = f'../data/fba_matrices_v5{reformate}.pkl'
    if os.path.exists(svpath):
        with open(svpath, 'rb') as f:
            data = pickle.load(f)
        S = data['S']
        lb = data['lb']
        ub = data['ub']
        if not isinstance(S, scipy.sparse.csr_matrix):
           S = scipy.sparse.csr_matrix(S)  
           with open(svpath, 'wb') as f:
               pickle.dump({'S': S, 'lb': lb, 'ub': ub}, f)
        print(f'Loaded FBA matrices from {svpath}.', flush=True)
        return S, lb, ub
    # Create stoichiometric matrix
    S_df = create_stoichiometric_matrix(model, array_type="DataFrame")
    lb = np.array([model.reactions.get_by_id(r).lower_bound for r in rxn_ids])
    ub = np.array([model.reactions.get_by_id(r).upper_bound for r in rxn_ids])

    S = S_df.to_numpy()
    S = scipy.sparse.csr_matrix(S)  # 若原本是 numpy array
    # reverse ex——reaction
    if reversed_trans:
        print('Reversing exchange reactions...', flush=True)
        for index, r in enumerate(rxn_ids):
            if r.startswith('EX_'):
                S[:, index] = -S[:, index]  # 取反即 reverse
    # SAVE
    with open(svpath, 'wb') as f:
        pickle.dump({'S': S, 'lb': lb, 'ub': ub}, f)
    print(f'Saved FBA matrices to {svpath}.', flush=True)
    return S, lb, ub

def metabolite_blocked(S, lb, ub, allrxns, universal_obj):
    S_sparse = csr_matrix(S)
    excludes = []
    if 'GmNeg' in universal_obj:
        excludes.append(allrxns.index('biomass_GmPos'))
    elif 'GmPos' in universal_obj:
        excludes.append(allrxns.index('biomass_GmNeg'))
    # find single reaction metabolites and their associated reactions
    met_single = []
    rxn_ofmetsingle=[]
    for i in range(S_sparse.shape[0]):
        count = len(S_sparse.getrow(i).indices)
        if count <= 2:
            if count ==1:
                met_single.append(i)
                rxn_ofmetsingle.append(S_sparse.getrow(i).indices[0])
            else:
                for item in S_sparse.getrow(i).indices:
                    if allrxns[item].startswith('SNK_'):
                        met_single.append(i)
                        rxn_ofmetsingle.append(S_sparse.getrow(i).indices[0])
                        
    # find singledirection metabolites and their associated reactions
    singledirection_met=[]
    rxn_ofsingledirmet=[]
    for i in range(S_sparse.shape[0]):
        row = S_sparse.getrow(i)
        if len(row.data) ==0:
            continue
        coutpos = [ x for x in row.data if x > 0]   
        if len(coutpos) == 0 or len(row.data) == len(coutpos):
            flagee = True
            # lb ub 都是正 或者负 
            for item in row.indices:
                if lb[item] < 0 and ub[item] > 0:
                    flagee= False
                    break 
            if flagee:
                singledirection_met.append(i)
                for item in row.data:
                    rxn_ofsingledirmet.append(item)
    excludes = set(excludes)
    for singlerxn in rxn_ofmetsingle:
        excludes.add(singlerxn)
    for singdrxn in rxn_ofsingledirmet:
        excludes.add(singdrxn)
    excludes = list(excludes)
    return excludes

def tasks_media_bound(tasks, media, allrxns, lb, ub, min_frac=0.01):
    r2id = {r: i for i, r in enumerate(allrxns)}
    n_rxns = len(allrxns)
    c_add = np.zeros(n_rxns, dtype=np.float16)
    c_remove = np.zeros(n_rxns, dtype=np.float16)
    media = get_media(media)
    rmedia = np.zeros(n_rxns, dtype=np.int8)
    # tasks
    for task in tasks:
        if task not in allrxns:
            continue
        idx = r2id[task]
        lb[idx] = max(lb[idx], min_frac)
        c_remove[idx] = 200  
        rmedia[idx] = 1
        
    # media + minimal uptake
    minitasks = ['EX_cpd00035_e','EX_cpd00051_e','EX_cpd00132_e','EX_cpd00041_e',
                 'EX_cpd00084_e','EX_cpd00053_e','EX_cpd00023_e','EX_cpd00033_e',
                 'EX_cpd00119_e','EX_cpd00322_e','EX_cpd00107_e','EX_cpd00039_e',
                 'EX_cpd00060_e','EX_cpd00066_e','EX_cpd00129_e','EX_cpd00054_e',
                 'EX_cpd00161_e','EX_cpd00065_e','EX_cpd00069_e','EX_cpd00156_e',
                 'EX_cpd00027_e','EX_cpd00149_e','EX_cpd00030_e','EX_cpd00254_e',
                 'EX_cpd00971_e','EX_cpd00063_e','EX_cpd10515_e','EX_cpd00205_e','EX_cpd00099_e']
    if len(media) != 0:
        media_cond = set(['EX_' + cpd for cpd in media])
        for i, r in enumerate(allrxns):
            if r.startswith('EX_'):
                if r in media_cond:
                    lb[i] = -100.0
                    ub[i] = 100.0
                    c_remove[i] = 100
                    rmedia[i] = 1
                elif r in minitasks:
                    lb[i] = -100.0
                    ub[i] = 100.0
                    c_remove[i] = 100
                    rmedia[i] = 1
                else:
                    lb[i] = 0.0
                    ub[i] = 1000.0
                    c_add[i] = 100
                    rmedia[i] = 0
    return lb, ub, c_add, c_remove, rmedia


def compute_uncertainty(pred_df, rxn_ec_mask, c_add, c_remove, theta):
    """
    Calculate reaction uncertaint: r0, c_add, c_remove。
    pred_df: DataFrame, shape [n_samples, n_ecs]
    rxn_ec_mask: np.array, shape [n_rxns, n_ecs]
    theta: float, threshold
    """
    pred_values = pred_df.values  # shape [n_samples, n_ecs]
    n_rxns = rxn_ec_mask.shape[0]
    r0 = np.zeros(n_rxns, dtype=np.int8)
    c_add = np.zeros(n_rxns, dtype=np.float16)
    c_remove = np.zeros(n_rxns, dtype=np.float16)
    noecmappingr=0
    for i in range(n_rxns):
        ec_indices = np.where(rxn_ec_mask[i] == 1)[0]
        if len(ec_indices) == 0:
            noecmappingr+=1
            continue
        if c_remove[i] > 0:  # in initial model skip
            r0[i] = 1
            continue
        ec_preds = pred_values[:, ec_indices]  # shape [n_samples, n_ecs_for_rxn]
        max_preds = np.max(ec_preds, axis=1)  # shape [n_samples]
        max_val = np.max(max_preds)
        if max_val <= theta:
            c_add[i] = theta - max_val
            r0[i] = 0
        else:
            above_theta = ec_preds[ec_preds > theta]
            if len(above_theta) == 0:   
                c_remove[i] = 0.0
            else:
                c_remove[i] = np.sum(above_theta - theta)
            r0[i] = 1
    print('Number of reactions with no EC mapping:', noecmappingr, flush=True)
    return r0, c_add, c_remove

def build_model(S, lb, ub, r0, c_add, c_remove, obj_idx, biomass_met_id, excludes):
    M,R = S.shape
    model = LpProblem("Metabolic_opt", LpMinimize)
    v = LpVariable.dicts("v", range(R), cat='Continuous')
    for j in range(R):
        v[j].lowBound = lb[j]
        v[j].upBound = ub[j]
    rfinal = LpVariable.dicts("r_final", range(R), lowBound=0, upBound=1, cat='Binary')
    
    S_sparse = csr_matrix(S)

    # Mass balance constraints
    for i in range(M):
        if i == biomass_met_id:
            continue  
        row = S_sparse.getrow(i)
        expr = lpSum(row.data[k] * v[row.indices[k]] for k in range(len(row.indices)) )
        model += expr == 0, f"mass_balance_{i}"
    for j in range(R):
        model += v[j] >= lb[j] * rfinal[j], f"v_lb_binary_{j}"
        model += v[j] <= ub[j] * rfinal[j], f"v_ub_binary_{j}"  
        if j in excludes:
            model += rfinal[j] == 0, f"exclude_rxn_{j}"
        if j == obj_idx:
            model += v[j] >= 1.0, f"biomass_production"
 
    total_cost = LpVariable("total_cost", lowBound=0)
    eps = 1e-3
    cost_expr = lpSum( (c_add[j]+eps) * rfinal[j] for j in range(R) if r0[j] == 0
                )+ lpSum((c_remove[j]+eps) * (1 - rfinal[j]) for j in range(R) if r0[j] == 1)
    model += cost_expr + 1e-6 * lpSum(v[j] for j in range(R))+ lpSum(1e-3 * rfinal[j] for j in range(R))
    
    model += total_cost == cost_expr, "total_cost_definition"
    model += total_cost
    for i in range(R):
        rfinal[i].setInitialValue(r0[i])  # 设置初始值为 r0[i]
        
    return model, v, rfinal

def rfinal2ec(rfinal_values, rxn_to_ec, ancestorsec, allrxns):
    active_ecs = set()
    mutated_ecs = set(ancestorsec)
    for i in range(len(rfinal_values)):
        if rfinal_values[i] > 0.5:
            ecs = rxn_to_ec.get(allrxns[i].split('_')[0], [])
            if len(ecs) > 0:
                active_ecs.update(ecs)
                mutated_ecs.difference_update(ecs)
    print(f'Active ECs: {len(active_ecs)}', flush=True)
    print(f'Mutated ECs: {len(mutated_ecs)}', flush=True)
    
    return active_ecs, mutated_ecs

def get_probs_df(df, fourecs, threshold):
    df = df[df.columns[df.columns.isin(fourecs)]]
    filtered_columns = df.columns
    pred_label = []
    pred_probs = []
    pred_scores = []
    for index, row in df.iterrows():
        score=[]
        currec=[]
        row = row.values
        for i in range(len(row)):
            if row[i] >= threshold:
                # currec.append(fourecs[i])
                currec.append(filtered_columns[i])
                score.append(row[i])
        if len(score) >= 1:
            np_score = np.array(score)
            raw_probs = 1 / (1 + np.exp(-np_score))
            # probs = raw_probs / np.sum(raw_probs)
            probs = raw_probs
            # currec_names = [ df.columns[x] for x in currec]
            paired_data = sorted(zip(currec, probs.tolist()), key=lambda item: item[0])
            currec = [name for name, p in paired_data]
            probs = [p for name, p in paired_data]
        else:
            maxval = np.max(row)
            if maxval >= 0:
                maxec = np.argmax(row)
                currec = [fourecs[maxec]]
                score = [row[maxec]]
                probs = [1.0]
            else:
                currec = []
                probs = []
                score = []
        
        pred_label.append(currec)
        pred_probs.append(probs)
        pred_scores.append(score)
        
    return pred_label, pred_probs, pred_scores
               
def update_topk(active_ecs, mutated_ecs, pred_df, ancestorsec,fourecs,theta,k=1):
    eps= 0.1
    with open('../data/enzymeobsolete.pkl', 'rb') as f:
        ecobselect = pickle.load(f)
    opt_df = pred_df.copy().astype(np.float64) 
    pred_preds, pred_probs, pred_scores = get_probs_df(pred_df, fourecs, threshold=theta)
    ori_active_ecs = set()
    for p in pred_preds:
        for ec in p:
            ori_active_ecs.add(ec)
    ori_mutated_ecs = set(ancestorsec) - ori_active_ecs
    opt_scores = [list(s) for s in pred_scores]
    opt_preds = [list(p) for p in pred_preds]
       
    # Part1: activate ecs
    for ec in active_ecs: 
        new_value = theta + eps
        if ec in ori_active_ecs:
            continue
        # else:
        s = re.sub(r'^EC[-:]', '', ec.strip())
        match = re.search(r'(\d+\.\d+\.\d+\.[\w]+)', s)
        if match:
            ec = match.group(1)
        if '-' in ec or ec not in ancestorsec:
            if ec.count('-')==1:
                ec = ec.replace('.-','')
            elif ec.count('-')==2:
                ec = ec.replace('.-.-','')
            elif ec.count('-')==3:
                ec = ec.replace('.-.-.-','')               
            if ec not in ecobselect.keys() and ec not in ancestorsec:
                continue
            else:
                if ec in ecobselect.keys():
                    ec = ecobselect[ec]
        maxval = opt_df[ec].max()
        if maxval != 0 :
            if ec in ori_active_ecs: # support the original val 
                maxp = opt_df[ec].idxmax()
                opt_df.loc[maxp, ec] = maxval + eps
                idx_array = np.where(opt_df.index == maxp)[0]
                if len(idx_array) > 0:
                    idx = idx_array[0]    
                opt_scores[idx]= [new_value]
            else:
                k_actual = min(k, len(opt_df))
                topk_idx = opt_df[ec].argsort()[-k_actual:][::-1]       
                for i in topk_idx:
                    protein_name = opt_df.index[i] 
                    opt_df.loc[protein_name, ec] = new_value
                    opt_scores[i].append(new_value) 
                    opt_preds[i].append(ec)
        else:
            digit3 = '.'.join(ec.split('.')[:2])
            thrdf = opt_df[opt_df.columns[opt_df.columns.str.startswith(digit3)]]
            thrdf_values = thrdf.values.flatten()
            k_actual = min(k, len(thrdf_values))
            if k_actual == 0:
                continue
            partitioned_indices = np.argpartition(thrdf_values, -k_actual)[-k_actual:]
            top_k_values = thrdf_values[partitioned_indices]
            sort_order = np.argsort(top_k_values)[::-1]
            partitioned_indices = partitioned_indices[sort_order]
            num_rows, num_cols = thrdf.shape
            row_indices = partitioned_indices // num_cols
            col_indices = partitioned_indices % num_cols
            protein_names = thrdf.index[row_indices]
            ec_names = thrdf.columns[col_indices]

            topkpec = list(zip(protein_names, ec_names))
            ps = set()
            for currp, currec in topkpec:
                ps.add(currp)
            for currp in ps:
                opt_df.loc[currp, ec] = new_value
                idx_array = np.where(opt_df.index == currp)[0]
                if len(idx_array) > 0:
                    idx = idx_array[0]
                opt_scores[idx].append(theta + eps)
                opt_preds[idx].append(ec)
                
    # Part2: mute ecs
    for ec in mutated_ecs:
        new_value = theta - eps
        if ec in ori_mutated_ecs:
            continue
        else:
            if ec in opt_df.columns:
                # clip to theta - 0.01
                idxs = opt_df.index[opt_df[ec] > new_value].tolist()
                for i in idxs:
                    opt_df.loc[i, ec] = new_value
                    idx_array = np.where(opt_df.index == i)[0]
                    if len(idx_array) > 0:
                        idx = idx_array[0]
                    if ec in opt_preds[idx]:
                        ecidx = opt_preds[idx].index(ec)
                        opt_preds[idx].remove(ec)
                        opt_scores[idx] = [s for j, s in enumerate(opt_scores[idx]) if j != ecidx]
    # compute probs
    opt_probs = []
    for i in range(len(opt_scores)):
        if opt_scores[i] == pred_scores[i]:
            currprobs = pred_probs[i]
        else:
            np_score = np.array(opt_scores[i])
            raw_probs = 1 / (1 + np.exp(-np_score))
            currprobs = raw_probs / np.sum(raw_probs)
            currprobs = currprobs.tolist()
        opt_probs.append(currprobs)
    return opt_df, opt_preds, opt_probs, opt_scores
          
def add_annotation(model, gram, obj='built'):
    ''' Add gene, metabolite, reaction ,biomass reaction annotations '''
    # Genes
    for gene in model.genes:
        gene._annotation = {}
        gene.annotation['sbo'] = 'SBO:0000243'
        gene.annotation['kegg.genes'] = gene.id
    
    # Metabolites
    for cpd in model.metabolites: 
        cpd._annotation = {}
        cpd.annotation['sbo'] = 'SBO:0000247'
        if 'cpd' in cpd.id: cpd.annotation['seed.compound'] = cpd.id.split('_')[0]

    # Reactions
    for rxn in model.reactions:
        rxn._annotation = {}
        if 'rxn' in rxn.id: rxn.annotation['seed.reaction'] = rxn.id.split('_')[0]
        compartments = set([x.compartment for x in list(rxn.metabolites)])
        if len(list(rxn.metabolites)) == 1:
            rxn.annotation['sbo'] = 'SBO:0000627' # exchange
        elif len(compartments) > 1:
            rxn.annotation['sbo'] = 'SBO:0000185' # transport
        else:
            rxn.annotation['sbo'] = 'SBO:0000176' # metabolic

    # Biomass reactions
    if obj == 'built':
        try:
            model.reactions.EX_biomass.annotation['sbo'] = 'SBO:0000632'
        except:
            pass
        if gram == 'none':
            biomass_ids = ['dna_rxn','rna_rxn','protein_rxn','teichoicacid_rxn','lipid_rxn','cofactor_rxn','rxn10088_c','biomass_rxn']
        else:
            biomass_ids = ['dna_rxn','rna_rxn','protein_rxn','teichoicacid_rxn','peptidoglycan_rxn','lipid_rxn','cofactor_rxn','GmPos_cellwall','rxn10088_c','GmNeg_cellwall','biomass_rxn_gp','biomass_rxn_gn']
        for x in biomass_ids:
            try:
                model.reactions.get_by_id(x).annotation['sbo'] = 'SBO:0000629'
            except:
                continue
    else:
        model.reactions.get_by_id(obj).annotation['sbo'] = 'SBO:0000629'

    return model

from pulp import *
import time
from tqdm import tqdm
from multiprocessing import cpu_count
def check_essentiality_fast(S, lb, ub, active_to_test, obj_idx, biomass_met_id):
    essential_results = {}
    M, R = S.shape
    S_sparse = csr_matrix(S)
    all_needed_indices = set(active_to_test) | {obj_idx}
    prob = LpProblem("Base_Essentiality", LpMinimize)
    v = LpVariable.dicts("v", all_needed_indices, cat='Continuous')

    for j in all_needed_indices:
        v[j].lowBound = lb[j]
        v[j].upBound = ub[j]
        
    for i in range(M):
        if i == biomass_met_id: continue
        row = S_sparse.getrow(i)

        valid_indices = [idx for idx in row.indices if idx in all_needed_indices]
        if valid_indices:
            prob += lpSum(S[i, k] * v[k] for k in valid_indices) == 0

    prob += v[obj_idx] >= 1.0
    
    for target_idx in tqdm(active_to_test, desc="Fast Checking Essentiality"):
        old_lb, old_ub = v[target_idx].lowBound, v[target_idx].upBound
        v[target_idx].lowBound, v[target_idx].upBound = 0, 0
        status = prob.solve(PULP_CBC_CMD(msg=0))
        essential_results[target_idx] = (LpStatus[status] == 'Infeasible')
        v[target_idx].lowBound, v[target_idx].upBound = old_lb, old_ub
        
    return essential_results

import cobra
from cobra.io import write_sbml_model
from cobra import Model, Reaction, Metabolite
import numpy as np
from scipy.sparse import issparse
def build_cobra_from_pulp_sol(args, allrxns, allmets, S, lb, ub, rfinal_values, v_values, obj_idx, universal):
    print(f"\n>>> Converting PuLP model to COBRA model: {args.name}", flush=True)
    cobra_config = cobra.Configuration()
    cobra_config.solver = "glpk"
    cobra_model = Model(args.name)

    met_objs = []
    for m_id in allmets:
        comp = m_id.split("_")[-1] if "_" in m_id else "c"
        m = Metabolite(id=m_id, compartment=comp)
        try:
            source_met = universal.metabolites.get_by_id(m_id)
            m.name = str(source_met.name)
            m.formula = source_met.formula
            m.charge = source_met.charge
        except KeyError:
            m.name = m_id
            
        met_objs.append(m)

    for j, rxn_id in enumerate(allrxns):
        if rfinal_values[j] > 0.5 or j == obj_idx:
            rxn = Reaction(id=rxn_id)
            try:
                source_rxn = universal.reactions.get_by_id(rxn_id)
                rxn.name = source_rxn.name
                rxn.annotation = dict(source_rxn.annotation) 
            except:
                rxn.name = rxn_id
            cobra_model.add_reactions([rxn])
            column = S[:, j]
            if issparse(column):
                column_coo = column.tocoo()
                met_indices = column_coo.row
                met_values = column_coo.data
            else:
                column_flat = np.asarray(column).flatten()
                met_indices = np.where(column_flat != 0)[0]
                met_values = column_flat[met_indices]
            
            met_dict = {met_objs[idx]: float(val) for idx, val in zip(met_indices, met_values)}
            rxn.add_metabolites(met_dict)
            
            rxn.lower_bound = float(lb[j])
            rxn.upper_bound = float(ub[j])
            rxn.annotation['pulp_flux'] = float(v_values[j])

    try:
        biomass_met = cobra_model.metabolites.get_by_id("cpd11416_c")
        sink_rxn = cobra_model.add_boundary(biomass_met, type="sink", lb=0, ub=1000)
        sink_rxn.id = "SK_cpd11416_c"
        original_obj_id = allrxns[obj_idx]
        cobra_model.objective = original_obj_id
        print(f"\nObjective: {cobra_model.objective.expression}", flush=True)
    except KeyError:
        cobra_model.objective = allrxns[obj_idx]
    cobra_sol = cobra_model.optimize()
    
    print("\n" + "-"*30)
    print(f"PuLP  obj val: {v_values[obj_idx]:.6f}", flush=True)
    print(f"COBRA obj val: {cobra_sol.objective_value:.6f}", flush=True)
    print(f"COBRA status : {cobra_sol.status}", flush=True)
    
    # 异常警报
    if cobra_sol.objective_value < 1e-6 and v_values[obj_idx] > 1e-6:
        print("\n❌ PuLP has solution but COBRA flux close to 0!", flush=True)
        print("Please check for details", flush=True)

    return cobra_model