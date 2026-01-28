import argparse
from copy import deepcopy
import pickle
import torch
import gc
from src.v5utils import *   
from multiprocessing import cpu_count

# User defined arguments
parser = argparse.ArgumentParser(description='Generate genome-scale metabolic network reconstruction using enzyme predictions.')
parser.add_argument('--input_file', type=str, default='none', help='Path to the input enzyme prediction file (.pkl)')
parser.add_argument('--svfolder', type=str, default='.', help='Directory to save output files')
parser.add_argument('--threshold', type=float, default=0.5, help='Probability threshold for enzyme presence')
parser.add_argument('--name', type=str, default='default', help='Name of the output model/experiment')
parser.add_argument('--timelimit', type=int, default=None, help='time limit for the model, in seconds. Default is None, which means no time limit.')
parser.add_argument('--media', type=str, default='default', help='List of metabolites composing the media condition. Not required.')
parser.add_argument('--tasks', type=list, default=[], help='List of metabolic tasks. Not required.')
parser.add_argument('--min_frac', type=float, default=0.01, help='Minimum objective fraction required during gapfilling')
parser.add_argument('--gap', type=float, default=0.05, help='Maximum gap in optimality allowed during optimization')
parser.add_argument('--gram', type=str, default='none', help='Type of Gram classificiation (positive or negative)')
parser.add_argument('--outmodel', type=str, default='default', help='Path/Name of output SBML model file')
parser.add_argument('--cpu', type=int, default=1, help='Number of processors to use')
parser.add_argument('--buildcobra', type=bool, default=False, help='Whether to build and save a COBRApy model')
parser.add_argument('--k',type=int, default=1, help='Top-k value for updating enzyme scores')
# parser.add_argument('--max_frac', type=float, default=0.5, help='Maximum objective fraction allowed during gapfilling')
# parser.add_argument('--seed',type=int, default=17)
# parser.add_argument('--remove_ratio',type=float, default=0)

args = parser.parse_args()

#----------------------------------------------------------------------------------------------------------------------#
if __name__ == "__main__":
    ## load data
    # Load reference mappings for EC numbers and reactions
    datap='data'
    seedr2ec,seedec2r=load_refmapping(datap)
    # check if seedr2ec contain val as None just simply remove them
    none_ec = [k for k, v in seedr2ec.items() if v is None]
    for k in none_ec:
        del seedr2ec[k]
    
    # Load universal model and ancestor information
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    universal, allrxns, allmet = load_universal()
    ancestorsec = load_ec(f'data/all_ancestors.txt')
    fourecs = load_ec(f'data/all_ec.txt')
    
    # Extract predictions and build reaction-EC mask
    pred_df = extract_pred(args.input_file, ancestorsec)
    rxn_ec_mask = build_rxn_ec_mask(allrxns, seedr2ec, ancestorsec)
    
    # Extract FBA matrices (Stoichiometric matrix, bounds)
    S, lb, ub = extract_fba_matrices(universal, allrxns,reversed_trans=True, device=device)

    ## prepare model parameters
    # Determine objective function based on Gram stain classification
    if str(args.gram) == 'positive':
        print('\nUsing Gram positive objective function')
        universal_obj = 'biomass_GmPos'
    elif str(args.gram) == 'negative':
        print('\nUsing Gram negative objective function')
        universal_obj = 'biomass_GmNeg'
    else:
        universal_obj = 'biomass'
        
    # Identify blocked metabolites and adjust bounds based on tasks and media
    obj_idx = allrxns.index(universal_obj)
    biomass_idx = allmet.index('cpd11416_c')
    excludes = metabolite_blocked(S, lb, ub, allrxns, universal_obj)
    lb, ub, c_add, c_remove, rmedia = tasks_media_bound(args.tasks, args.media,
                                                allrxns, lb, ub, args.min_frac)
    
    # Compute uncertainty parameters (r0, c_add, c_remove) for the optimization model
    r0, c_add, c_remove = compute_uncertainty(pred_df, rxn_ec_mask, c_add, c_remove, theta=args.threshold)
        
    for i, val in enumerate(r0):
        if rmedia[i] == 1:
            r0[i] = 1
    
    solpath = f'{args.svfolder}/v5_sol_{args.name}.pkl'
    print('Solution path:', solpath, flush=True)
    
    # Check if a solution already exists; if not, build and solve the MILP model
    if not os.path.exists(solpath):
        ## build model and solve
        model, v, rfinal = build_model(S, lb, ub, r0, c_add, c_remove, obj_idx, biomass_idx, excludes)
        print(f'\nModel has Built Successfully.', flush=True)
        # solver options
        solver_options = []
        if args.timelimit is not None:
            solver_options.append(f"sec {args.timelimit}")
            
        num_threads = min(8, cpu_count())
        solver_options = [f"threads {num_threads}",f"ratio {args.gap}",
                            "heuristics on","cuts on","presolve on" ]
                        
        cbc_options = " ".join(solver_options)
        # if you want to see the solver output, set msg=True
        # solver = PULP_CBC_CMD(msg=True, warmStart=True, timeLimit=args.timelimit, options=[cbc_options])
        # model.writeLP("v5coin_model.lp")
        solver = PULP_CBC_CMD(msg=False, warmStart=True, timeLimit=args.timelimit, options=[cbc_options])
        status = model.solve(solver)
        
        # check solution status
        if LpStatus[status] == 'Optimal':
            v_values = [v[j].value() for j in v]
            rfinal_values = [rfinal[j].value() for j in rfinal]
            objvalue = model.objective.value()
            total_reactions = sum(rfinal_values)
            print('>>>Model Optimal.', flush=True)
            print(f'Number of reactions in the final model: {int(total_reactions)}\nObjective value: {objvalue:.4f}', flush=True)
            # changednum = sum(rfinal_values != r0)
            # print(f'Number of changed reactions: {int(changednum)}')
            del model, v, rfinal
            gc.collect()
        
            # save solution
            solution = {'v_values': v_values, 'rfinal_values': rfinal_values, 'objvalue': objvalue}
            with open(solpath, 'wb') as f:
                pickle.dump(solution, f)
            print('Solution saved to:', solpath, flush=True)
            
        elif LpStatus[status] == 'Infeasible':
            print('>>>Model Infeasible.',flush=True)
            exit()
        else:
            print('>>>Solver Status:', LpStatus[status],flush=True)
            exit()
    else:
        # load solution
        with open(solpath, 'rb') as f:
            solution = pickle.load(f)
            v_values = solution['v_values']
            rfinal_values = solution['rfinal_values']
            objvalue = solution['objvalue']
        print('Solution loaded from:', solpath, flush=True)
   
    # Update predictions and scores based on the optimization results
    svdf = f'{args.svfolder}/v5_df_{args.name}.pkl'
    svpreds = f'{args.svfolder}/v5_preds_probs_scores_{args.name}.pkl'
    if not os.path.exists(svdf) or not os.path.exists(svpreds):
        active_ecs, mutated_ecs = rfinal2ec(rfinal_values, seedr2ec, ancestorsec, allrxns)
        opt_df, opt_preds, opt_probs, opt_scores = update_topk(active_ecs, mutated_ecs, pred_df, ancestorsec, fourecs, args.threshold, k=args.k)

        opt_df.to_pickle(svdf)
        print('Updated predictions saved to:', svdf, flush=True)
        
        svpreds = f'{args.svfolder}/v5_preds_probs_scores_{args.name}.pkl'
        with open(svpreds, 'wb') as f:
            pickle.dump((opt_preds, opt_probs, opt_scores), f)
        print('Updated preds, probs, scores saved to:', svpreds, flush=True)
    
    # Perform essentiality check if requested
    if args.check_essentiail:
        esslistfile = f'{args.svfolder}/v5_esslist_{args.name}.pkl'
        if not os.path.exists(esslistfile):
            active_indices = [j for j in range(len(allrxns)) if rfinal_values[j] > 0.5]
            active_to_test = [j for j in active_indices if j != obj_idx]
            results = check_essentiality_fast(S, lb, ub, active_to_test, obj_idx, biomass_idx)
            essential_count = sum(results.values())
            total_active = len(active_indices)
            essentiality_ratio = essential_count / total_active

            print("-" * 30)
            print(f'Essen Eval for:{args.name}')
            print(f"totalrxn:{total_active}\tessentialrxn:{essential_count}\tessentialratio:{essentiality_ratio:.4f}")
            # 打印前 10 个必需反应的索引
            essential_list = [idx for idx, is_ess in results.items() if is_ess]
            # save essential list
            
            esslistfile = f'{args.svfolder}/v5_esslist_{args.name}.pkl'
            with open(esslistfile, 'wb') as f:
                    pickle.dump(essential_list, f)
            print(f'essential_list:{esslistfile}')
        
        
        # print(f"前10个必需反应索引: {essential_list[:10]}")
        
    # build metabolic model 
    if args.buildcobra:
        import cobra
        from cobra.io import write_sbml_model
        # save model 
        if args.outmodel == 'default':
            sbml_path = f'{args.svfolder}/v5_model_{args.name}.xml'
        else:
            sbml_path = args.outmodel
            
        if not os.path.exists(sbml_path):
            cobra_model = build_cobra_from_pulp_sol(args, allrxns, allmet, S, lb, ub, rfinal_values, v_values, obj_idx, universal)
            for rxn in cobra_model.reactions:
                if not rxn.name:
                    rxn.name = rxn.id
            cobra_model = add_annotation(cobra_model, gram=args.gram, obj='built')
            
            print('cobra_model.objective:', cobra_model.objective, flush=True)
            write_sbml_model(cobra_model, sbml_path)
            print('COBRA model saved to:', sbml_path, flush=True)