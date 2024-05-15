from rich.console import Console
import numpy as np
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import shutil
import time
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

from score.sc_args import parse_args, init_dataset
from score import version
from score.compare_methods import compare_methods_wilcoxon
from score.utils.matrix_ops import get_flattened_matrices, viz_preprocessing
from score.utils.utils import resolution_to_name, res_name_to_int


console = Console()


def version_callback(print_version: bool) -> None:
    """Print the version of the package."""
    if print_version:
        console.print(f"[yellow]SCORE[/] version: [bold blue]{version}[/]")


def app():
    parser = argparse.ArgumentParser()
    if sys.argv[1] == 'embed':
        args, x, y, depths, batches, dataset, valid_dataset = parse_args(
            parser)
            
        from score.experiments.vade_experiment import VaDEExperiment
        from score.experiments.ensemble_experiment import EnsembleExperiment
        from score.run_experiment import pca_2d_exp, lsi_2d_exp, cisTopic_exp, scVI_exp, peakVI_exp, scGAD_exp, threeDVI_exp, toki_exp, ins_exp, higashi_exp, snap_atac_exp
        from score.experiments.lsi_1d_experiment import LSIExperiment
        from score.experiments.pca_experiment import PCAExperiment
        from score.experiments.schictools_experiments import ScHiCToolsExperiment
        from score.experiments.scHiCluster_experiment import ScHiClusterExperiment

        if args.baseline_sweep or args.embedding_algs is None:
            embedding_algs = ['InnerProduct'
                              'scHiCluster+vc_sqrt_norm_convolution,random_walk']
        elif isinstance(args.embedding_algs, list):
            embedding_algs = args.embedding_algs
        else:
            embedding_algs = [args.embedding_algs]
        val_y = None
        if valid_dataset is not None:
            val_x, val_y, _, _ = init_dataset(
                valid_dataset, False, False, None)
        if args.read_distribution is not None:
            mitotic_frac = dataset.reference.sort_index()['mitotic_frac']
            local_frac = dataset.reference.sort_index()['local_frac']
            features = {'batch': batches, 'depth': depths,
                        'mitotic': mitotic_frac, 'local': local_frac}
        else:
            features = {'batch': batches, 'depth': depths}
        if args.valid_clusters is not None:
            valid_mask = np.int32(
                dataset.reference['cluster'].isin(args.valid_clusters))
            features['valid'] = valid_mask
        for feat in dataset.reference.columns:
            if feat not in features and feat in ['age', 'sex', 'donor', 'region', 'subtype', 'cellclass']:
                features[feat] = dataset.reference.sort_index()[feat].values
        
        if args.preprocessing_sweep:  # sweep across all conventional preprocessing methods
            if any([alg.endswith('-') for alg in embedding_algs]):
                console.print(f"[red]Cannot use '-' suffix with --preprocessing-sweep[/]")
                sys.exit(1)
            elif any(['+' in alg for alg in embedding_algs]):
                console.print(f"[red]Cannot use '+' suffix with --preprocessing-sweep[/]")
                sys.exit(1)
            default_ops = ['vc_sqrt_norm', 'oe_norm', 'convolution', 'random_walk', 'quantile_0.95', 'google', 'network_enhance']
            new_algs = [alg + '-' for alg in embedding_algs] + [f'{alg}+{op}' for alg in embedding_algs for op in default_ops]
            default_combinations = [['vc_sqrt_norm', 'convolution', 'random_walk'], 
                                    ['vc_sqrt_norm', 'convolution', 'random_walk', 'quantile_0.85'],  # scHiCluster
                                    ['quantile_0.95', 'google'],
                                    ['vc_sqrt_norm', 'google'],
                                    ['vc_sqrt_norm', 'network_enhance'],
                                    ['vc_sqrt_norm', 'quantile_0.95', 'google']]
            new_algs += [f'{alg}+{",".join(op)}' for alg in embedding_algs for op in default_combinations]
            embedding_algs = new_algs
        elif args.distance_sweep:  # sweep across short/mid/long range distance considerations
            new_algs = []
            for alg in embedding_algs:
                for d in [1, '2Mb', '4Mb', '6Mb', '10Mb', '15Mb', '20Mb', '30Mb', '50Mb']:
                    new_algs.append(alg + '/' + str(d))
            embedding_algs = new_algs
        elif args.resolution_sweep:  # bin data to lower resolutions and test methods on these as well as provided resolution
            console.print(f"[green]Preparing resolution sweep starting from {dataset.res_name} resolution...[/]")
            new_algs = []
            for alg in embedding_algs:
                new_algs.append(alg + '@' + dataset.res_name)
            for factor in [2, 3, 4, 5, 10, 15, 20]:
                new_res = dataset.resolution * factor
                print(new_res)
                if new_res > 10e6:
                    break
                new_res_name = resolution_to_name(new_res)
                out_file = f"{dataset.dataset_name}_{new_res_name}_coarsened.scool"
                if out_file not in os.listdir("data/scools"):
                    console.print(f"[yellow]Binning data to {new_res_name} resolution...[/]")
                    dataset.write_binned_scool(f"data/scools/{out_file}", factor=factor, new_res_name=new_res_name)
                else:
                    console.print(f"[yellow]Already have {new_res_name} resolution...[/]")
                for alg in embedding_algs:
                    new_algs.append(alg + '@' + new_res_name)
            embedding_algs = new_algs

        skip_preprocessing = args.no_preprocessing
        load_results = args.load_results
        # if the last run used some preprocessing, we can use it for the next run
        previous_resolution = None
        previous_preprocessing = None
        previous_preprocessed_mats = None
        schictools_data = None

        if args.simulate and not args.append_simulated:
            console.print(f"[green]Embedding simulated data...[/]")
            console.print(
                f"[green]We need to downsample the bulk data, this only needs to be done once...[/]")
            os.makedirs('scools', exist_ok=True)
            new_name = f"{dataset.dataset_name}_{dataset.res_name}_{args.simulate_n}_{args.simulate_depth}"
            dataset.dataset_name = new_name
            for i in range(int(args.n_runs)):  # unique dataset for each experiment
                scool_file = f"{new_name}_rep{i}.scool"
                if scool_file in os.listdir('scools'):
                    print(scool_file, 'already exists...')
                    scool_file = 'scools/' + scool_file
                else:
                    scool_file = 'scools/' + scool_file
                    dataset.write_scool(
                        scool_file, simulate=args.simulate, n_proc=8)
            # update after we have generated first scool
            dataset.update_from_scool(scool_file)

        console.print("[bold green]Embedding data using:[/]")
        console.print(f"[purple4]{embedding_algs}[/]")

        for method in embedding_algs:
            start_time = time.time()
            method_name = method.lower()
            operations = None
            strata_options = []
            resolution_options = []
            if '/' in method_name and '@' in method_name:  # providing both distance and resolution in name
                strata_options = method_name.split('/')  # / specifies distance cutoff
                method_name = strata_options[0]
                resolution_options = strata_options[1].split('@')
            elif '/' in method_name:  # only providing distance in name
                strata_options = method_name.split('/')
                method_name = strata_options[0]
            elif '@' in method_name:  # only providing resolution in name
                resolution_options = method_name.split('@')
                method_name = resolution_options[0]

            if len(resolution_options) > 1:
                res_name = resolution_options[1].replace('m', 'M')
                if res_name != dataset.res_name:
                    resolution = res_name_to_int(res_name)
                    args.resolution = resolution
                    dataset.cell_list = [c.replace(dataset.res_name, res_name) for c in dataset.cell_list]
                    if args.resolution_sweep:
                        dataset.update_from_scool(f"data/scools/{dataset.dataset_name}_{res_name}_coarsened.scool", keep_depth=True)
                    else:
                        dataset.update_from_scool(f"data/scools/{dataset.dataset_name}_{res_name}.scool", keep_depth=True)
                    dataset.reference.index = dataset.reference.index.map(lambda s: s.replace(dataset.res_name, res_name)) 
                    dataset.res_name = res_name
                else:
                    resolution = dataset.resolution
            else:
                resolution = dataset.resolution
                

            if len(strata_options) > 1:
                strata_options = strata_options[1]
            else:
                strata_options = ''
            exp_name = method_name
            if operations is None and not skip_preprocessing:
                if method_name == 'schicluster':
                    # default, can be overidden with --no_preprocessing or by appending a '-' to the end of method name
                    operations = ['vc_sqrt_norm', 'convolution', 'random_walk']
                    exp_name = method_name

            # preprocessing settings
            if '+' in method_name:  # preprocessing setting included in method name
                method_name, operations = method_name.split('+')
                exp_name = method_name
                operations = operations.split(',')
                exp_name += ':' + ','.join(operations)
                skip_preprocessing = False
            elif method_name.endswith('-'):  # no preprocessing
                method_name = method_name[:-1]
                exp_name = method_name + ':none'
                skip_preprocessing = True
            elif args.schictools_preprocessing is not None and operations is None:
                operations = args.schictools_preprocessing
                exp_name = method_name
                exp_name += ':' + ','.join(args.schictools_preprocessing)
                skip_preprocessing = False
            elif operations is not None:  # default preprocessing for certain methods
                exp_name = method_name
                exp_name += ':' + ','.join(operations)
            if args.random_walk_iter != 1:
                exp_name += f',{args.random_walk_iter}_iter'
            if args.random_walk_ratio != 1.0:
                exp_name += f',{args.random_walk_ratio}'

            # distance settings
            if len(strata_options) > 0:
                # distance can either be in Mb, kb, or strata
                if strata_options.endswith('mb'):
                    args.n_strata = int(int(strata_options[:-2]) * 1e6 / dataset.resolution)
                elif strata_options.endswith('m'):
                    args.n_strata = int(int(strata_options[:-1]) * 1e6 / dataset.resolution)
                elif strata_options.endswith('kb'):
                    args.n_strata = int(int(strata_options[:-1]) * 1e3 / dataset.resolution)
                else:
                    args.n_strata = int(strata_options)
                exp_name += '<' + strata_options
            if args.strata_offset is not None:
                if args.strata_offset != 0:
                    exp_name += f'>{args.strata_offset}'

            wandb_config = None
            if args.wandb:
                wandb_config = {
                    **args.__dict__, 'method': method_name, 'dataset': dataset.dataset_name, 'preprocessing': operations}

            if method_name == 'schicluster':
                schicluster_exp = ScHiClusterExperiment(exp_name, y, features, dataset, resolution_name=dataset.res_name, n_experiments=int(args.n_runs), operations=operations,
                                                        simulate=args.simulate, load=load_results, schictools_data=schictools_data, name_suffix=args.subname,
                                                        n_strata=args.schictools_n_strata, strata_offset=int(args.schictools_strata_offset) if args.strata_offset is not None else 0,
                                                        val_data=(None, val_y), val_dataset=valid_dataset, other_args=args)
                schicluster_exp.run(
                    load=load_results, log_wandb=args.wandb, wandb_config=wandb_config)
                if args.no_cache:  # remove schictools tmp data
                    console.print("[yellow]Removing schictools tmp data...[/]")
                    try:
                        shutil.rmtree(f'schictools_data/{dataset.dataset_name}/{dataset.res_name}')
                    except FileNotFoundError:
                        pass


            elif method_name in ['fasthicrep', 'hicrep', 'innerproduct', 'selfish']:
                hicrep_exp = ScHiCToolsExperiment(exp_name, y, features, dataset,
                                                  method=method_name.replace(
                                                      'fast', ''),
                                                  schictools_data=schictools_data,
                                                  embedding_method='MDS', operations=operations,
                                                  resolution_name=dataset.res_name, n_strata=int(args.n_strata) if int(args.n_strata) > 0 else 20,
                                                  strata_offset=int(args.strata_offset) if args.strata_offset is not None else 0,
                                                  n_experiments=int(args.n_runs),
                                                  simulate=args.simulate, append_simulated=args.append_simulated,
                                                  load=load_results,
                                                  name_suffix=args.subname,
                                                  val_data=(None, val_y), val_dataset=valid_dataset, other_args=args)

                hicrep_exp.run(load=load_results,
                               log_wandb=args.wandb, wandb_config=wandb_config)
                if load_results:
                    hicrep_exp.loaded_data = None
                previous_preprocessing = operations
                previous_resolution = resolution
                if args.no_cache:  # remove schictools tmp data
                    console.print("[yellow]Removing schictools tmp data...[/]")
                    try:
                        shutil.rmtree(f'schictools_data/{dataset.dataset_name}/{dataset.res_name}')
                    except FileNotFoundError:
                        pass

            elif method_name == 'cistopic':
                cisTopic_exp(x, y, features, dataset, args, exp_name, operations, minc=args.cistopic_minc,
                             maxc=args.cistopic_maxc, load_results=load_results, wandb_config=wandb_config)

            elif method_name == 'scgad':
                random_walk = False
                if operations is not None:
                    if 'random_walk' in operations:
                        random_walk = True
                scGAD_exp(x, y, features, dataset, args.anchor_file, args, exp_name, operations,
                          depth_norm=not args.no_depth_norm, random_walk=random_walk, load_results=load_results, wandb_config=wandb_config)
                if args.no_cache:  # remove sparse matrix file tmp data
                    try:
                        sparse_dir = f'schic-topic-model/data/{dataset.dataset_name}_{dataset.res_name}'
                        shutil.rmtree(sparse_dir)
                    except FileNotFoundError:
                        pass
                    try:
                        interaction_dir = f'schic-topic-model/data/{dataset.dataset_name}_{dataset.res_name}_interactions'
                        shutil.rmtree(interaction_dir)
                    except FileNotFoundError:
                        pass

            elif method_name == '3dvi':
                threeDVI_exp(x, y, features, dataset, args.anchor_file, args, exp_name,
                             load_results=load_results, wandb_config=wandb_config)
                if args.no_cache: 
                    try:
                        interaction_dir = f'schic-topic-model/data/{dataset.dataset_name}_{dataset.res_name}_interactions'
                        shutil.rmtree(interaction_dir)
                    except FileNotFoundError:
                        pass
                    try:
                        results_dir = f"threeDVI/results/{dataset.dataset_name}_{dataset.res_name}"
                        shutil.rmtree(results_dir)
                    except FileNotFoundError:
                        pass

            elif method_name == 'toki' or method_name == 'detoki' or method_name == 'deDOC':
                toki_exp(x, y, features, dataset, args,
                         load_results=load_results, wandb_config=wandb_config)

            elif method_name == 'ins' or method_name == 'insulationscore' or method_name == 'insscore':
                ins_exp(x, y, features, dataset, args,
                        load_results=load_results, wandb_config=wandb_config)

            # elif method_name == 'loops':
            #     x = get_loops_data(dataset)
            #     val_x = None
            #     # exp = LSIExperiment(exp_name, x, y, features, dataset, preprocessing=operations,
            #     #                     n_experiments=args.n_runs, simulate=args.simulate, append_simulated=args.append_simulated, other_args=args)
            #     # exp.run(load=load_results, log_wandb=args.wandb,
            #     #         start_time=start_time, wandb_config=wandb_config)
            #     exp = PCAExperiment(exp_name, x, y, features, dataset, n_experiments=args.n_runs, simulate=args.simulate, append_simulated=args.append_simulated,
            #                         val_data=(val_x, val_y), val_dataset=valid_dataset, other_args=args)
            #     exp.run(load=load_results, log_wandb=args.wandb,
            #             start_time=start_time, wandb_config=wandb_config)
                
            elif method_name == 'loops_higashi':
                higashi_exp(x, y, features, dataset, args, fast_higashi=False,
                            load_results=load_results, wandb_config=wandb_config, loops_file='/mnt/rds/genetics01/JinLab/dmp131/sc-thrillpark/sc-thrillpark/examples/glue/data/important_loops.txt')

            elif method_name == 'loops_cistopic':
                cisTopic_exp(x, y, features, dataset, args, exp_name, operations, minc=args.cistopic_minc,
                             maxc=args.cistopic_maxc, load_results=load_results, wandb_config=wandb_config,
                             loops_file='/mnt/rds/genetics01/JinLab/dmp131/sc-thrillpark/sc-thrillpark/examples/glue/data/important_loops.txt')
                
            elif method_name == 'scab':
                x = dataset.compartment_vectors()
                exp = PCAExperiment(exp_name, x, y, features, dataset, n_experiments=int(args.n_runs), simulate=args.simulate, append_simulated=args.append_simulated,
                                    other_args=args)
                exp.run(load=load_results, log_wandb=args.wandb,
                        start_time=start_time, wandb_config=wandb_config)
                
            elif method_name == 'insulation':
                os.makedirs('data/dataset_summaries', exist_ok=True)
                x = dataset.insulation_vectors()
                exp = PCAExperiment(exp_name, x, y, features, dataset, n_experiments=int(args.n_runs), simulate=args.simulate, append_simulated=args.append_simulated,
                                    other_args=args)
                exp.run(load=load_results, log_wandb=args.wandb,
                        start_time=start_time, wandb_config=wandb_config)
                # remove insulation files if no cache
                if args.no_cache:
                    try:
                        os.remove(f'data/dataset_summaries/{dataset.dataset_name}_insulation_{dataset.res_name}.npy')
                    except FileNotFoundError:
                        pass
            
            elif method_name == '1d_pca':
                if load_results:
                    x = None 
                else:
                    x = get_flattened_matrices(dataset, int(args.n_strata), preprocessing=operations, agg_fn=np.sum)
                val_x = None
                if valid_dataset is not None:
                    val_x = get_flattened_matrices(valid_dataset, int(args.n_strata), preprocessing=operations, agg_fn=np.sum)
                exp = PCAExperiment(exp_name, x, y, features, dataset, n_experiments=int(args.n_runs), simulate=args.simulate, append_simulated=args.append_simulated,
                                    val_data=(val_x, val_y), val_dataset=valid_dataset, other_args=args)
                exp.run(load=load_results, log_wandb=args.wandb,
                        start_time=start_time, wandb_config=wandb_config)

            elif method_name == '1d_lsi':
                if load_results:
                    x = None
                else:
                    x = get_flattened_matrices(dataset, int(args.n_strata), preprocessing=operations, agg_fn=np.sum)
                exp = LSIExperiment(exp_name, x, y, features, dataset, preprocessing=operations,
                                    n_experiments=int(args.n_runs), simulate=args.simulate, append_simulated=args.append_simulated, other_args=args)
                exp.run(load=load_results, log_wandb=args.wandb,
                        start_time=start_time, wandb_config=wandb_config)

            elif method_name == '2d_pca':
                pca_2d_exp(x, y, features, dataset, args, exp_name, operations,
                           load_results=load_results, wandb_config=wandb_config)

            elif method_name == '2d_lsi':
                lsi_2d_exp(x, y, features, dataset, args, operations,
                           load_results=load_results, wandb_config=wandb_config)

            elif method_name == 'snapatac':
                snap_atac_exp(x, y, features, dataset, args, operations,
                           load_results=load_results, wandb_config=wandb_config)

            elif method_name == 'scvi':
                scVI_exp(x, y, features, dataset, args, operations, one_dim=True,
                         load_results=load_results, wandb_config=wandb_config)

            elif method_name == 'scvi_2d':
                scVI_exp(x, y, features, dataset, args, operations, one_dim=False,
                         load_results=load_results, wandb_config=wandb_config)

            elif method_name == 'peakvi':
                peakVI_exp(x, y, features, dataset, args, operations, one_dim=True,
                           load_results=load_results, wandb_config=wandb_config)

            elif method_name == 'peakvi_2d':
                peakVI_exp(x, y, features, dataset, args, one_dim=False,
                           load_results=load_results, wandb_config=wandb_config)

            elif method_name == 'vade':
                if args.load_vade_from is None:
                    from score.methods.vade import train_vade
                    experiment = VaDEExperiment('vade', x, y, features, dataset, encoder=None, eval_inner=False, other_args=args)
                    for run_i in range(int(args.n_runs)):
                        train_vade(features, dataset, experiment, run_i, args, preprocessing=operations,
                                load_results=load_results, wandb_config=wandb_config)
                else:
                    experiment = VaDEExperiment('vade', x, y, features, dataset, encoder=None, eval_inner=True, other_args=args)
                    experiment.run(load=False, outer_iter=0, start_time=start_time, log_wandb=args.wandb)
                if args.no_cache:
                    try:
                        shutil.rmtree(f'vade/{dataset.dataset_name}')
                    except FileNotFoundError:
                        pass
                    try:
                        shutil.rmtree(f'vade_models/{dataset.dataset_name}')
                    except FileNotFoundError:
                        pass

            elif method_name == 'ensemble':
                ensemble_exp = EnsembleExperiment(
                    'ensemble', x, y, features, dataset, other_args=args)
                ensemble_exp.run(load=False, start_time=start_time)

            elif method_name == 'higashi':
                higashi_exp(x, y, features, dataset, args, fast_higashi=False, operations=operations,
                            load_results=load_results, wandb_config=wandb_config)
                if args.no_cache:
                    try:
                        higashi_data_dir = f'data/higashi_data/{dataset.dataset_name}_{dataset.res_name}'
                        shutil.rmtree(higashi_data_dir)
                    except FileNotFoundError:
                        pass

            elif method_name == 'fast_higashi':
                higashi_exp(x, y, features, dataset, args,
                            load_results=load_results, wandb_config=wandb_config)
                
            elif method_name == 'scghost':
                higashi_exp(x, y, features, dataset, args, fast_higashi=False,
                            load_results=load_results, wandb_config=wandb_config, run_scghost=True)
                # if args.no_cache:
                #     try:
                #         higashi_data_dir = f'data/higashi_data/{dataset.dataset_name}_{dataset.res_name}'
                #         shutil.rmtree(higashi_data_dir)
                #     except FileNotFoundError:
                #         pass

        console.print("[bright_green]Done running embedding methods...[/]")
        if len(embedding_algs) > 1 and int(args.n_runs) > 1:
            console.print("[bright_green]Comparing performance...[/]")
            compare_methods_wilcoxon(dataset.dataset_name, dataset.res_name, main_metric='accuracy', emb_plots=not args.no_viz, results_dir=args.out)

    elif sys.argv[1] == 'heatmaps':
        args, x, y, depths, batches, dataset, valid_dataset = parse_args(
            parser)
        default_ops = ['vc_sqrt_norm', 'oe_norm', 'convolution', 'random_walk']
        default_combinations = [['vc_sqrt_norm'],
                                ['vc_sqrt_norm', 'random_walk'],
                                ['vc_sqrt_norm', 'convolution', 'random_walk']]
        for comb in default_combinations:
            default_ops.append(comb)
        viz_preprocessing(dataset, None)
        for operations in default_ops:
            if isinstance(operations, list):
                viz_preprocessing(dataset, operations)
            else:
                viz_preprocessing(dataset, [operations])

    elif sys.argv[1] == 'compare':
        # args, x, y, depths, batches, dataset, valid_dataset = parse_args(
        #     parser)
        args = parse_args(parser)
        if isinstance(args.embedding_algs, list):
            embedding_algs = args.embedding_algs
        else:
            embedding_algs = [args.embedding_algs]
        console.print(
            "[bright_green]Skipping embedding methods, going straight to performance comparison...[/]")
        results_dir = args.out
        for res_dir in os.listdir(os.path.join(results_dir, args.dset)):
            print(res_dir)
            if res_dir.endswith('kb') or res_dir.endswith('Mb') or res_dir.endswith('M'):
                compare_methods_wilcoxon(args.dset, res_dir, main_metric='accuracy', emb_plots=not args.no_viz, results_dir=args.out)

    elif sys.argv[1] == 'cooler':
        console.print("[bright_green]Creating .scool file...[/]")
        args, x, y, depths, batches, dataset, valid_dataset = parse_args(
            parser)
        if args.scool_downsample is None:
            downsample_frac = None
        else:
            downsample_frac = float(args.scool_downsample)
        if args.out.endswith('.scool'):
            dataset.write_scool(args.out, simulate=args.simulate, append_simulated=args.append_simulated, downsample_frac=downsample_frac)
        else:
            os.makedirs("data/scools/", exist_ok=True)
            dataset.write_scool(f"data/scools/{dataset.dataset_name}_{dataset.res_name}.scool",
                                simulate=args.simulate, append_simulated=args.append_simulated, downsample_frac=downsample_frac)

    elif sys.argv[1] == '1d_agg':
        args, x, y, depths, batches, dataset, valid_dataset = parse_args(
            parser)
        dataset.write_cell_bin_matrix(f"{dataset.dataset_name}_cell_matrix")

    elif sys.argv[1] == 'summary':
        args, x, y, depths, batches, dataset, valid_dataset = parse_args(
            parser)
        dataset.get_avg_cis_reads()
        dataset.distance_summary()
        if dataset.resolution <= 50000:
            dataset.check_mitotic_cells(from_frags=False)

    elif sys.argv[1] == 'merge':
        args, x, y, depths, batches, dataset, valid_dataset = parse_args(
            parser)
        dataset.write_pseudo_bulk_coolers()

    elif sys.argv[1] == 'bin':
        console.print("[bright_green]Coarsening .scool file to lower resolution...[/]")
        args, x, y, depths, batches, dataset, valid_dataset = parse_args(
            parser)
        factor = args.bin_factor
        new_resolution = dataset.resolution * factor
        if new_resolution > 1e6:
            new_res_name = f'{int(new_resolution/1e6)}M'
        else:
            new_res_name = f'{int(new_resolution/1e3)}kb'
        if args.out.endswith('.scool'):
            dataset.write_binned_scool(args.out, factor=factor, new_res_name=new_res_name)
        else:
            dataset.write_binned_scool(f'data/scools/{dataset.dataset_name}_{new_res_name}.scool', factor=factor, new_res_name=new_res_name)

    else:
        if 'help' not in sys.argv[1]:
            console.print("[bright_red]Unrecognized main argument...[/]")
        else:
            console.print("[bright_green]Welcome to SCORE! The current options are:[/]")
            console.print("[yellow]\tscore cooler --help\t| convert a raw dataset to .scool[/]")
            console.print("[yellow]\tscore bin --help\t| coarsen an existing .scool file to lower resolution[/]")
            console.print("[yellow]\tscore embed --help\t| run an embedding/clustering pipeline[/]")
            console.print("[yellow]\tscore compare --help\t| compare the results of various embeddings[/]")



if __name__ == "__main__":
    app()
