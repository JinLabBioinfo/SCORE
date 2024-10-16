import os
import sys
import json
import joblib
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from rich.console import Console
from score.data_helpers.schic_data_generator import DataGenerator


def anchor_list_to_dict(anchors):
    anchor_dict = {}
    for i, anchor in enumerate(anchors):
        anchor_dict[anchor] = i
    return anchor_dict


def init_dataset(train_generator, load_data, binarize, sparse_matrices_filename):
    x = []
    y = []
    depths = []
    batches = []
    if load_data:
        for cell_i in tqdm(range(train_generator.n_cells)):
            cell, label, depth, batch = train_generator.get_cell_by_index(cell_i, downsample=False)
            if cell is None:  # cell had no reads
                continue
            x.append(cell)
            y.append(label)
            depths.append(depth)
            batches.append(batch)
        if binarize:
            x = np.array(x, dtype=np.bool)
        else:
            x = np.array(x, dtype='float32')
        try:
            with open(os.path.join('data/sparse_matrices', sparse_matrices_filename), 'wb') as f:
                joblib.dump(train_generator.sparse_matrices, f)  # and save the sparse matrix dict for use later
        except MemoryError:
            print('Not enough memory to save')
    else:
        x, y, depths, batches = train_generator.get_cell_anchor_vectors(load=False)

    y_train = np.array(y)
    depths = np.array(depths)
    batches = np.array(batches)

    return x, y_train, depths, batches

def parse_args(parser, extra_args=None, verbose=True):
    import cooler
    import multiprocessing
    console = Console()
    
    parser.add_argument('--dset', type=str, help='dataset name, used to store results', default='pfc')
    parser.add_argument('--dataset', type=str, help='dataset name pointing to directory containing required files', default=None)
    parser.add_argument('--out', type=str, help='directory for writing output files and vizualizations', default='results')
    parser.add_argument('--subname', type=str, help=argparse.SUPPRESS, default=None)
    parser.add_argument('--exp', type=str, help=argparse.SUPPRESS, default='')
    parser.add_argument('--seed', default=36, type=int, help=argparse.SUPPRESS)
    parser.add_argument('--dataset_config', required=False, type=str, default=None, help='path to .json configuration file')
    parser.add_argument('--color_config', required=False, type=str, default=None)
    parser.add_argument('--scool', required=False, type=str, help='path to .scool scHi-C dataset file')
    parser.add_argument('--data_dir', required=False, type=str, help='path to interaction files if not using .scool')
    parser.add_argument('--anchor_file', required=False, type=str, help='path to anchor/bin reference file if not using .scool')
    parser.add_argument('--assembly', required=False, type=str, default='hg19', help='genome assembly used for mapping. important for methods like scGAD and Higashi which consider specific genomic loci')
    parser.add_argument('--reference', required=False, type=str, help='path to tsb delimited reference file containing at least cell name column and any other metadata like celltype, depth, etc')
    parser.add_argument('--subsample_n_cells', default=None, type=int, help=argparse.SUPPRESS)
    parser.add_argument('--bin_factor', default=2, type=int)
    parser.add_argument('--no_cache', action='store_true')
    parser.add_argument('--read_distribution', required=False, type=str)
    parser.add_argument('--min_depth', default=0, type=int)
    parser.add_argument('--max_depth', default=None, type=int)
    parser.add_argument('--n_threads', default=max(1, int(multiprocessing.cpu_count() / 4)), type=int)
    parser.add_argument('--n_runs', default=1, type=int, help='repeat each embeddding/clustering run to get a representative sample')
    parser.add_argument('--n_strata', default=16, type=int, help='force the number of strata used by embedding methods')
    parser.add_argument('--latent_dim', default=32, type=int, help='number of dimensions in final embedding')
    parser.add_argument('--strata_offset', default=None, type=int, help='ignore strata within this range')
    parser.add_argument('--n_cell_types', default=None, type=int, help=argparse.SUPPRESS)
    parser.add_argument('--random_walk_iter', default=1, type=int, help=argparse.SUPPRESS)
    parser.add_argument('--random_walk_ratio', default=1.0, type=float, help=argparse.SUPPRESS)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--vade_wandb', action='store_true')
    parser.add_argument('--continuous', action='store_true', help='data represents continuous rather than discrete cell states (e.g cell cycle)')
    parser.add_argument('--filter_mitotic', action='store_true')
    parser.add_argument('--use_xy', action='store_true')
    parser.add_argument('--mitotic_max', default=0.017, type=float, help=argparse.SUPPRESS)
    parser.add_argument('--mitotic_min', default=0.002, type=float, help=argparse.SUPPRESS)
    parser.add_argument('--ignore_filter', action='store_true')
    parser.add_argument('--ignore_chr_filter', action='store_true')
    parser.add_argument('--load_results', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--load_cells', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--no_viz', action='store_true')
    parser.add_argument('--viz_innerproduct', action='store_true')
    parser.add_argument('--resolution', default='', type=str)
    parser.add_argument('--file_suffix', default='', type=str, help=argparse.SUPPRESS)
    parser.add_argument('--gene_ref', type=str, default='data/hg19_genes_scGAD', help=argparse.SUPPRESS)
    parser.add_argument('--variable_loci', type=str, default='data/rna_atac.anchors', help=argparse.SUPPRESS)
    parser.add_argument('--only_clusters', nargs='+', help='names of celltypes/clusters which will be included, all others are dropped', default=None)
    parser.add_argument('--valid_clusters', nargs='+', help='names of celltypes/clusters which will be used as a validation set when computing clustering metrics', default=None)
    parser.add_argument('--eval_name', type=str, default='subcluster', help='name of evaluation set, used for saving results') 
    parser.add_argument('--eval_celltypes', nargs='+', help='a celltype/cluster subset which will be used to compute additional metrics', default=None)
    parser.add_argument('--embedding_algs', '--methods', nargs='+', help='names of algorithms to run on the given dataset', default=None)
    parser.add_argument('--schictools_preprocessing', nargs='+', help=argparse.SUPPRESS, default=None)
    parser.add_argument('--no_preprocessing', action='store_true', help='skip all preprocessing steps')
    parser.add_argument('--schictools_n_strata', type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--schictools_strata_offset', type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument('--cistopic_minc', type=int, default=8, help=argparse.SUPPRESS)
    parser.add_argument('--cistopic_maxc', type=int, default=32, help=argparse.SUPPRESS)
    parser.add_argument('--ins_dir', default='data/ins_data', help=argparse.SUPPRESS)
    parser.add_argument('--toki_dir', default='data/toki_data', help=argparse.SUPPRESS)
    parser.add_argument('--toki_delta_scale', type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument('--no_depth_norm', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--tad_count', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--indep_data', type=str, help=argparse.SUPPRESS)
    parser.add_argument('--frags', action='store_true', help=argparse.SUPPRESS)

    # sweeps
    parser.add_argument('--baseline_sweep', action='store_true')
    parser.add_argument('--distance_sweep', action='store_true')
    parser.add_argument('--resolution_sweep', action='store_true')
    parser.add_argument('--preprocessing_sweep', action='store_true')

    # simulated datasets args
    parser.add_argument('--downsample', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--scool_downsample', type=float, default=None, help='downsample each cell by this fraction before writing scool')
    parser.add_argument('--simulate', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--append_simulated', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--bulk_coolers_dir', type=str, help=argparse.SUPPRESS)
    parser.add_argument('--simulate_n', type=int, default=10, help=argparse.SUPPRESS)
    parser.add_argument('--simulate_depth', type=int, default=10000, help=argparse.SUPPRESS)

    #scGAD args
    parser.add_argument('--rna_file', type=str, help=argparse.SUPPRESS)
    parser.add_argument('--rna_anndata', type=str, help=argparse.SUPPRESS)  # for scanpy integration
    parser.add_argument('--coembed', action='store_true')
    parser.add_argument('--rna_ref', type=str, help=argparse.SUPPRESS)
    parser.add_argument('--atac_file', type=str, help=argparse.SUPPRESS)

    # 3dvi args
    parser.add_argument('--three3dvi_epochs', default=30, type=int, help=argparse.SUPPRESS)

    # Higashi args
    parser.add_argument('--higashi_epochs', default=30, type=int, help=argparse.SUPPRESS)
    parser.add_argument('--higashi_n_strata', type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--higashi_strata_offset', type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument('--higashi_dryrun', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--fast_higashi_tol', type=float, default=1e-3, help=argparse.SUPPRESS)

    # VaDE args
    parser.add_argument('--binarize', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--counts', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--load_vade_from', default=None, type=int, help=argparse.SUPPRESS)
    parser.add_argument('--beta', default=1, type=int, help=argparse.SUPPRESS)
    parser.add_argument('--n_clusters', default=None, type=int, help=argparse.SUPPRESS)
    parser.add_argument('--start_filters', default=4, type=int, help=argparse.SUPPRESS)
    parser.add_argument('--start_filter_size', default=5, type=int, help=argparse.SUPPRESS)
    parser.add_argument('--stride', default=2, type=int, help=argparse.SUPPRESS)
    parser.add_argument('--stride_y', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--gaussian_output', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--n_epochs', default=500, type=int, help=argparse.SUPPRESS)
    parser.add_argument('--batch_size', default=64, type=int, help=argparse.SUPPRESS)
    parser.add_argument('--viz_interval', default=2, type=int, help=argparse.SUPPRESS)
    parser.add_argument('--pretrain', default=False, type=bool, help=argparse.SUPPRESS)
    parser.add_argument('--lr', default=3e-4, type=float, help=argparse.SUPPRESS)
    parser.add_argument('--weight_decay', default=0.0001, type=float, help=argparse.SUPPRESS)
    if extra_args is None:
        args = parser.parse_args(sys.argv[2:])
    else:  # running tests from pytest
        args = parser.parse_args(sys.argv[6:])
        args.__dict__ = {**args.__dict__,  **extra_args}
    if args.wandb:
        os.environ["WANDB_SILENT"] = "true"

    try:
        if sys.argv[1] == 'compare':
            return args
        if sys.argv[1] == 'cooler' or sys.argv[1] == 'bin':  # do not filter if just writing or binning data
            args.ignore_filter = True
            args.ignore_chr_filter = True
    except Exception:
        pass
    if args.dataset is not None:
        # extract dataset name from path
        args.dset = args.dataset.split('/')[-1]
        if args.resolution != '':
            args.scool = f"{args.dataset}/{args.dset}_{args.resolution}.scool"
        else:  # find the first scool file in the directory
            scool_files = [f for f in os.listdir(args.dataset) if f.endswith('.scool')]
            if len(scool_files) == 0:
                console.print(f"[red]No .scool files found in directory[/] {args.dataset}")
                sys.exit(1)
            args.scool = f"{args.dataset}/{scool_files[0]}"
        ref_files = [f for f in os.listdir(args.dataset) if f.endswith('ref')]
        if len(ref_files) > 0:
            args.reference = f"{args.dataset}/{ref_files[0]}"
        else:
            console.print(f"[yellow]No reference file found in directory[/] {args.dataset}")
            sys.exit(1)

    if (args.scool is None and args.data_dir is None and args.anchor_file is None) and args.dataset_config is None:
        console.print(f"[red]Must provide either: [/]")
        console.print(f"[magenta ]1) .scool file[/]")
        console.print(f"[magenta ]2) path to data directory and anchor file[/]")
        console.print(f"[magenta ]3) path to bulk coolers and --simulate flag[/]")
    dataset_config = args.dataset_config

    if dataset_config is None:
        os.makedirs('data/dataset_configs', exist_ok=True)
        with open(os.path.join('data/dataset_configs', args.dset), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
    else:
        with open(os.path.join(dataset_config), 'r') as f:
            # merge args with config file, overwriting config values with user-provided values
            tmp_dict = args.__dict__.copy()  # all default args (and explicitly passed user args)
            aux_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)  # aux parser for only user-passed args
            for arg in vars(args):
                aux_parser.add_argument('--'+arg, action='store_true' if isinstance(getattr(args, arg), bool) else None)
            cli_args, _ = aux_parser.parse_known_args()
            tmp_dict.update(json.load(f))  # first overwrite defaults with config vals
            tmp_dict.update(cli_args.__dict__)  # then overwrite user-passed args
            args.__dict__ = tmp_dict
    if verbose:
        console.print(f"[green]Dataset:[/] {args.dset}")
    dataset_name = args.dset
    color_config = args.color_config
    
    if args.scool is None:
        data_dir = args.data_dir
        anchor_file = args.anchor_file
        scool_file = None
        content_of_scool = None
    else:
        scool_file = args.scool
    assembly = args.assembly
    file_reads = args.reference
    read_distribution = args.read_distribution
    min_depth = int(args.min_depth)
    if args.max_depth is not None:
        max_depth = int(args.max_depth)
    else:
        max_depth = None
    n_strata = int(args.n_strata)
    if args.schictools_n_strata is not None:
        n_strata = int(args.schictools_n_strata)
    if args.higashi_n_strata is not None:
        n_strata = int(args.higashi_n_strata)
    strata_offset = args.strata_offset
    if strata_offset is None:
        strata_offset = 0
    strata_offset = int(strata_offset)
    file_suffix = args.file_suffix
    if args.resolution != '' and not args.frags:
        file_suffix += '.' + args.resolution
    filter_by_depth = not args.ignore_filter
    filter_mitotic = args.filter_mitotic
    downsample = args.downsample
    binarize = args.binarize
    if args.embedding_algs is not None:
        for alg in args.embedding_algs:
            if alg.lower() == 'vade' or alg.lower() == 'va3de':
                if not args.binarize and not args.counts:
                    console.print(f"[bright_red]If using Va3DE you must provide either --counts or --binarize[/]")
                    console.print(f"[yellow]--counts will model the counts of each interaction and will work in most cases[/]")
                    console.print(f"[yellow]--binarize clips all values to 0/1 and will converge faster and is more well-suited for modeling sparse data (low depth/high resolution) and long-range structure[/]")
                    sys.exit(1)
    load_data = args.load_cells

    if args.resolution != '' and not args.frags:
        res_name = file_suffix[file_suffix.rfind('.') + 1:]
        resolution = 0
        if 'M' in res_name:
            resolution = int(args.resolution[:-1]) * 1000000
        elif 'mb' in res_name.lower():
            resolution = int(args.resolution[:-2]) * 1000000
        elif 'kb' in res_name:
            resolution = int(args.resolution[:-2]) * 1000
    elif args.frags:
        resolution = 10000
        res_name = '10kb'
    else:
        console.print(f"[yellow]Resolution not provided, inferring from bins, make sure this is right...[/]")
        content_of_scool = cooler.fileops.list_coolers(scool_file)
        c = cooler.Cooler(f"{scool_file}::{content_of_scool[0]}")
        anchor_list = c.bins()[:]
        anchor_list = anchor_list[['chrom', 'start', 'end']]
        anchor_list['len'] = (anchor_list['end'] - anchor_list['start']).abs()
        median_len = anchor_list['len'].median()  # some datasets use variable bin size, we check the median to get a good estimate
        if median_len > 2e6:  # 2Mb is the lowest resolution we consider
            res_name = '2M'
            resolution = 2000000
        elif median_len >= 1e6:  # 1Mb is typical low resolution
            res_name = '1M'
            resolution = 1000000
        elif median_len >= 500e3:
            res_name = '500kb'
            resolution = 500000
        elif median_len >= 200e3:
            res_name = '200kb'
            resolution = 200000
        elif median_len >= 50e3:
            res_name = '50kb'
            resolution = 50000
        else:  # if higher resolution than 200kb, we round to the nearest 1000
            res_name = f"{int(median_len / 1000)}kb"
            resolution = int(median_len / 1000) * 1000
        console.print(f"[magenta]Inferred {res_name} resolution...[/]")


    if args.simulate:  # we need to create our own reference
        reference = {'cell': [], 'depth': [], 'batch': [], 'cluster': []}
        coolers = os.listdir(args.bulk_coolers_dir)
        for cool_file in coolers:
            cluster_name = cool_file.replace('.mcool', '').replace('.cool', '')
            for i in range(args.simulate_n):
                reference['cell'].append(f"{cool_file}_{i}.{res_name}")
                reference['depth'].append(args.simulate_depth)
                reference['batch'].append(0)
                reference['cluster'].append(cluster_name)
        reference = pd.DataFrame.from_dict(reference)
        reference.sort_values(by='cell', inplace=True)
        reference.to_csv(f'data/data_info_{args.dset}', sep='\t', index=False)
        full_reference = reference.copy(deep=True)  # keep a copy of the full reference for filtering later
    if (args.append_simulated and args.simulate) or (not args.simulate):  # append reference if we can
        if args.append_simulated:
            tmp_ref = reference.copy(deep=True)
            tmp_ref['type'] = 'simulated'
            tmp_ref['cluster'] = tmp_ref['cluster'].apply(lambda s: s + ' simulated')
        reference = pd.read_csv(file_reads, sep='\t')
        if 'cell' not in reference.columns:
            console.print(f"[bright_red]Missing required cell IDs/names in reference file[/] {file_reads}")
            sys.exit(1)
        reference.sort_values(by='cell', inplace=True)
        if 'celltype' in reference.columns:
            reference.rename(columns={'celltype': 'cluster'}, inplace=True)
        if 'cluster' not in reference.columns:
            reference['cluster'] = 'n/a'
        if 'batch' not in reference.columns:
            reference['batch'] = 0
        if 'depth' not in reference.columns:
            reference['depth'] = 1
        reference['cluster'] = reference['cluster'].astype(str)
        reference.drop_duplicates(subset=['cell'], inplace=True, keep='last')
        reference['depth'] = reference['depth'].astype(int)
        if file_suffix != '':
            reference['cell'] = reference['cell'].apply(lambda r: r.replace('.remap', file_suffix))  # add file ending to all cell names if using different resolution
            if not reference.iloc[0]['cell'].endswith(file_suffix):
                reference['cell'] = reference['cell'] + file_suffix
        elif args.frags:
            reference['cell'] = reference['cell'].apply(lambda r: r + file_suffix)
        else:
            if content_of_scool[0].endswith(res_name):  # if the cooler was saved with resolution file endings
                reference['cell'] = reference['cell'].apply(lambda r: r + f".{res_name}") 
        full_reference = reference.copy(deep=True)  # keep a copy of the full reference for filtering later
        full_reference['filtered_reason'] = 'not filtered'
        if min_depth > 0 and filter_by_depth:
            reference = reference[reference['depth'] >= min_depth]
            full_reference.loc[full_reference['depth'] < min_depth, 'filtered_reason'] = 'reference depth < min_depth'
        if max_depth is not None and filter_by_depth:
            reference = reference[reference['depth'] <= max_depth].reset_index()
            full_reference.loc[full_reference['depth'] <= max_depth, 'filtered_reason'] = 'reference depth > max_depth'
        reference['cluster'] = reference['cluster'].apply(lambda s: s.replace('.remap', ''))  # remove any unecessary suffixes
        if reference['batch'].dtype != int:
            batches = reference['batch'].unique()
            batch_dict = {b: i for i, b in enumerate(batches)}
            reference['batch'] = reference['batch'].map(batch_dict)
        reference['batch'] = reference['batch'] - reference['batch'].min()  # force zero indexed batches for batch removal training
        reference['type'] = 'real'
        if args.append_simulated:
            reference = pd.concat([reference, tmp_ref])  # add simulated reference to real reference
   

    valid_reference = None
    if args.valid_clusters is not None:
        mask = reference['cluster'].isin(args.valid_clusters)
        valid_reference = reference[mask].reset_index()
        reference = reference[~mask].reset_index()

    if args.only_clusters is not None:
        if isinstance(args.only_clusters, list):
            reference = reference[reference['cluster'].isin(args.only_clusters)]
            full_reference.loc[~full_reference['cell'].isin(reference['cell']), 'filtered_reason'] = 'celltype not in only_clusters'
        else:
            reference = reference[reference['cluster'] == args.only_clusters]
            full_reference.loc[~full_reference['cell'].isin(reference['cell']), 'filtered_reason'] = 'celltype not in only_clusters'
    if args.subsample_n_cells is not None:
        reference = reference.sample(n=int(args.subsample_n_cells), random_state=args.seed)

    if color_config is not None:
        try:
            with open(os.path.join('data/dataset_colors', color_config), 'r') as f:
                color_dict = json.load(f)
            if args.only_clusters is not None:
                cluster_names_tmp = list(color_dict['names'])
                cluster_names = []
                for c in cluster_names_tmp:
                    if c in args.only_clusters:
                        cluster_names.append(c)
            else:
                cluster_names = list(color_dict['names'])
        except FileNotFoundError:
            print('No color config found, using default colors')
            cluster_names = list(reference['cluster'].unique())
    else:
        cluster_names = list(reference['cluster'].unique())
    
    reference.set_index('cell', inplace=True)
    if valid_reference is not None:
        valid_reference.set_index('cell', inplace=True)
    
    if verbose:
        console.print(f"[magenta]Total Cells: {len(reference)}[/]")

    sparse_matrices = {}
    sparse_matrices_filename = 'sparse_matrices_%s_%s.sav' % (dataset_name, res_name)
    

    if args.simulate and not args.append_simulated:
        scool_file = None
        content_of_scool = None
        coolers = os.listdir(args.bulk_coolers_dir)
        if coolers[0].endswith('.mcool'):
            c = cooler.Cooler(os.path.join(args.bulk_coolers_dir, coolers[0] + '::resolutions/10000'))
        else:
            c = cooler.Cooler(os.path.join(args.bulk_coolers_dir, coolers[0]))
        anchor_list = c.bins()[:]
        anchor_list = anchor_list[['chrom', 'start', 'end']]
        anchor_list['anchor'] = np.arange(len(anchor_list))
        anchor_list.rename(columns={'chrom': 'chr'}, inplace=True)
        anchor_list = anchor_list.dropna().reset_index(drop=True)
        anchor_list.to_csv('bins_bulk.tsv', sep='\t', index=False)
        anchor_dict = anchor_list_to_dict(anchor_list['anchor'].values)  # convert to anchor --> index dictionary
    elif args.scool is None:
        try:
            anchor_list = pd.read_csv(anchor_file, sep='\t', names=['chr', 'start', 'end', 'anchor', 'length'],
                                    usecols=['chr', 'start', 'end', 'anchor'], engine='python')  # read anchor list file
        except ValueError:
            anchor_list = pd.read_csv(anchor_file, sep='\t', names=['chr', 'start', 'end', 'anchor', 'length', '?'],
                                    usecols=['chr', 'start', 'end', 'anchor'], engine='python')  # read anchor list file
        anchor_list['anchor'] = anchor_list['anchor'].astype(str)
        
        anchor_dict = anchor_list_to_dict(anchor_list['anchor'].values)  # convert to anchor --> index dictionary

        
        if load_data:
            os.makedirs('data/sparse_matrices', exist_ok=True)
            if sparse_matrices_filename in os.listdir('data/sparse_matrices'):
                print('Loading sparse matrix data...')
                with open(os.path.join('data/sparse_matrices', sparse_matrices_filename), 'rb') as f:
                    sparse_matrices = joblib.load(f)
    else:
        data_dir = None
        content_of_scool = cooler.fileops.list_coolers(scool_file)
        c = cooler.Cooler(f"{scool_file}::{content_of_scool[0]}")
        anchor_list = c.bins()[:]
        anchor_list = anchor_list[['chrom', 'start', 'end']]
        anchor_list['anchor'] = np.arange(len(anchor_list))
        anchor_list['anchor'] = anchor_list['anchor'].astype(str)
        anchor_list.rename(columns={'chrom': 'chr'}, inplace=True)
        anchor_dict = anchor_list_to_dict(anchor_list['anchor'].values)  # convert to anchor --> index dictionary
    
    read_dist_ref = None
    if read_distribution is not None:  # filter out cells by mitotic or self-ligation contacts
        reference['mitotic_frac'] = 0
        reference['local_frac'] = 0
        try:
            read_dist_ref = pd.read_csv(read_distribution, sep='\t')
            read_dist_ref['cell'] = read_dist_ref['cell'].apply(lambda s: s + file_suffix)
            read_dist_ref = read_dist_ref[read_dist_ref['self_ratio'] <= 0.35]
            reference = reference[reference['cell'].isin(read_dist_ref['cell'])]
            full_reference.loc[~full_reference['cell'].isin(reference['cell']), 'filtered_reason'] = 'self_ratio > 0.35'
        except FileNotFoundError:
            print('Could not load read distribution file, are you sure it is there?', read_distribution)
            pass
        for i, row in read_dist_ref.iterrows():
            cellname = row['cell']
            if cellname in reference.index:
                reference.loc[cellname, 'local_frac'] = row['local']
                reference.loc[cellname, 'mitotic_frac'] = row['mitotic']
    n_cell_types = args.n_cell_types
    if n_cell_types is None:
        n_cell_types = len(reference['cluster'].unique())
    if assembly is None:
        # check if chr21 or chr 22 are present, if so, use hg19
        if 'chr21' in anchor_list['chr'].unique() or 'chr22' in anchor_list['chr'].unique():
            assembly = 'hg19'
        else:  # otherwise assume mouse genome
            assembly = 'mm10'
        console.print(f"[magenta]No assembly provided, assuming {assembly}[/]")
   
    train_generator = DataGenerator(sparse_matrices, anchor_list, anchor_dict, data_dir, reference, full_reference=full_reference, res_name=res_name, scool_file=scool_file, scool_contents=content_of_scool, assembly=assembly,
                                    n_clusters=n_cell_types, class_names=cluster_names, resolution=resolution, downsample=downsample, 
                                    simulate_from_bulk=args.simulate, bulk_cooler_dir=args.bulk_coolers_dir, simulate_n=args.simulate_n, simulate_depth=args.simulate_depth,
                                    filter_cells_by_depth=filter_by_depth, ignore_chr_filter=args.ignore_chr_filter,
                                    standardize=args.gaussian_output, dataset_name=dataset_name, preprocessing=args.schictools_preprocessing, limit2Mb=n_strata, rotated_offset=strata_offset, binarize=binarize, color_config=color_config, 
                                    active_regions=None, no_viz=args.no_viz)
    valid_generator = None
    if args.valid_clusters is not None:
        valid_generator = DataGenerator(sparse_matrices, anchor_list, anchor_dict, data_dir, valid_reference, res_name=res_name, scool_file=scool_file, scool_contents=content_of_scool, assembly=assembly,
                                    n_clusters=n_cell_types, resolution=resolution, downsample=downsample, 
                                    simulate_from_bulk=args.simulate, bulk_cooler_dir=args.bulk_coolers_dir, simulate_n=args.simulate_n, simulate_depth=args.simulate_depth,
                                    filter_cells_by_depth=filter_by_depth, ignore_chr_filter=args.ignore_chr_filter,
                                    standardize=args.gaussian_output, dataset_name=dataset_name, preprocessing=args.schictools_preprocessing, limit2Mb=n_strata, rotated_offset=strata_offset, binarize=binarize, color_config=color_config, 
                                    active_regions=None, no_viz=args.no_viz)


    if filter_mitotic:
        read_dist_ref = train_generator.check_mitotic_cells(from_frags=False)
        read_dist_ref['cell'] = read_dist_ref['cell'].apply(lambda s: s + f'.{res_name}')
        read_dist_ref = read_dist_ref[read_dist_ref['mitotic'] <= args.mitotic_max]
        read_dist_ref = read_dist_ref[read_dist_ref['mitotic'] >= args.mitotic_min]
        new_cell_list = []
        new_cell_list = read_dist_ref['cell'].to_list()
        train_generator.cell_list = new_cell_list
        train_generator.n_cells = len(new_cell_list)
        try:
            train_generator.reference = train_generator.reference.loc[new_cell_list]
        except KeyError:
            new_cell_list = [s.replace(f".{train_generator.res_name}", "") for s in new_cell_list]
            train_generator.reference = train_generator.reference.loc[new_cell_list]


    if verbose:
        for cluster_name in cluster_names:
            console.print(f"[green]{cluster_name}[/]: {np.sum(train_generator.reference['cluster'] == cluster_name)}")
    
    x, y_train, depths, batches = init_dataset(train_generator, load_data, binarize, sparse_matrices_filename)

    return args, x, y_train, depths, batches, train_generator, valid_generator
