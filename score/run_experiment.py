import os 
import sys
import time
import shutil
import numpy as np
from importlib.resources import files
from tqdm import tqdm, trange

from score.experiments.bandnorm_experiment import BandNormExperiment
from score.experiments.threedvi_experiment import DVIExperiment
from score.experiments.toki_experiment import TokiExperiment, call_tads
from score.experiments.IS_experiment import ISExperiment, run_ins_score
from score.experiments.lsi_2d_experiment import LSI2DExperiment
from score.experiments.pca_2d_experiment import PCA2DExperiment
from score.experiments.scvi_experiment import ScVIExperiment
from score.experiments.peakvi_experiment import PeakVIExperiment
from score.experiments.lda_experiment import LDAExperiment
from score.experiments.higashi_experiment import HigashiExperiment
from score.experiments.snapatac_experiment import SnapATACExperiment
from score.experiments.fast_higashi_experiment import FastHigashiExperiment

from score.utils.matrix_ops import get_flattened_matrices
from score.utils.utils import matrix_to_interaction
from score.methods.threeDVI import train_3dvi
from score.data_helpers.higashi_config import write_higashi_config



def write_cisTopic_data(interaction_dir, dataset, anchor_file, norm=True, rw=False):
    out_dir = 'schic-topic-model/data/%s_%s' % (dataset.dataset_name, dataset.res_name)
    if not os.path.exists(interaction_dir):
        os.makedirs(out_dir, exist_ok=True)
        dataset.write_lda_data(out_dir, norm=norm, rw=rw)
        os.makedirs(interaction_dir, exist_ok=True)
        if anchor_file is None:  # if using .scool, we need to write a separate bin file for cisTopic to read
            anchor_file = f"schic-topic-model/cisTopic_bins_{dataset.res_name}.bed"
            dataset.anchor_list.to_csv(anchor_file, sep='\t', header=False, index=False)
            matrix_to_interaction(out_dir, anchor_file, dataset.resolution, interaction_dir)
        else:
            matrix_to_interaction(out_dir, anchor_file, dataset.resolution, interaction_dir)


def pca_2d_exp(x, y, features, dataset, args, exp_name=None, operations=None, load_results=False, wandb_config=None):
    import anndata as ad
    import pandas as pd
    import scanpy as sc
    start_time = time.time()
    if load_results:
        x = None 
    else:
        chr_pcs = []
        chr_list = list(pd.unique(dataset.anchor_list['chr']))
        for chr_name in tqdm(chr_list):
            x = get_flattened_matrices(dataset, int(args.n_strata), preprocessing=operations, chr_only=chr_name, 
                                       rw_iter=args.random_walk_iter, rw_ratio=args.random_walk_ratio)
            if operations is None:  # using raw count data
                hic = ad.AnnData(x, dtype='int32')
            else:  # using whatever values are passed in (preprocessed data like normalized probs)
                hic = ad.AnnData(x)
            hic.obs_names = sorted(dataset.cell_list)
            hic.obs_names = hic.obs_names.map(lambda s: s.replace(f'.{dataset.res_name}', ''))
            chr_anchors = dataset.anchor_list[dataset.anchor_list['chr'] == chr_name]
            genomic_pos = chr_anchors.apply(lambda row: f"{row['chr']}:{row['start']}-{row['end']}", axis=1)
            new_var_names = pd.concat([genomic_pos.iloc[k:] + f'-{k}' for k in range(args.n_strata)])
            hic.var_names = new_var_names
            if operations is None:
                sc.pp.filter_genes(hic, min_counts=1)
                sc.pp.filter_genes(hic, min_cells=1)
            try:
                sc.tl.pca(hic, n_comps=min(args.latent_dim, hic.shape[1] - 1, hic.shape[0] - 1))
                chr_pcs.append(np.nan_to_num(np.array(hic.obsm['X_pca'])))
            except:  # not enough reads for PCA
                pass
        x = np.concatenate(chr_pcs, axis=1)
    if exp_name is None:
        exp_name = '2d_pca'
        if operations is not None:
            exp_name += ':' + ','.join(operations)
    exp = PCA2DExperiment(exp_name, x, y, features, dataset, preprocessing=operations, n_strata=int(args.n_strata), n_experiments=int(args.n_runs), simulate=args.simulate, append_simulated=args.append_simulated, other_args=args)
    exp.run(load=load_results, log_wandb=args.wandb, start_time=start_time, wandb_config=wandb_config)


def lsi_2d_exp(x, y, features, dataset, args, operations=None, load_results=False, wandb_config=None):
    start_time = time.time()
    if load_results:
        x = None 
    else:
        x = get_flattened_matrices(dataset, int(args.n_strata), preprocessing=operations, rw_iter=args.random_walk_iter, rw_ratio=args.random_walk_ratio)
    exp_name = '2d_lsi'
    if operations is not None:
        exp_name += ':' + ','.join(operations)
    exp = LSI2DExperiment(exp_name, x, y, features, dataset, preprocessing=operations, n_strata=int(args.n_strata), n_experiments=int(args.n_runs), simulate=args.simulate, append_simulated=args.append_simulated, other_args=args)
    exp.run(load=load_results, log_wandb=args.wandb, start_time=start_time, wandb_config=wandb_config)

def snap_atac_exp(x, y, features, dataset, args, operations=None, load_results=False, wandb_config=None):
    start_time = time.time()
    if load_results:
        x = None 
    else:
        x = get_flattened_matrices(dataset, int(args.n_strata), preprocessing=operations, rw_iter=args.random_walk_iter, rw_ratio=args.random_walk_ratio)
    exp_name = 'snapatac'
    if operations is not None:
        exp_name += ':' + ','.join(operations)
    exp = SnapATACExperiment(exp_name, x, y, features, dataset, preprocessing=operations, n_strata=int(args.n_strata), n_experiments=int(args.n_runs), simulate=args.simulate, append_simulated=args.append_simulated, other_args=args)
    exp.run(load=load_results, log_wandb=args.wandb, start_time=start_time, wandb_config=wandb_config)



def cisTopic_exp(x, y, features, dataset, args, exp_name=None, operations=None, minc=8, maxc=32, load_results=False, wandb_config=None):
    start_time = time.time()
    if load_results:
        x = None 
    else:
        x = get_flattened_matrices(dataset, int(args.n_strata), preprocessing=operations, rw_iter=args.random_walk_iter, rw_ratio=args.random_walk_ratio)
        #x = np.array([x.A.ravel().squeeze() for x in x])
    if exp_name is None:
        exp_name = 'cistopic'
        if operations is not None:
            exp_name += ':' + ','.join(operations)
    exp = LDAExperiment(exp_name, x, y, features, dataset, n_strata=int(args.n_strata), preprocessing=operations, n_experiments=int(args.n_runs), simulate=args.simulate, append_simulated=args.append_simulated, other_args=args)
    exp.run(load=load_results, log_wandb=args.wandb, start_time=start_time, wandb_config=wandb_config)


def scVI_exp(x, y, features, dataset, args, operations, load_results=False, wandb_config=None, one_dim=False):
    start_time = time.time()
    if load_results:
        x = None 
    else:
        if one_dim:
            x = dataset.write_cell_bin_matrix(max_dist=int(args.n_strata))
        else:
            x = get_flattened_matrices(dataset, int(args.n_strata), preprocessing=operations, rw_iter=args.random_walk_iter, rw_ratio=args.random_walk_ratio)
    exp_name = 'scvi'
    if not one_dim:
        exp_name += '_2d'
    if operations is not None:
        exp_name += ':' + ','.join(operations)
    exp = ScVIExperiment(exp_name, x, y, features, dataset, n_strata=1 if one_dim else int(args.n_strata), preprocessing=operations, n_experiments=int(args.n_runs), simulate=args.simulate, append_simulated=args.append_simulated, other_args=args)
    exp.run(load=load_results, log_wandb=args.wandb, start_time=start_time, wandb_config=wandb_config)


def peakVI_exp(x, y, features, dataset, args,load_results=False, wandb_config=None, one_dim=False):
    start_time = time.time()
    if load_results:
        x = None 
    else:
        if one_dim:
            x = dataset.write_cell_bin_matrix(max_dist=int(args.n_strata))
        else:
            x = get_flattened_matrices(dataset, int(args.n_strata), rw_iter=args.random_walk_iter, rw_ratio=args.random_walk_ratio)
    name = 'peakvi'
    if not one_dim:
        name += '_2d'
    exp = PeakVIExperiment(name, x, y, features, dataset, n_strata=1 if one_dim else int(args.n_strata), n_experiments=int(args.n_runs), simulate=args.simulate, append_simulated=args.append_simulated, other_args=args)
    exp.run(load=load_results, log_wandb=args.wandb, start_time=start_time, wandb_config=wandb_config)


def scGAD_exp(x, y, features, dataset, anchor_file, args, exp_name, operations, depth_norm=True, random_walk=False, load_results=False, wandb_config=None):
    if load_results:
        experiment = BandNormExperiment(exp_name, x, y, features, dataset, args, depth_norm=depth_norm, eval_inner=False, simulate=args.simulate, append_simulated=args.append_simulated, other_args=args)
        for run_i in range(int(args.n_runs)):
            start_time = time.time()
            experiment.run(load=load_results, outer_iter=run_i, start_time=start_time)
    else:
        interaction_dir = 'schic-topic-model/data/%s_%s_interactions' % (dataset.dataset_name, dataset.res_name)
        if random_walk:
            write_cisTopic_data(interaction_dir, dataset, anchor_file, norm=True, rw=True)
        else:
            write_cisTopic_data(interaction_dir, dataset, anchor_file, norm=False)
        bandnorm_out_dir = 'BandNorm/output'
        os.makedirs(bandnorm_out_dir, exist_ok=True)
        experiment = BandNormExperiment(exp_name, x, y, features, dataset, args, depth_norm=depth_norm, eval_inner=False, simulate=args.simulate, append_simulated=args.append_simulated, other_args=args)
        for directory in sys.path:
            scgad_script = os.path.join(directory, 'score/methods/BandNorm/train.R')
            if os.path.isfile(scgad_script):
                scgad_cmd = f"Rscript {scgad_script} {interaction_dir} "
                out_file = f"{bandnorm_out_dir}/{dataset.dataset_name}_{dataset.res_name}.tsv"
                scgad_cmd += f"{dataset.assembly} {out_file} "
                #scgad_cmd += f"/mnt/rds/genetics01/JinLab/dmp131/score/score/methods/BandNorm/islet_anchors {out_file} "
                if depth_norm:
                    scgad_cmd += "TRUE"  # depth norm
                else:
                    scgad_cmd += "FALSE"  # depth norm
                if args.rna_file is not None:
                    scgad_cmd += f" {args.rna_file}"
                    scgad_cmd += f" {args.reference}"
                    if args.atac_file is not None:
                        scgad_cmd += f" {args.atac_file}"
                for run_i in range(int(args.n_runs)):
                    start_time = time.time()
                    os.system(scgad_cmd)
                    experiment.run(load=load_results, outer_iter=run_i, start_time=start_time, log_wandb=args.wandb, wandb_config=wandb_config)
                break


def threeDVI_exp(x, y, features, dataset, anchor_file, args, exp_name, load_results=False, random_walk=False, wandb_config=None):
    if load_results:
        experiment = DVIExperiment(exp_name, x, y, features, dataset, n_experiments=int(args.n_runs), eval_inner=False, simulate=args.simulate, append_simulated=args.append_simulated, load_results=True, other_args=args)
        for run_i in range(int(args.n_runs)):
            start_time = time.time()
            experiment.run(load=True, outer_iter=run_i, start_time=start_time)
    else:
        out_dir = 'schic-topic-model/data/%s_%s' % (dataset.dataset_name, dataset.res_name)
        interaction_dir = 'schic-topic-model/data/%s_%s_interactions' % (dataset.dataset_name, dataset.res_name)
        if not os.path.exists(interaction_dir):
            os.makedirs(out_dir, exist_ok=True)
            if random_walk:
                dataset.write_lda_data(out_dir, norm=True, rw=True)
            else:
                dataset.write_lda_data(out_dir, norm=False)
            os.makedirs(interaction_dir, exist_ok=True)
            if anchor_file is None:  # if using .scool, we need to write a separate bin file for cisTopic to read
                anchor_file = f"schic-topic-model/cisTopic_bins_{dataset.res_name}.bed"
                dataset.anchor_list.to_csv(anchor_file, sep='\t', header=False, index=False)
                matrix_to_interaction(out_dir, anchor_file, dataset.resolution, interaction_dir)
            else:
                matrix_to_interaction(out_dir, anchor_file, dataset.resolution, interaction_dir)
        ref_dir = 'threeDVI/reference/%s_%s' % (dataset.dataset_name, dataset.res_name)
        dataset.write_3dvi_ref(ref_dir)
        experiment = DVIExperiment('3dvi', x, y, features, dataset, n_experiments=int(args.n_runs), eval_inner=False, simulate=args.simulate, append_simulated=args.append_simulated, other_args=args)
        for run_i in range(int(args.n_runs)):
            start_time = time.time()
            chrom_sizes_fp = str(files("score").joinpath(f"static/{dataset.assembly}.chrom.sizes.txt"))
            train_3dvi(dataset.limit2Mb, 'whole', dataset.resolution, interaction_dir, 
                        f"threeDVI/results/{dataset.dataset_name}_{dataset.res_name}",
                        f"{ref_dir}/data_summary.txt", chrom_sizes_fp, batchRemoval=True,
                        nLatent=64, gpuFlag=True, parallelCPU=1, pcaNum=50, umapPlot=True, tsnePlot=False, n_epochs=args.three3dvi_epochs)
            
            experiment.run(load=load_results, outer_iter=run_i, start_time=start_time, log_wandb=args.wandb, wandb_config=wandb_config)


def toki_exp(x, y, features, dataset, args, load_results=False, wandb_config=None):
    toki_dir = os.path.join(args.toki_dir, dataset.dataset_name, dataset.res_name)
    tad_dir = os.path.join(args.toki_dir, dataset.dataset_name, dataset.res_name + '_TAD')
    os.makedirs(toki_dir, exist_ok=True)
    os.makedirs(tad_dir, exist_ok=True)
    start_time = time.time()
    if load_results:
        exp_name = "detoki"
        experiment = TokiExperiment(exp_name, x, y, features, dataset, tad_dir=tad_dir, simulate=args.simulate, append_simulated=args.append_simulated, other_args=args)
        experiment.run(load=True, start_time=start_time)
    else:
        current_tads = os.listdir(tad_dir)
        for cell_i, cell_name in tqdm(enumerate(sorted(dataset.cell_list)), total=len(dataset.cell_list)):
            if (cell_name) not in current_tads:
                cell_i, cell_name, tads = call_tads(cell_i, cell_name, dataset, os.path.join(toki_dir, str(cell_i)), not args.tad_count, core_n=args.n_threads)
                np.savetxt(os.path.join(tad_dir, cell_name), tads)
                shutil.rmtree(os.path.join(toki_dir, str(cell_i)))  # delete dense matrices and intermediate chr TADs

        exp_name = "detoki"
        experiment = TokiExperiment(exp_name, x, y, features, dataset, tad_dir=tad_dir, simulate=args.simulate, append_simulated=args.append_simulated, n_experiments=int(args.n_runs), other_args=args)
        experiment.run(load=False, log_wandb=args.wandb, start_time=start_time, wandb_config=wandb_config)


def ins_exp(x, y, features, dataset, args, load_results=False, wandb_config=None):
    tad_count = args.tad_count
    toki_dir = os.path.join(args.ins_dir, dataset.dataset_name, dataset.res_name)
    if not tad_count:
        tad_dir = os.path.join(args.ins_dir, dataset.dataset_name, dataset.res_name + '_INS')
    else:
        tad_dir = os.path.join(args.ins_dir, dataset.dataset_name, dataset.res_name + '_INS_count')
    os.makedirs(toki_dir, exist_ok=True)
    os.makedirs(tad_dir, exist_ok=True)
    start_time = time.time()
    if load_results:
        experiment = ISExperiment("insscore", x, y, features, dataset, tad_dir=tad_dir, simulate=args.simulate, append_simulated=args.append_simulated, n_experiments=int(args.n_runs), other_args=args)
        experiment.run(load=load_results, start_time=start_time)
    else:
        run_ins_score(tad_dir, toki_dir, dataset, args.assembly, n_threads=args.n_threads)
        experiment = ISExperiment("insscore", x, y, features, dataset, tad_dir=tad_dir, simulate=args.simulate, append_simulated=args.append_simulated, n_experiments=int(args.n_runs), other_args=args)
        experiment.run(load=load_results, start_time=start_time, log_wandb=args.wandb, wandb_config=wandb_config)


def higashi_exp(x, y, features, dataset, args, load_results=False, fast_higashi=True, operations=None, wandb_config=None, loops_file=None, run_scghost=False):
    from higashi.Higashi_wrapper import Higashi
    from fasthigashi.FastHigashi_Wrapper import FastHigashi
    if not load_results:
        out_dir = 'data/higashi_data/%s_%s' % (dataset.dataset_name, dataset.res_name)
        os.makedirs(out_dir, exist_ok=True)
        data_file = os.path.join(out_dir, 'data.txt')
        #if 'data.txt' not in os.listdir(out_dir):
        dataset.write_higashi_data(data_file, os.path.join(out_dir, 'label_info.pickle'), loops_file=loops_file, n_strata=args.higashi_n_strata, strata_offset=args.higashi_strata_offset, operations=operations)
        config = f"data/higashi_data/{dataset.dataset_name}_config_{dataset.res_name}.json"
        chrom_sizes_fp = str(files("score").joinpath(f"static/{dataset.assembly}.chrom.sizes.txt"))
        cytoband_fp = str(files("score").joinpath(f"static/{dataset.assembly}_cytoBand.txt"))
        human_config_fp = str(files("score").joinpath("static/human_config.json"))
        mouse_config_fp = str(files("score").joinpath("static/mouse_config.json"))
        write_higashi_config(f"{dataset.dataset_name}_{dataset.res_name}", 
                            chrom_sizes_fp, 
                            cytoband_fp,
                            config,
                            human_config_fp if dataset.assembly in ['hg19', 'hg38'] else mouse_config_fp,
                            int(args.n_strata) * dataset.resolution, dataset.resolution, 'data/higashi_data/', int(args.higashi_epochs))
    if fast_higashi:
        exp_name = 'fast_higashi'
    elif run_scghost:
        exp_name = 'scghost'
    else:
        exp_name = 'higashi'
    if operations is not None:
        exp_name += ':' + ','.join(operations)
    if fast_higashi:
        experiment = FastHigashiExperiment('fast_higashi', x, y, features, dataset, depth_norm=not args.no_depth_norm, eval_inner=False, simulate=args.simulate, append_simulated=args.append_simulated, other_args=args)
    else:
        experiment = HigashiExperiment('scghost' if run_scghost else 'higashi', x, y, features, dataset, eval_inner=False, simulate=args.simulate, append_simulated=args.append_simulated, other_args=args)
    
    if fast_higashi:
        # set torch device
        import torch
        torch.cuda.is_available = lambda : False
        # Initialize the Higashi instance
        for run_i in range(int(args.n_runs)):
            start_time = time.time()
            if not load_results:
                higashi_model = Higashi(config)
                model = FastHigashi(config_path=config,
                    path2input_cache=out_dir,
                    path2result_dir=out_dir,
                    off_diag=args.n_strata,
                    filter=False,
                    do_conv=True,
                    do_rwr=True,
                    do_col=False,
                    no_col=False)
                higashi_model.process_data()
                model.prep_dataset()


                model.run_model(dim1=.6,
                    rank=min(args.latent_dim, dataset.n_cells),
                    n_iter_parafac=1,
                    extra="")

                experiment.model = model

            experiment.run(load=load_results, outer_iter=run_i, start_time=start_time, log_wandb=args.wandb, wandb_config=wandb_config)
    else:
        # Initialize the Higashi instance
        for run_i in range(int(args.n_runs)):
            start_time = time.time()
            if not load_results:
                higashi_model = Higashi(config)
                if run_i == 0:
                    #Data processing (only needs to be run for once)
                    higashi_model.process_data()

                higashi_model.prep_model()
                if not args.higashi_dryrun:
                    higashi_model.train_for_embeddings()
                else:
                    higashi_model.save_embeddings()

            if run_scghost:
                higashi_model.train_for_imputation_nbr_0()
                higashi_model.train_for_imputation_with_nbr()
                higashi_model.impute_with_nbr()
                import higashi
                import json
                import pickle
                import subprocess
                import matplotlib.pyplot as plt

                higashi_path = "/".join(higashi.__file__.split("/")[:-1])

                # first write anchor reference to bed file for cpg computation
                dataset.anchor_list[['chr', 'start', 'end']].to_csv(os.path.join(higashi_path, "anchors.bed"), sep="\t", header=False, index=False)
                cpg_file = os.path.join(out_dir, "anchors.cpg.txt")
                cpg_cmd = ["sh", 
                           os.path.join(higashi_path, "cpg.sh"), 
                           "/mnt/rstor/genetics/JinLab/xxl244/Reference_Indexes/mm10_bowtie_index/mm10.fa",
                           os.path.join(higashi_path, "anchors.bed"),
                           '>', cpg_file]
                subprocess.call(cpg_cmd)

                command = ["python", os.path.join(higashi_path, "scCompartment.py"), "-c", config, "--calib_file", cpg_file, "--calib", "--neighbor"]
                subprocess.call(command)

                count = 0
                fig = plt.figure(figsize=(12, 4*5))
                for id_ in np.random.randint(0, 148, 5):
                    # code to fetch imputed contact maps
                    ori, nbr5 = higashi_model.fetch_map("chr3", id_)
                    count += 1
                    ax = plt.subplot(5, 2, count * 2 - 1)
                    ax.imshow(ori.toarray(), cmap='Reds', vmin=0.0, vmax=np.quantile(ori.data, 0.95))
                    ax.set_xticks([], [])
                    ax.set_yticks([], [])
                    if count == 1:
                        ax.set_title("raw")
                    
                    ax = plt.subplot(5, 2, count * 2)
                    ax.imshow(nbr5.toarray(), cmap='Reds', vmin=0.0, vmax=np.quantile(nbr5.data, 0.95))
                    ax.set_xticks([], [])
                    ax.set_yticks([], [])
                    if count == 1:
                        ax.set_title("higashi, k=5")
                    
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, "imputed_contact_maps.png"))
                plt.close()


                with open(config,"r") as f:
                    config_info = json.load(f)
                scg_config = f"data/higashi_data/{dataset.dataset_name}_config_{dataset.res_name}_scghost.json"
                # filepath settings
                schic_directory = config_info['temp_dir']
                label_info_path = os.path.join(config_info['data_dir'], "label_info.pickle")
                label_info_cell_type_key = "cell type"
                scg_out_directory = os.path.join(config_info['temp_dir'] + "scghost")
                NUM_CHROMOSOMES = len(config_info['chrom_list'])
                embed_name = config_info['embedding_name']
                chrom_list = config_info['chrom_list']
                nbr_num= config_info['neighbor_num']
                chromosomes = {chrom_num+1 : {
                    'adj' : f'raw/{chrom}_sparse_adj.npy',
                    'imputed' : f'{chrom}_{embed_name}_nbr_{nbr_num}_impute.hdf5',
                    'integer' : chrom_num+1,
                } for chrom_num,chrom in enumerate(chrom_list)}

                chrom_sizes = config_info['genome_reference_path']
                chrom_indices = None
                embeddings_path = os.path.join(config_info['temp_dir'], 'embed', config_info['embedding_name']+"_0_origin.npy")
                higashi_scab_path = os.path.join(config_info['temp_dir'], 'scCompartment.hdf5')

                # hyperparameters
                random_walk_num_walks = 50
                random_walk_ignore_top = 0.02
                random_walk_top_percentile = 0.25
                eps = 1e-8
                num_clusters = 5
                batch_size = 16
                epochs = 5
                resolution = 500000
                neighbor_contacts = False
                kmeans_init = 1

                # misc settings
                nearest_neighbor_override = None
                gpu_uniques = True
                cluster_gpu_caching = True
                settings_dict = {
                    'schic_directory': schic_directory,
                    'label_info': {
                        'path': label_info_path,
                        'cell_type_key': label_info_cell_type_key,
                    },
                    'data_directory': scg_out_directory,
                    'chromosomes': chromosomes,
                    'chrom_sizes': chrom_sizes,
                    'chrom_indices': chrom_indices,
                    'embeddings_path': embeddings_path,
                    'higashi_scab_path': higashi_scab_path,
                    'cell_type': None,
                    'random_walk': {
                        'num_walks': random_walk_num_walks,
                        'ignore_top': random_walk_ignore_top,
                        'top_percentile': random_walk_top_percentile,
                    },
                    'eps': eps,
                    'num_clusters': num_clusters,
                    'batch_size': batch_size,
                    'epochs': epochs,
                    'resolution': resolution,
                    'neighbor_contacts': neighbor_contacts,
                    'nearest_neighbor_override': nearest_neighbor_override,
                    'gpu_uniques': gpu_uniques,
                    'cluster_gpu_caching': cluster_gpu_caching,
                    'kmeans_init': kmeans_init,
                }
                with open(scg_config, "w") as outfile: 
                    json_string = json.dumps(settings_dict, indent=4)
                    outfile.write(json_string)
                scghost_path = '/mnt/rds/genetics01/JinLab/dmp131/scGHOST'
                scg_script_path = os.path.join(scghost_path, "scghost.py")
                subprocess.call(['python {script_path} --config {json}'.format(script_path=scg_script_path, json=scg_config)],shell=True)
                # enter labels.pkl path
                label_filepath = os.path.join(scg_out_directory, 'labels.pkl')
                labels = pickle.load(open(label_filepath,'rb'))

                # enter cropped_indices.pkl path
                cropped_indices_filepath = os.path.join(scg_out_directory, 'cropped_indices.pkl')
                cropped_indices = pickle.load(open(cropped_indices_filepath,'rb'))

                # enter resolution
                resolution = dataset.resolution

                # enter bed file output directory
                bed_file_directory = os.path.join(scg_out_directory, 'bed_files')
                chrom_prefix = 'chr' # change this to '' if chromosomes are labeled chr1,chr2,... instead of 1,2,...

                sc_subcompartment_names = ['scA1','scA2','scB1','scB2','scB3'] # default for scGHOST k=5

                os.makedirs(bed_file_directory,exist_ok=True)

                num_cells = labels[ list( labels.keys() )[0] ].shape[0]

                for cell_num in trange(num_cells):

                    with open(os.path.join(bed_file_directory,f'cell_{cell_num}.bed'),'w') as f:

                        for chromosome in labels:

                            annotations = labels[chromosome][cell_num]

                            for locus in range(len(annotations)):

                                position = cropped_indices[chromosome][locus]
                                annotation = sc_subcompartment_names[ annotations[locus] ]

                                line = f'{chrom_prefix}{chromosome}\t{int(position * resolution)}\t{int((position+1) * resolution)}\t{annotation}\n'
                                f.write(line)

                def get_expected(M,eps=1e-8):
                    E = np.zeros_like(M)
                    l = len(M)

                    for i in range(M.shape[0]):
                        contacts = np.diag(M,i)
                        expected = contacts.sum() / (l-i)
                        # expected = np.median(contacts)
                        x_diag,y_diag = np.diag_indices(M.shape[0]-i)
                        x,y = x_diag,y_diag+i
                        E[x,y] = expected

                    E += E.T
                    E = np.nan_to_num(E) + eps
                    
                    return E
                    
                def get_oe_matrix(M):
                    E = get_expected(M)
                    oe = np.nan_to_num(M / E)
                    np.fill_diagonal(oe,1)
                    
                    return oe
                    

                stacked_pcs = []

                for chrom_num in range(1,20):
                    chrom_indices = pickle.load(open(os.path.join(scg_out_directory, 'chrom_indices.pkl'),'rb'))['%d' % chrom_num]
                    sparse_M = np.load(os.path.join(schic_directory, chromosomes[chrom_num]['adj']),allow_pickle=True)
                    pseudo_bulk = sparse_M.sum(axis=0).toarray()
                    cov = np.sqrt(pseudo_bulk.sum(axis=1))
                    pseudo_bulk /= cov[None,:]
                    pseudo_bulk /= cov[:,None]
                    pseudo_bulk = np.nan_to_num(pseudo_bulk)[chrom_indices][:,chrom_indices]
                    pseudo_OE = get_oe_matrix(pseudo_bulk)

                    Rpool = np.nan_to_num(np.corrcoef(pseudo_OE))
                    Rpoolmean = Rpool.mean(axis=0,keepdims=True)
                    Rpool = Rpool - Rpoolmean
                    _,_,V = np.linalg.svd(Rpool)

                    Es = np.load(os.path.join(scg_out_directory, f'{chrom_num}_embeddings.npy'))
                    embedding_corrs = np.zeros((Es.shape[0],Es.shape[1],Es.shape[1]))

                    num_cells = len(Es)

                    for i in trange(num_cells):
                        embedding_corrs[i] = np.corrcoef(Es[i])

                    pcs = np.zeros((Es.shape[0],Es.shape[1]))

                    for i,ec in enumerate(embedding_corrs):
                        tec = ec - Rpoolmean
                        pc = tec.dot(V[0,:].T)
                        pcs[i] = pc
                        
                    stacked_pcs.append(pcs)
                    
                stacked_pcs = np.hstack(stacked_pcs)
                np.save(os.path.join(scg_out_directory, 'pcs.npy'),stacked_pcs)
                experiment.run(load=load_results, outer_iter=run_i, start_time=start_time, log_wandb=args.wandb, wandb_config=wandb_config)
            else:
                experiment.run(load=load_results, outer_iter=run_i, start_time=start_time, log_wandb=args.wandb, wandb_config=wandb_config)
                
