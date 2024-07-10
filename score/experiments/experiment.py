import os
import time
import json
import pickle
import numpy as np
import pandas as pd

from rich.console import Console
from munkres import Munkres
from functools import partial
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score, \
    homogeneity_score, completeness_score, v_measure_score, roc_auc_score, silhouette_score

console = Console()


class Experiment():
    def __init__(self, name, x, y, features, data_generator, n_experiments=5, eval_inner=True, simulate=False, append_simulated=False, save_dir='results', 
                 resolution=None, skip_tsne=False, style='default', name_suffix=None, latent_dim=32,
                 schictools_data=None, val_schictools_data=None,
                 val_data=None, val_dataset=None, other_args=None):
        import matplotlib
        import matplotlib.pyplot as plt
        self.name = name  # the name of the experiment, determines save directories and labels
        self.dataset_name = data_generator.dataset_name
        self.x = x
        self.y = np.squeeze(np.asarray(y).ravel())
        self.cell_colors = None
        self.features_dict = features
        self.data_generator = data_generator  # data generator class storing all info about current dataset (cluster names, num points, etc)
        self.n_clusters = data_generator.n_clusters
        self.n_classes = data_generator.n_classes
        self.latent_dim = latent_dim
        self.cluster_names = data_generator.classes
        self.n_cells = self.data_generator.n_cells
        self.n_experiments = int(n_experiments)  # number of evaluations to compute
        self.eval_inner = eval_inner  # this indicates if calling get_embedding produces a new embedding or fetches a cached one
        self.simulate = simulate
        self.append_simulated = append_simulated
        self.save_dir = save_dir
        self.resolution_name = data_generator.res_name

        if resolution is None:
            self.resolution = 0
            if 'M' in self.resolution_name:
                self.resolution = int(self.resolution_name[:-1]) * 1000000
            elif 'kb' in self.resolution_name:
                self.resolution = int(self.resolution_name[:-2]) * 1000
        else:
            self.resolution = resolution
        self.resolution = int(self.resolution / data_generator.resize_amount)
        self.val_dataset = val_dataset
        if val_dataset is not None:
            self.val_x, self.val_y = val_data 
            self.val_y = np.squeeze(np.asarray(self.val_y).ravel())
            self.n_clusters = val_dataset.n_classes
            self.n_classes = val_dataset.n_classes
            self.n_cells = val_dataset.n_cells
        if other_args is None:
            self.other_args = {}
        else:
            self.other_args = other_args.__dict__
        if 'out' in self.other_args.keys():
            self.save_dir = self.other_args['out']
        self.random_walk_iter = 1
        if 'random_walk_iter' in self.other_args.keys():
            self.random_walk_iter = self.other_args['random_walk_iter']
        self.random_walk_ratio = 1.0
        if 'random_walk_ratio' in self.other_args.keys():
            self.random_walk_ratio = self.other_args['random_walk_ratio']
        self.skip_tsne = skip_tsne
        if name_suffix is None:
            self.out_dir = os.path.join(self.save_dir, data_generator.dataset_name, self.resolution_name, name)
        else:
            self.out_dir = os.path.join(self.save_dir, data_generator.dataset_name + '_' + name_suffix, self.resolution_name, name)
        self.schictools_data = schictools_data
        self.val_schictools_data = val_schictools_data
        self.plot_viz = not self.data_generator.no_viz

        os.makedirs(self.out_dir, exist_ok=True)
        self.metrics = {'wall_time': []}  # dictionary to store stats. starts with time, new metrics automatically added by defining new entry in metric_algs
        self.metrics_no_pc1 = {}
        self.current_metrics = {}
        self.current_metrics_no_pc1 = {}
        self.metric_algs = {'ari': adjusted_rand_score,
                        'nmi': normalized_mutual_info_score,
                        'ami': adjusted_mutual_info_score,
                        'homogeneity': homogeneity_score,
                        'completeness': completeness_score,
                        'v-measure': v_measure_score,
                        'accuracy': self.cluster_acc,
                        'silhouette': partial(silhouette_score, metric='cosine'),
                        'silhouette-gt': partial(silhouette_score, metric='cosine')}  # use cosine distance for comparing embeddings of different volumes
        self.clustering_algs = {'k-means': KMeans,
                                'agglomerative': AgglomerativeClustering,
                                'gmm': GaussianMixture}
        # populate the metrics dict with empty lists (each run will accumulate metrics)
        for metric_name in self.metric_algs.keys():
            for i, alg in enumerate(list(self.clustering_algs.keys()) + ['louvain', 'leiden']):  # louvain is a special case
                metric_alg_key = self.get_metric_alg_key(metric_name, alg)
                self.metrics[metric_alg_key] = []
                self.metrics_no_pc1[metric_alg_key] = []
            self.metrics[f"best_{metric_name}"] = []  # max over clustering algs
            self.metrics_no_pc1[f"best_{metric_name}"] = []
        for cluster in self.cluster_names:
            self.metrics['croc_%s' % cluster] = []
        if self.n_cells < 50:
            self.scatter_size = 30
        elif self.n_cells < 100:
            self.scatter_size = 24
        elif self.n_cells < 200:
            self.scatter_size = 20
        elif self.n_cells < 500:
            self.scatter_size = 14
        else:
            self.scatter_size = 4

        color_matrix = None
        self.cluster_names = list(self.data_generator.classes)
        if val_dataset is not None:
            self.cluster_names = list(self.val_dataset.classes)
        self.color_dict = None 
        if self.data_generator.color_config is not None:
            try:
                with open(os.path.join('data/dataset_colors', self.data_generator.color_config), 'r') as f:
                    self.color_dict = json.load(f)
            except FileNotFoundError:  
                pass 
        if self.color_dict is not None:  # provided color file and it was found
            color_matrix = []
            self.cluster_names = []
            for c in self.color_dict['names']:
                if c in data_generator.classes:
                    self.cluster_names.append(c)
            for cluster in self.cluster_names:
                color_matrix.append(self.color_dict['colors'][cluster])
            color_matrix = np.array(color_matrix)
            self.cluster_cmap = matplotlib.colors.ListedColormap(color_matrix / 255.0)
            self.color_dict = self.color_dict['colors']
            for c in self.color_dict.keys():
                self.color_dict[c] = np.array([x / 255.0 for x in self.color_dict[c]])
        elif self.n_clusters == 2:
            self.cluster_cmap = plt.cm.get_cmap("bwr", self.n_clusters)
        elif self.n_clusters == 3:
            self.cluster_cmap = plt.cm.get_cmap("brg", self.n_clusters)
        elif self.n_clusters <= 9:
            self.cluster_cmap = plt.cm.get_cmap("tab10", self.n_clusters)
        elif self.n_clusters <= 20:
            self.cluster_cmap = plt.cm.get_cmap("tab20", self.n_clusters)
        else:
            self.cluster_cmap = plt.cm.get_cmap("gist_ncar", self.n_clusters)
        if color_matrix is not None:
            self.save_color_config(color_matrix, self.cluster_names)
        plt.style.use(style)
        print()
        console.print(f"[bold yellow]{self.data_generator.dataset_name}[/] - [bold green]{name}[/]")

    def prepare_schictools_data(self, dataset):
        if self.rewrite or self.resolution_name not in os.listdir(self.dataset_dir) or self.simulate:
            self.rewrite = True
            network, self.chr_lengths = dataset.write_scHiCTools_matrices(out_dir=self.data_dir, resolution=self.resolution, rewrite=self.rewrite)
        else:
            network = sorted(dataset.cell_list)
            network = [os.path.join(self.data_dir, f) for f in network]
            self.chr_lengths = {}
            chr_list = list(pd.unique(dataset.anchor_list['chr']))
            for chr_name in chr_list:
                if chr_name == 'chrM':
                    continue
                chr_anchors = dataset.anchor_list.loc[dataset.anchor_list['chr'] == chr_name]
                chr_length = len(chr_anchors)
                self.chr_lengths[chr_name] = chr_length * self.resolution
        return network

    def load_schictools_data(self, network, full_maps=False):
        from scHiCTools import scHiCs
        chromosomes = 'except YM'
        if 'use_xy' in self.other_args.keys():
            if self.other_args['use_xy']:
                chromosomes = 'all'
        loaded_data = scHiCs(network, reference_genome=self.chr_lengths, resolution=self.resolution, chromosomes=chromosomes,
                            keep_n_strata=self.n_strata, strata_offset=self.strata_offset, exclusive_strata=False, store_full_map=full_maps, format='npz',
                            operations=self.operations, t=self.random_walk_iter, random_walk_ratio=self.random_walk_ratio)
        return loaded_data

    def save_color_config(self, color_matrix, cluster_names):
        color_dict = {'names': cluster_names, 'colors': {}}
        for cluster, c in zip(cluster_names, color_matrix):
            color_dict['colors'][cluster] = [int(x) for x in c]  # json can't serialize np.ndarray
        os.makedirs('data/dataset_colors', exist_ok=True)
        with open(os.path.join('data/dataset_colors', f"{self.data_generator.dataset_name}.json"), 'w') as f:
            json.dump(color_dict, f, indent=2)

    def get_metric_alg_key(self, metric_name, alg_name):
        return metric_name + '_' + alg_name

    def acroc(self, embedding, metric_suffix=''):
        center = np.mean(embedding, axis=0)
        thetas = []
        for i in range(embedding.shape[0]):
            embedding[i] -= center
        thetas = np.arctan2(embedding[..., 1], embedding[..., 0]) + np.pi
        aucs = []
        for cluster_label in np.unique(self.y):
            cluster_name = self.cluster_names[cluster_label]
            positive = self.y == cluster_label
            pos_thetas = thetas[positive]
            mean_theta = np.mean(pos_thetas)
            diffs = np.minimum(np.abs(thetas - mean_theta), np.abs(thetas - 2 * np.pi - mean_theta))
            probs = 1 - (diffs / diffs.max())
            auc = roc_auc_score(positive, probs)
            if auc < 0.5:
                auc = 1 - auc
            self.metrics['croc_%s' % cluster_name].append(auc)
            aucs.append(auc)
        acroc_name = f"acroc-{metric_suffix}"
        self.current_metrics[acroc_name] = np.mean(aucs)
        if acroc_name in self.metrics.keys():
            self.metrics[acroc_name].append(np.mean(aucs))
        else:
            self.metrics[acroc_name] = [np.mean(aucs)]
        return aucs

    @staticmethod
    def cluster_acc(y_true, y_pred):
        """
        Calculate clustering accuracy. Require scikit-learn and munkres package.
        # Arguments
            y: true labels, numpy.array with shape `(n_samples,)`
            y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        # Return
            accuracy, in [0,1]
        """
        y_true = y_true.astype(np.int64)
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        m = Munkres()
        ind = m.compute(w.max() - w)
        ind = np.array(ind)
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    def plot_from_adata(self, adata, iter, valid=False, additional_out_dir='', log_wandb=False):
        import matplotlib
        import matplotlib.pyplot as plt
        import plotly.express as px
        import scanpy as sc
        sc.settings.verbosity = 0
        import anndata as ad

        os.makedirs(os.path.join(self.out_dir, 'celltype_plots'), exist_ok=True)
        if additional_out_dir != '':
            os.makedirs(os.path.join(self.out_dir, f'celltype_plots{additional_out_dir}'), exist_ok=True)
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        sc.pl.pca(adata, color="celltype", ax=axs[0], show=False, title="PCA Embeddings", legend_loc=None, palette=self.color_dict)
        sc.pl.tsne(adata, color="celltype", ax=axs[1], show=False, title="t-SNE Embeddings", legend_loc=None, palette=self.color_dict)
        sc.pl.umap(adata, color="celltype", ax=axs[2], show=False, title="UMAP Embeddings", palette=self.color_dict)
        plt.tight_layout()
        fig.subplots_adjust(top=0.9)
        fig.suptitle(f"{self.dataset_name} - {self.name}")
        fig.savefig(os.path.join(self.out_dir, f'celltype_plots{additional_out_dir}/embedding.png'))
        fig.savefig(os.path.join(self.out_dir, f'celltype_plots{additional_out_dir}/embedding.pdf'))
        plt.close()

        # try:
        #     sc.pl.clustermap(adata, obs_keys='celltype', show=False)
        #     plt.savefig(os.path.join(self.out_dir, f'celltype_plots{additional_out_dir}/clustermap.png'))
        #     plt.savefig(os.path.join(self.out_dir, f'celltype_plots{additional_out_dir}/clustermap.pdf'))
        #     plt.close()

        #     fig, mat_ax = plt.subplots(figsize=(8, 8))
        #     sc.pl.matrixplot(adata, var_names=adata.var_names, groupby='celltype', dendrogram=True, cmap='Spectral', show=False, ax=mat_ax)
        #     fig.savefig(os.path.join(self.out_dir, f'celltype_plots{additional_out_dir}/matrixplot.png'))
        #     fig.savefig(os.path.join(self.out_dir, f'celltype_plots{additional_out_dir}/matrixplot.pdf'))
        #     plt.close()

        #     sc.pl.correlation_matrix(adata, 'celltype', show=False, cmap='RdYlBu_r')
        #     plt.savefig(os.path.join(self.out_dir, f'celltype_plots{additional_out_dir}/correlation_matrix.png'))
        #     plt.savefig(os.path.join(self.out_dir, f'celltype_plots{additional_out_dir}/correlation_matrix.pdf'))
        #     plt.close()
        # except ValueError:  # only one celltype, can't make these plots
        #     pass

        fig, umap_ax = plt.subplots(figsize=(6, 6))
        sc.pl.umap(adata, color="celltype", ax=umap_ax, show=False, legend_loc=None, title='', palette=self.color_dict)
        umap_ax.set_xticks([])
        umap_ax.set_yticks([])
        umap_ax.axis('off')
        plt.savefig(os.path.join(self.out_dir, f'celltype_plots{additional_out_dir}/umap.png'))
        plt.savefig(os.path.join(self.out_dir,f'celltype_plots{additional_out_dir}/umap.pdf'))
        plt.close()

        fig, tsne_ax = plt.subplots(figsize=(6, 6))
        sc.pl.tsne(adata, color="celltype", ax=tsne_ax, show=False, legend_loc=None, title='', palette=self.color_dict)
        tsne_ax.set_xticks([])
        tsne_ax.set_yticks([])
        tsne_ax.axis('off')
        plt.savefig(os.path.join(self.out_dir, f'celltype_plots{additional_out_dir}/tsne.png'))
        plt.savefig(os.path.join(self.out_dir, f'celltype_plots{additional_out_dir}/tsne.pdf'))
        plt.close()

        fig, pc_ax = plt.subplots(figsize=(6, 6))
        sc.pl.pca(adata, color="celltype", ax=pc_ax, show=False, legend_loc=None, title='', palette=self.color_dict)
        pc_ax.set_xticks([])
        pc_ax.set_yticks([])
        pc_ax.axis('off')
        pc_ax.set_title('')
        plt.savefig(os.path.join(self.out_dir, f'celltype_plots{additional_out_dir}/pca.png'))
        plt.savefig(os.path.join(self.out_dir, f'celltype_plots{additional_out_dir}/pca.pdf'))
        plt.close()

        # also compute UMAP and t-SNE without PC1
        adata_no_pc1 = adata.copy()
        adata_no_pc1.obsm['X_pca'] = adata_no_pc1.obsm['X_pca'][:, 1:]
        sc.pp.neighbors(adata_no_pc1, use_rep='X_pca')
        sc.tl.umap(adata_no_pc1)
        sc.tl.tsne(adata_no_pc1)

        fig, umap_ax = plt.subplots(figsize=(6, 6))
        sc.pl.umap(adata_no_pc1, color="celltype", ax=umap_ax, show=False, legend_loc=None, title='', palette=self.color_dict)
        umap_ax.set_xticks([])
        umap_ax.set_yticks([])
        umap_ax.axis('off')
        plt.savefig(os.path.join(self.out_dir, f'celltype_plots{additional_out_dir}/umap_no_pc1.png'))
        plt.savefig(os.path.join(self.out_dir,f'celltype_plots{additional_out_dir}/umap_no_pc1.pdf'))
        plt.close()

        fig, tsne_ax = plt.subplots(figsize=(6, 6))
        sc.pl.tsne(adata_no_pc1, color="celltype", ax=tsne_ax, show=False, legend_loc=None, title='', palette=self.color_dict)
        tsne_ax.set_xticks([])
        tsne_ax.set_yticks([])
        tsne_ax.axis('off')
        plt.savefig(os.path.join(self.out_dir, f'celltype_plots{additional_out_dir}/tsne_no_pc1.png'))
        plt.savefig(os.path.join(self.out_dir, f'celltype_plots{additional_out_dir}/tsne_no_pc1.pdf'))
        plt.close()

        


        # plot results of each clustering algorithm
        os.makedirs(os.path.join(self.out_dir, 'clustering'), exist_ok=True)
        if additional_out_dir != '':
            os.makedirs(os.path.join(self.out_dir, f'clustering{additional_out_dir}'), exist_ok=True)
        for cluster_alg in list(self.clustering_algs.keys()) + ['leiden', 'louvain']:
            try:
                fig, axs = plt.subplots(1, 3, figsize=(18, 6))
                sc.pl.pca(adata, color=cluster_alg, ax=axs[0], show=False, title="PCA Embeddings", legend_loc=None)
                sc.pl.tsne(adata, color=cluster_alg, ax=axs[1], show=False, title="t-SNE Embeddings", legend_loc=None)
                sc.pl.umap(adata, color=cluster_alg, ax=axs[2], show=False, title="UMAP Embeddings")
                fig.suptitle(f"{self.dataset_name} - {self.name}: {cluster_alg}")
                fig.savefig(os.path.join(self.out_dir, f'clustering{additional_out_dir}/embedding_{cluster_alg}.png'))
                fig.savefig(os.path.join(self.out_dir, f'clustering{additional_out_dir}/embedding_{cluster_alg}.pdf'))
                plt.close()

                fig, umap_ax = plt.subplots(figsize=(6, 6))
                sc.pl.umap(adata, color=cluster_alg, ax=umap_ax, show=False, legend_loc=None, title='')
                umap_ax.set_xticks([])
                umap_ax.set_yticks([])
                umap_ax.axis('off')
                plt.savefig(os.path.join(self.out_dir, f'clustering{additional_out_dir}/umap_{cluster_alg}.png'))
                plt.savefig(os.path.join(self.out_dir,f'clustering{additional_out_dir}/umap_{cluster_alg}.pdf'))
                plt.close()

                fig, tsne_ax = plt.subplots(figsize=(6, 6))
                sc.pl.tsne(adata, color=cluster_alg, ax=tsne_ax, show=False, legend_loc=None, title='')
                tsne_ax.set_xticks([])
                tsne_ax.set_yticks([])
                tsne_ax.axis('off')
                plt.savefig(os.path.join(self.out_dir, f'clustering{additional_out_dir}/tsne_{cluster_alg}.png'))
                plt.savefig(os.path.join(self.out_dir,f'clustering{additional_out_dir}/tsne_{cluster_alg}.pdf'))
                plt.close()

                fig, pc_ax = plt.subplots(figsize=(6, 6))
                sc.pl.pca(adata, color=cluster_alg, ax=pc_ax, show=False, legend_loc=None, title='')
                pc_ax.set_xticks([])
                pc_ax.set_yticks([])
                pc_ax.axis('off')
                pc_ax.set_title('')
                plt.savefig(os.path.join(self.out_dir, f'clustering{additional_out_dir}/pca_{cluster_alg}.png'))
                plt.savefig(os.path.join(self.out_dir,f'clustering{additional_out_dir}/pca_{cluster_alg}.pdf'))
                plt.close()
            except Exception as e:
                print(e)

        # plot visuaizations of each feature (e.g depth, batch, etc)
        os.makedirs(os.path.join(self.out_dir, 'other_feats'), exist_ok=True)
        if additional_out_dir != '':
            os.makedirs(os.path.join(self.out_dir, f'other_feats{additional_out_dir}'), exist_ok=True)
        for feat in self.features_dict.keys():
            # if len(adata.obs[feat].unique()) == 1:
            #     continue
            try:
                fig, axs = plt.subplots(1, 3, figsize=(18, 6))
                sc.pl.pca(adata, color=feat, ax=axs[0], show=False, title="PCA Embeddings", legend_loc=None)
                sc.pl.tsne(adata, color=feat, ax=axs[1], show=False, title="t-SNE Embeddings", legend_loc=None)
                sc.pl.umap(adata, color=feat, ax=axs[2], show=False, title="UMAP Embeddings")
                fig.suptitle(f"{self.dataset_name} - {self.name}: {feat}")
                fig.savefig(os.path.join(self.out_dir, f'other_feats{additional_out_dir}/embedding_{feat}.png'))
                fig.savefig(os.path.join(self.out_dir, f'other_feats{additional_out_dir}/embedding_{feat}.pdf'))
                plt.close()

                fig, umap_ax = plt.subplots(figsize=(6, 6))
                sc.pl.umap(adata, color=feat, ax=umap_ax, show=False, legend_loc=None, colorbar_loc=None, title='')
                umap_ax.set_xticks([])
                umap_ax.set_yticks([])
                umap_ax.axis('off')
                plt.savefig(os.path.join(self.out_dir, f'other_feats{additional_out_dir}/umap_{feat}.png'))
                plt.savefig(os.path.join(self.out_dir,f'other_feats{additional_out_dir}/umap_{feat}.pdf'))
                plt.close()

                fig, tsne_ax = plt.subplots(figsize=(6, 6))
                sc.pl.tsne(adata, color=feat, ax=tsne_ax, show=False, legend_loc=None, colorbar_loc=None, title='')
                tsne_ax.set_xticks([])
                tsne_ax.set_yticks([])
                tsne_ax.axis('off')
                plt.savefig(os.path.join(self.out_dir, f'other_feats{additional_out_dir}/tsne_{feat}.png'))
                plt.savefig(os.path.join(self.out_dir,f'other_feats{additional_out_dir}/tsne_{feat}.pdf'))
                plt.close()

                fig, pc_ax = plt.subplots(figsize=(6, 6))
                sc.pl.pca(adata, color=feat, ax=pc_ax, show=False, legend_loc=None, colorbar_loc=None, title='')
                pc_ax.set_xticks([])
                pc_ax.set_yticks([])
                pc_ax.axis('off')
                pc_ax.set_title('')
                plt.savefig(os.path.join(self.out_dir, f'other_feats{additional_out_dir}/pca_{feat}.png'))
                plt.savefig(os.path.join(self.out_dir,f'other_feats{additional_out_dir}/pca_{feat}.pdf'))
                plt.close()
            except Exception as e:
                console.print(f"[yellow]Error plotting {feat} embeddings: {e}[/]")

        if "rna_anndata" in self.other_args.keys() and "coembed" in self.other_args.keys():
            if self.other_args['coembed']:
                print('Loading RNAseq reference data...')
                self.rna = ad.read_h5ad(self.other_args['rna_anndata'])
                try:
                    self.rna.obs['celltype'] = self.rna.obs['cluster']
                except Exception as e:
                    print(e)
                try:
                    self.rna.obs['batch'] = self.rna.obs['individual']
                except Exception as e:
                    print(e)
                self.rna.layers["counts"] = self.rna.X.copy()
                print(adata)
                self.rna.obs['old_celltype'] = self.rna.obs['celltype']
                self.rna.obs['celltype'] = self.rna.obs['celltype'].apply(lambda s: s + '_rna')

                celltypes = sorted(adata.obs['celltype'].unique())
                rna_celltypes = sorted(self.rna.obs['celltype'].unique())
                n_clusters = max(len(celltypes), len(rna_celltypes))
                colors = list(plt.cm.tab20(np.int32(np.linspace(0, n_clusters + 0.99, n_clusters))))
                if self.data_generator.color_config is not None:
                    try:
                        with open(os.path.join('data/dataset_colors', self.data_generator.color_config), 'r') as f:
                            color_map = json.load(f)
                    except FileNotFoundError:  
                        pass 
                    color_map = color_map['colors']
                    for c in color_map.keys():
                        color_map[c] = np.array([x / 255.0 for x in color_map[c]])
                else:
                    color_map = {celltype: colors[i] for i, celltype in enumerate(celltypes)}
                rna_color_map = {celltype: colors[i] for i, celltype in enumerate(rna_celltypes)}
                color_map = {**color_map, **rna_color_map}
                color_map['Other'] = 'gray'
                print('Embedding RNA reference...')
                sc.pp.highly_variable_genes(self.rna, n_top_genes=2000, flavor="seurat_v3")
                #self.rna.var.loc[:, 'highly_variable'] = True
                #sc.pp.highly_variable_genes(self.rna, min_mean=0.0125, max_mean=3, min_disp=0.5)
                sc.pp.normalize_total(self.rna)
                sc.pp.log1p(self.rna)
                sc.pp.scale(self.rna)
                sc.tl.pca(self.rna, n_comps=100, svd_solver="auto")
                sc.pp.neighbors(self.rna, n_pcs=100, metric="cosine")
                sc.tl.umap(self.rna)

                fig = sc.pl.umap(self.rna, color=["celltype"], return_fig=True, wspace=0.6, palette=rna_color_map)
                fig.tight_layout()
                fig.savefig(f"{self.out_dir}/pfc_rna_reference_umap.png")
                plt.close()

                fig, ax = plt.subplots(figsize=(6, 6))
                sc.pl.umap(self.rna, color=['celltype'], ax=ax, show=False, legend_loc=None, title='', palette=rna_color_map)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.axis('off')
                fig.savefig(f"{self.out_dir}/pfc_rna_only_no_legend.png", dpi=400)
                plt.close()
                
                print('Coembedding...')
                self.rna.obs['dataset'] = 'RNA'
                adata.obs['dataset'] = 'scHi-C'
                adata_concat = self.rna.concatenate(adata)
                #sc.tl.pca(adata_concat)
                #sc.external.pp.bbknn(adata_concat, batch_key='dataset')
                #sc.external.pp.scanorama_integrate(adata_concat, 'dataset', knn=10)
                #sc.pp.neighbors(adata_concat, metric="cosine", use_rep='X_scanorama')
                sc.external.pp.harmony_integrate(adata_concat, 'dataset')
                sc.pp.neighbors(adata_concat, metric="cosine", use_rep='X_pca_harmony')
                sc.tl.umap(adata_concat)
                fig = sc.pl.umap(adata_concat, color=['celltype', 'dataset'], return_fig=True)
                fig.tight_layout()
                fig.savefig(f"{self.out_dir}/pfc_coembed.png")
                plt.close()

                fig = sc.pl.umap(adata_concat[adata_concat.obs['dataset'] == 'scHi-C'], color=['celltype', 'depth'], return_fig=True, palette=color_map)
                fig.tight_layout()
                fig.savefig(f"{self.out_dir}/pfc_coembed_hic_only.png")
                plt.close()

                fig = sc.pl.umap(adata_concat[adata_concat.obs['dataset'] == 'RNA'], color=['celltype'], return_fig=True, palette=rna_color_map)
                #fig.tight_layout()
                fig.savefig(f"{self.out_dir}/pfc_coembed_rna_only.png")
                plt.close()

                fig, ax = plt.subplots(figsize=(6, 6))
                sc.pl.umap(adata_concat[adata_concat.obs['dataset'] == 'RNA'], color=['celltype'], ax=ax, show=False, legend_loc=None, title='', palette=rna_color_map)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.axis('off')
                fig.savefig(f"{self.out_dir}/pfc_coembed_rna_only_no_legend.png", dpi=400)
                plt.close()


                fig = sc.pl.umap(adata_concat, color=["celltype"], palette=color_map, groups=celltypes, wspace=0.35, size=80, return_fig=True)
                fig.tight_layout()
                fig.savefig(f"{self.out_dir}/pfc_coembed_grayed_out.png")
                fig.savefig(f"{self.out_dir}/pfc_coembed_grayed_out.pdf")
                plt.close()

                fig, ax = plt.subplots(figsize=(6, 6))
                sc.pl.umap(adata_concat, color=['celltype'], palette=color_map, groups=celltypes, size=80, ax=ax, show=False, legend_loc=None, title='')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.axis('off')
                fig.savefig(f"{self.out_dir}/pfc_coembed_grayed_out_no_legend.png", dpi=400)
                plt.close()

                sc.tl.rank_genes_groups(adata, groupby='celltype', method='wilcoxon')
                fig = sc.pl.rank_genes_groups_dotplot(adata, n_genes=4, return_fig=True)
                fig.savefig(f"{self.out_dir}/pfc_scgad_scanpy_dotplot.png")
                plt.close()

                neurons = adata_concat[adata_concat.obs['celltype'].isin(['L2/3', 'L2/3_rna', 'L4', 'L4_rna', 'L5', 'L6', 'L5/6_rna', 'L5/6-CC_rna',
                                                                            'Pvalb', 'Sst', 'IN-PV_rna', 'IN-SST_rna', 'Ndnf', 'Vip', 'IN-SV2C_rna', 'IN-VIP_rna'])].copy()
                sc.pp.neighbors(neurons, metric="cosine", use_rep='X_pca_harmony')
                sc.tl.umap(neurons)

                fig = sc.pl.umap(neurons, color=['celltype', 'dataset'], return_fig=True)
                fig.tight_layout()
                fig.savefig(f"{self.out_dir}/pfc_coembed_neurons.png")
                plt.close()

                fig = sc.pl.umap(neurons[neurons.obs['dataset'] == 'scHi-C'], color=['celltype', 'depth'], return_fig=True, palette='tab20')
                fig.tight_layout()
                fig.savefig(f"{self.out_dir}/pfc_coembed_neurons_hic_only.png")
                plt.close()

                fig, ax = plt.subplots(figsize=(6, 6))
                sc.pl.umap(neurons[neurons.obs['dataset'] == 'RNA'], color=['celltype'], ax=ax, show=False, legend_loc=None, title='', palette=rna_color_map)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.axis('off')
                fig.savefig(f"{self.out_dir}/pfc_coembed_neurons_rna_only_no_legend.png", dpi=400)
                plt.close()

                fig = sc.pl.umap(neurons, color=["celltype"], palette=color_map, groups=celltypes, wspace=0.35, size=80, return_fig=True)
                fig.tight_layout()
                fig.savefig(f"{self.out_dir}/pfc_neurons_coembed_grayed_out.png")
                fig.savefig(f"{self.out_dir}/pfc_neurons_coembed_grayed_out.pdf")
                plt.close()

                fig, ax = plt.subplots(figsize=(6, 6))
                sc.pl.umap(neurons, color=['celltype'], palette=color_map, groups=celltypes, ax=ax, show=False, legend_loc=None, title='', size=80)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.axis('off')
                fig.savefig(f"{self.out_dir}/pfc_neurons_coembed_grayed_out_no_legend.png", dpi=400)
                plt.close()

                fig = sc.pl.rank_genes_groups_matrixplot(adata, n_genes=4, vmin=-1, vmax=1, cmap='bwr', return_fig=True)
                fig.savefig(f"{self.out_dir}/pfc_scgad_scanpy_matrixplot.png")
                plt.close()

        if log_wandb:
            import wandb
            plot_df = pd.DataFrame.from_dict({'UMAP_1': adata.obsm['X_umap'][:, 0], 'UMAP_2': adata.obsm['X_umap'][:, 1],
                                                't-SNE_1': adata.obsm['X_tsne'][:, 0], 't-SNE_2': adata.obsm['X_tsne'][:, 1],
                                                'PC_1': adata.obsm['X_pca'][:, 0], 'PC_2': adata.obsm['X_pca'][:, 1],
                                                'celltype': adata.obs['celltype']})
                                            
            for ax_source in ['UMAP', 't-SNE', 'PC']:
                fig = px.scatter(plot_df, x=f"{ax_source}_1", y=f"{ax_source}_2", color="celltype", title=f"{self.name} {self.resolution_name}")
                table = wandb.Table(columns = [f"plotly_figure_{ax_source}"])
                # Create path for Plotly figure
                path_to_plotly_html = os.path.join(self.out_dir, f"plotly_figure_{ax_source}.html")
                fig.write_html(path_to_plotly_html, auto_play = False) # Setting auto_play to False prevents animated Plotly charts from playing in the table automatically
                # Add Plotly figure as HTML file into Table
                table.add_data(wandb.Html(path_to_plotly_html))
                wandb.log({f"plotly_{ax_source}": table})

    def evaluate_and_plot(self, embedding, start_time, plot=False, iter='', cm=None, save=True, valid=True, log_wandb=False):
        import wandb
        import anndata as ad
        import scanpy as sc
        if valid:
            y = self.val_y 
        else:
            y = self.y
        matrix_reduce = None
        if cm is not None:
            self.cluster_cmap = cm
        if 'pca' in self.name or '3DVI' in self.name:  # input is already PCs
            pc_embeddings = embedding
            pc_embeddings_no_pc1 = pc_embeddings[:, 1:]
        else:
            pca = PCA(n_components=min(embedding.shape[1], embedding.shape[0]))  # get full PC decomposition
            while True:
                try:
                    pc_embeddings = pca.fit_transform(embedding)
                    break
                except np.linalg.LinAlgError:
                    print('SVD did not converge :(')
            pc_embeddings_no_pc1 = pc_embeddings[:, 1:]
        pred_labels = {}
        console.print(f"[yellow]Running clustering algorithms...[/]")
        for alg in self.clustering_algs.keys():
            try:
                if 'Unknown' in self.cluster_names:  # remove Unknown cells from clustering metrics
                    known_cells = y != self.cluster_names.index('Unknown')
                    n_cluster_offset = 1
                    print('Removing %d cells that are unknown from clustering metrics...' % np.sum(~known_cells))
                elif 'Other' in self.cluster_names:  # remove Unknown cells from clustering metrics
                    known_cells = y != self.cluster_names.index('Other')
                    n_cluster_offset = 1
                    print('Removing %d cells that are unknown from clustering metrics...' % np.sum(~known_cells))
                else:
                    n_cluster_offset = 0
                    known_cells = np.ones_like(y) == 1  # all cells
                if valid:
                    n_cluster_offset = self.n_clusters - len(self.val_dataset.reference['cluster'].unique())
                if alg == 'gmm':
                    gmm = self.clustering_algs[alg](n_components=self.n_clusters - n_cluster_offset, covariance_type='diag')
                    predicted_labels = gmm.fit_predict(embedding)
                    pc_gmm = self.clustering_algs[alg](n_components=self.n_clusters - n_cluster_offset)
                    pc_predicted_labels = pc_gmm.fit_predict(pc_embeddings_no_pc1)
                elif alg in ['dbscan', 'optics', 'affinity-propagation']:  # algorithms that infer number of clusters 
                    clustering_alg = self.clustering_algs[alg]()
                    pc_predicted_labels = clustering_alg.fit_predict(pc_embeddings_no_pc1)
                    predicted_labels = clustering_alg.fit_predict(embedding)
                else:
                    clustering_alg = self.clustering_algs[alg](n_clusters=self.n_clusters - n_cluster_offset)
                    pc_predicted_labels = clustering_alg.fit_predict(pc_embeddings_no_pc1)
                    predicted_labels = clustering_alg.fit_predict(embedding)
                pred_labels[alg] = predicted_labels
                try:
                    clustering_res = {}
                    for c in np.unique(predicted_labels):
                        c_cells = [self.data_generator.cell_list[i] for i in np.argwhere(predicted_labels == c).squeeze()]
                        clustering_res[str(c)] = c_cells
                    out_dict = os.path.join(self.out_dir, 'clusters_%s.json' % alg)
                    with open(out_dict, 'w') as f:
                        json.dump(clustering_res, f, indent=2)
                except Exception as e:  # only one unique label predicted
                    pass

                for metric_name in self.metric_algs.keys():
                    metric_alg_key = self.get_metric_alg_key(metric_name, alg)
                    if 'silhouette' in metric_name:
                        if metric_name == 'silhouette':
                            labels = predicted_labels[known_cells]
                            pc_labels = pc_predicted_labels[known_cells]
                        else:  # using ground truth labels to compute silhouette
                            labels = y[known_cells]
                            pc_labels = labels
                        try:
                            self.current_metrics[metric_alg_key] = self.metric_algs[metric_name](embedding[known_cells], labels)
                        except ValueError:
                            self.current_metrics[metric_alg_key] = 0
                        try:
                            self.current_metrics_no_pc1[metric_alg_key] = self.metric_algs[metric_name](pc_embeddings_no_pc1[known_cells], pc_labels)
                        except ValueError:
                            self.current_metrics_no_pc1[metric_alg_key] = 0
                    else:
                        self.current_metrics[metric_alg_key] = self.metric_algs[metric_name](y[known_cells], predicted_labels[known_cells])
                        self.current_metrics_no_pc1[metric_alg_key] = self.metric_algs[metric_name](y[known_cells], pc_predicted_labels[known_cells])
            except ValueError as e:
                print(e)
                for metric_name in self.metric_algs.keys():
                    metric_alg_key = self.get_metric_alg_key(metric_name, alg)
                    self.current_metrics[metric_alg_key] = 0
                    self.current_metrics_no_pc1[metric_alg_key] = 0
            finally:
                for metric_name in self.metric_algs.keys():
                    metric_alg_key = self.get_metric_alg_key(metric_name, alg)
                    self.metrics[metric_alg_key].append(self.current_metrics[metric_alg_key])
                    self.metrics_no_pc1[metric_alg_key].append(self.current_metrics_no_pc1[metric_alg_key])
        if 'eval_celltypes' in self.other_args.keys():  # compute celltype specific metrics
            if self.other_args['eval_celltypes'] is not None:
                keep_celltypes = self.other_args['eval_celltypes']
                known_celltypes_mask = np.zeros_like(y).astype(bool)
                for celltype in keep_celltypes:
                    known_celltypes_mask = known_celltypes_mask | (y == self.cluster_names.index(celltype))
                celltype_embedding = embedding[known_celltypes_mask]
                celltype_y = y[known_celltypes_mask]
                celltype_pc_embedding = pc_embeddings_no_pc1[known_celltypes_mask]
                celltype_pred_labels = {}
                for alg in self.clustering_algs.keys():
                    try:
                        if alg == 'gmm':
                            gmm = self.clustering_algs[alg](n_components=len(keep_celltypes), covariance_type='diag')
                            predicted_labels = gmm.fit_predict(celltype_embedding)
                            pc_gmm = self.clustering_algs[alg](n_components=len(keep_celltypes))
                            pc_predicted_labels = pc_gmm.fit_predict(celltype_pc_embedding)
                        elif alg in ['dbscan', 'optics', 'affinity-propagation']:  # algorithms that infer number of clusters 
                            clustering_alg = self.clustering_algs[alg]()
                            pc_predicted_labels = clustering_alg.fit_predict(celltype_pc_embedding)
                            predicted_labels = clustering_alg.fit_predict(celltype_embedding)
                        else:
                            clustering_alg = self.clustering_algs[alg](n_clusters=len(keep_celltypes))
                            pc_predicted_labels = clustering_alg.fit_predict(celltype_pc_embedding)
                            predicted_labels = clustering_alg.fit_predict(celltype_embedding)
                        celltype_pred_labels[alg] = predicted_labels
                        for metric_name in self.metric_algs.keys():
                            metric_alg_key = self.other_args['eval_name'] + '_' + self.get_metric_alg_key(metric_name, alg)
                            if metric_name == 'silhouette':
                                try:
                                    self.current_metrics[metric_alg_key] = self.metric_algs[metric_name](celltype_embedding, predicted_labels)
                                except ValueError:
                                    self.current_metrics[metric_alg_key] = 0
                                try:
                                    self.current_metrics_no_pc1[metric_alg_key] = self.metric_algs[metric_name](celltype_pc_embedding, pc_predicted_labels)
                                except ValueError:
                                    self.current_metrics_no_pc1[metric_alg_key] = 0
                            else:
                                self.current_metrics[metric_alg_key] = self.metric_algs[metric_name](celltype_y, predicted_labels)
                                self.current_metrics_no_pc1[metric_alg_key] = self.metric_algs[metric_name](celltype_y, pc_predicted_labels)
                    except ValueError as e:
                        print(e)
                        for metric_name in self.metric_algs.keys():
                            metric_alg_key = self.other_args['eval_name'] + '_' + self.get_metric_alg_key(metric_name, alg)
                            self.current_metrics[metric_alg_key] = 0
                            self.current_metrics_no_pc1[metric_alg_key] = 0
                    for metric_name in self.metric_algs.keys():
                        metric_alg_key = self.other_args['eval_name'] + '_' + self.get_metric_alg_key(metric_name, alg)
                        if metric_alg_key not in self.metrics.keys():
                            self.metrics[metric_alg_key] = []
                            self.metrics_no_pc1[metric_alg_key] = []
                        self.metrics[metric_alg_key].append(self.current_metrics[metric_alg_key])
                        self.metrics_no_pc1[metric_alg_key].append(self.current_metrics_no_pc1[metric_alg_key])


        # convert the embedding matrix into an annoted data object for more clustering and visualization
        adata = ad.AnnData(X=np.array(embedding))
        adata.var_names = [str(i) for i in range(embedding.shape[1])]
        adata.obs['cell'] = self.data_generator.reference.index
        adata.obs['celltype'] = np.array([self.cluster_names[i] for i in y])
        if self.color_dict is not None:
            adata.uns['celltype_colors'] = self.color_dict
        adata.obs['cluster'] = y
        for feat in self.features_dict.keys():  # add all user-provided metadata features
            adata.obs[feat] = self.features_dict[feat]
            if feat == 'batch':
                adata.obs['batch'] = adata.obs['batch'].astype('category')
            if feat == 'age':
                adata.obs['age'] = adata.obs['age'].fillna(0).astype(int)
        for cluster_alg in pred_labels.keys():  # add predicted labels from each clustering alg to adata object
            adata.obs[cluster_alg] = pd.Categorical(pred_labels[cluster_alg])
        sc.tl.pca(adata, n_comps=embedding.shape[1], svd_solver="auto")
        console.print(f"[yellow]Computing KNN graph...[/]")
        sc.pp.neighbors(adata)
        sc.tl.louvain(adata)
        sc.tl.leiden(adata)
        if '0' in iter and plot:
            console.print(f"[yellow]Computing UMAP embedding...[/]")
            sc.tl.umap(adata)
            console.print(f"[yellow]Computing t-SNE embedding...[/]")
            sc.tl.tsne(adata, random_state=36)
            adata.write_h5ad(os.path.join(self.out_dir, 'anndata_obj.h5ad'))
            
        for alg in ['louvain', 'leiden']:
            console.print(f"[yellow]Running {alg} clustering...[/]")
            predicted_labels = np.int32(adata.obs[alg])
            pc_predicted_labels = predicted_labels
            for metric_name in self.metric_algs.keys():
                metric_alg_key = self.get_metric_alg_key(metric_name, alg)
                if metric_name == 'silhouette':
                    try:
                        self.current_metrics[metric_alg_key] = self.metric_algs[metric_name](embedding[known_cells], predicted_labels[known_cells])
                    except ValueError:
                        self.current_metrics[metric_alg_key] = 0
                    try:
                        self.current_metrics_no_pc1[metric_alg_key] = self.metric_algs[metric_name](pc_embeddings_no_pc1[known_cells], pc_predicted_labels[known_cells])
                    except ValueError:
                        self.current_metrics_no_pc1[metric_alg_key] = 0
                else:
                    self.current_metrics[metric_alg_key] = self.metric_algs[metric_name](y[known_cells], predicted_labels[known_cells])
                    self.current_metrics_no_pc1[metric_alg_key] = self.metric_algs[metric_name](y[known_cells], pc_predicted_labels[known_cells])
                self.metrics[metric_alg_key].append(self.current_metrics[metric_alg_key])
                self.metrics_no_pc1[metric_alg_key].append(self.current_metrics_no_pc1[metric_alg_key])

        if 'eval_celltypes' in self.other_args.keys():  # compute celltype specific metrics
            console.print(f"[yellow]Running celltype specific clustering...[/]")
            if self.other_args['eval_celltypes'] is not None:
                eval_adata = adata[known_celltypes_mask].copy()
                for cluster_alg in celltype_pred_labels.keys():  # add predicted labels from each clustering alg to adata object
                    eval_adata.obs[cluster_alg] = pd.Categorical(celltype_pred_labels[cluster_alg])
                sc.pp.neighbors(eval_adata)
                sc.tl.louvain(eval_adata)
                sc.tl.leiden(eval_adata)
                if '0' in iter and plot:
                    sc.tl.umap(eval_adata)
                    sc.tl.tsne(eval_adata, random_state=36)
                    if self.color_dict is not None:
                        eval_adata.uns['celltype_colors'] = self.color_dict
                    try:
                        eval_adata.write_h5ad(os.path.join(self.out_dir, f'{self.other_args["eval_name"]}_anndata_obj.h5ad'))
                    except Exception as e:
                        print(e)
                for alg in ['louvain', 'leiden']:
                    predicted_labels = np.int32(eval_adata.obs[alg])
                    pc_predicted_labels = predicted_labels
                    for metric_name in self.metric_algs.keys():
                        metric_alg_key = self.other_args['eval_name'] + '_' + self.get_metric_alg_key(metric_name, alg)
                        if metric_name == 'silhouette':
                            self.current_metrics[metric_alg_key] = self.metric_algs[metric_name](celltype_embedding, predicted_labels)
                            self.current_metrics_no_pc1[metric_alg_key] = self.metric_algs[metric_name](celltype_pc_embedding, pc_predicted_labels)
                        else:
                            self.current_metrics[metric_alg_key] = self.metric_algs[metric_name](celltype_y, predicted_labels)
                            self.current_metrics_no_pc1[metric_alg_key] = self.metric_algs[metric_name](celltype_y, pc_predicted_labels)
                        if metric_alg_key not in self.metrics.keys():
                            self.metrics[metric_alg_key] = []
                            self.metrics_no_pc1[metric_alg_key] = []
                        self.metrics[metric_alg_key].append(self.current_metrics[metric_alg_key])
                        self.metrics_no_pc1[metric_alg_key].append(self.current_metrics_no_pc1[metric_alg_key])

        # now aggregate best performance across all clustering algs
        agg_cluster_algs = list(self.clustering_algs.keys()) + ['leiden', 'louvain']
        for metric_name in self.metric_algs.keys():
            best_key = f"best_{metric_name}"
            self.current_metrics[best_key] = np.max([self.current_metrics[self.get_metric_alg_key(metric_name, alg)] for alg in agg_cluster_algs])
            self.current_metrics_no_pc1[best_key] = np.max([self.current_metrics_no_pc1[self.get_metric_alg_key(metric_name, alg)] for alg in agg_cluster_algs])
            self.metrics[best_key].append(self.current_metrics[best_key])
            self.metrics_no_pc1[best_key].append(self.current_metrics_no_pc1[best_key])
        if 'eval_celltypes' in self.other_args.keys():  
            console.print(f"[yellow]Running celltype specific eval...[/]")
            if self.other_args['eval_celltypes'] is not None:
                for metric_name in self.metric_algs.keys():
                    best_key = f"best_{self.other_args['eval_name']}_{metric_name}"
                    self.current_metrics[best_key] = np.max([self.current_metrics[self.other_args['eval_name'] + '_' + self.get_metric_alg_key(metric_name, alg)] for alg in agg_cluster_algs])
                    self.current_metrics_no_pc1[best_key] = np.max([self.current_metrics_no_pc1[self.other_args['eval_name'] + '_' + self.get_metric_alg_key(metric_name, alg)] for alg in agg_cluster_algs])
                    if best_key not in self.metrics.keys():
                        self.metrics[best_key] = []
                        self.metrics_no_pc1[best_key] = []
                    self.metrics[best_key].append(self.current_metrics[best_key])
                    self.metrics_no_pc1[best_key].append(self.current_metrics_no_pc1[best_key])

        time_elapsed = time.time() - start_time 
        self.metrics['wall_time'].append(time_elapsed)
        saved_results = {'z': embedding, 'y': y, 'cell': np.array(self.data_generator.reference.index), 'depths': self.features_dict, 'wall_time': time_elapsed}
        if valid:
            out_dict = os.path.join(self.out_dir, 'val_embedding_%s.pickle' % iter)
        else:
            out_dict = os.path.join(self.out_dir, 'embedding_%s.pickle' % iter)
        if save:
            with open(out_dict, 'wb') as handle:
                pickle.dump(saved_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if plot:
            console.print(f"[green]Finished in: {time_elapsed} seconds[/]")
            if '0' in iter:
                console.print(f"[yellow]Saving plots...[/]")
                self.plot_from_adata(adata, iter, valid=valid, log_wandb=log_wandb)
                if 'eval_celltypes' in self.other_args.keys():
                    console.print(f"[yellow]Running celltype specific plotting...[/]")
                    if self.other_args['eval_celltypes'] is not None:
                        self.plot_from_adata(eval_adata, iter, valid=valid, additional_out_dir=f'/{self.other_args["eval_name"]}', log_wandb=log_wandb)
            console.print(f"[green]Finished plotting...[/]")
        if 'continuous' in self.other_args.keys():
            if self.other_args['continuous'] is not None:
                if self.other_args['continuous']:
                    # finally compute circular ROC for continuous datasets (TODO: better options for continuous datasets)
                    try:
                        self.acroc(pc_embeddings, metric_suffix='pca')
                        # if matrix_reduce is None:
                        #     reducer = umap.UMAP(transform_seed=36, random_state=36, metric='cosine')
                        #     matrix_reduce = reducer.fit_transform(embedding)
                        #self.acroc(matrix_reduce, metric_suffix='umap')
                    except Exception as e:
                        print(e)

    def save_metrics(self):
        with open(os.path.join(self.out_dir, '%s.pickle' % self.name), 'wb') as handle:
            pickle.dump(self.metrics, handle)
        with open(os.path.join(self.out_dir, '%s.json' % self.name), 'w') as f:
            json.dump(self.metrics, f, indent=2)
        with open(os.path.join(self.out_dir, '%s_no_pc1.pickle' % self.name), 'wb') as handle:
            pickle.dump(self.metrics_no_pc1, handle)
        with open(os.path.join(self.out_dir, '%s.no_pc1.json' % self.name), 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def load_metrics(self):
        with open(os.path.join(self.out_dir, '%s.pickle' % self.name), 'rb') as handle:
            self.metrics = pickle.load(handle)
        with open(os.path.join(self.out_dir, '%s_no_pc1.pickle' % self.name), 'rb') as handle:
            self.metrics_no_pc1 = pickle.load(handle)

    def load_embedding_from_file(self, emb_i=0):
        out_dict = os.path.join(self.out_dir, f'embedding__{emb_i}.pickle')
        try:
            with open(out_dict, 'rb') as handle:
                res = pickle.load(handle)
        except FileNotFoundError:  # no repeated experiments
            out_dict = os.path.join(self.out_dir, 'embedding__0.pickle')
            with open(out_dict, 'rb') as handle:
                res = pickle.load(handle)
        cellnames = np.array(res['cell'])
        y = []
        for cell in cellnames:
            cell = '.'.join(cell.split('.')[:-1]) + '.' + self.data_generator.res_name
            idx = self.cluster_names.index(self.data_generator.reference.loc[cell, 'cluster'])
            y.append(idx)
        self.y = np.array(y)
        self.features_dict['depth'] = np.array(res['depths']['depth'])
        return np.array(res['z'])

    def run(self, load=True, cm=None, outer_iter=0, start_time=None, valid=False, log_wandb=False, wandb_config=None):
        import wandb
        if log_wandb:
            exp_config = {'name': self.name,
                          'n_clusters': self.n_clusters,
                          'n_classes': self.n_classes,
                          'cluster_names': self.cluster_names,
                          'n_cells': self.n_cells,
                          'resolution': self.resolution,
                          'resolution_name': self.resolution_name
                          }
            if 'n_strata' in self.other_args:
                exp_config['distance'] = f"{int(self.resolution * float(self.other_args['n_strata']) / 1e6)}Mb"
                
            exp_config = {**exp_config, **self.other_args}
            while True:
                try:
                    wandb_run = wandb.init(project='SCORE', entity='dylan-plummer', reinit=True,
                                        config={**wandb_config, **exp_config})
                    break
                except Exception as e:
                    print(e)
                    print('Wandb error, retrying...')
                    time.sleep(60)
        if start_time is None:
            start_time = time.time()
        if load and '%s.pickle' % self.name in os.listdir(self.out_dir):
            if self.eval_inner:
                for i in range(self.n_experiments):
                    embedding = self.load_embedding_from_file(i)
                    self.evaluate_and_plot(embedding, start_time, plot=self.plot_viz if (i == 0) else False, iter='_%d' % i, cm=cm, valid=valid, save=False, log_wandb=log_wandb)
                    if log_wandb:
                        wandb.log({**self.current_metrics, **self.current_metrics_no_pc1, 'wall_time': self.metrics['wall_time'][-1]})
            else:
                embedding = self.load_embedding_from_file(outer_iter)
                self.evaluate_and_plot(embedding, start_time, plot=self.plot_viz if (outer_iter == 0) else False, iter='_%d' % outer_iter, cm=cm, valid=valid, save=False, log_wandb=log_wandb)
                if log_wandb:
                    wandb.log({**self.current_metrics, **self.current_metrics_no_pc1, 'wall_time': self.metrics['wall_time'][-1]})
        else:
            if self.eval_inner:
                for i in range(self.n_experiments):
                    if self.simulate and not self.append_simulated:  # update dataset to new simulated replicate
                        scool_file = f"scools/{self.data_generator.dataset_name}_rep{i}.scool"
                        self.data_generator.update_from_scool(scool_file)
                        self.features_dict['depth'] = self.data_generator.reference['depth'].values
                    embedding = self.get_embedding(iter_n=i)
                    self.evaluate_and_plot(embedding, start_time, plot=self.plot_viz if (i == 0) else False, iter='_%d' % i, cm=cm, valid=valid, log_wandb=log_wandb)
                    if log_wandb:
                        wandb.log({**self.current_metrics, **self.current_metrics_no_pc1, 'wall_time': self.metrics['wall_time'][-1]})
            else:
                if self.simulate and not self.append_simulated:  # update dataset to new simulated replicate
                    scool_file = f"scools/{self.data_generator.dataset_name}_rep{outer_iter}.scool"
                    self.data_generator.update_from_scool(scool_file)
                    self.features_dict['depth'] = self.data_generator.reference['depth'].values
                embedding = self.get_embedding(iter_n=outer_iter)
                self.evaluate_and_plot(embedding, start_time, plot=self.plot_viz if (outer_iter == 0) else False, iter='_%d' % outer_iter, cm=cm, valid=valid, log_wandb=log_wandb)
                if log_wandb:
                    wandb.log({**self.current_metrics, **self.current_metrics_no_pc1, 'wall_time': self.metrics['wall_time'][-1]})
            # remove nans from metrics
            for metric_alg in self.metrics.keys():
                try:
                    self.metrics[metric_alg] = [str(x) if not np.isnan(x) else str(0) for x in self.metrics[metric_alg]]
                except TypeError:
                    self.metrics[metric_alg] = [str(x) for x in self.metrics[metric_alg]]
                    pass
                try:
                    self.metrics_no_pc1[metric_alg] = [str(x) if not np.isnan(x) else str(0) for x in self.metrics_no_pc1[metric_alg]]
                except KeyError:
                    pass
                except TypeError:
                    self.metrics_no_pc1[metric_alg] = [str(x) for x in self.metrics_no_pc1[metric_alg]]
                    pass
            self.save_metrics()
            
        for metric_alg in self.metrics.keys():
            if 'ari' in metric_alg or 'acroc' in metric_alg or 'silhouette' in metric_alg:
                try:
                    console.print(f"[green]{metric_alg}[/]: [bold blue]{np.mean(np.float32(self.metrics[metric_alg])):.2f}[/] +/- [bold red]{np.std(np.float32(self.metrics[metric_alg])):.2f}[/]")
                except Exception as e:
                    print(e)
                    print(self.metrics[metric_alg])
                    pass
        if log_wandb and self.plot_viz:
            wandb.log({"umap": wandb.Image(os.path.join(self.out_dir, 'celltype_plots/umap.png'))})
            wandb.log({"tsne": wandb.Image(os.path.join(self.out_dir, 'celltype_plots/tsne.png'))})
            wandb.log({"pca": wandb.Image(os.path.join(self.out_dir, 'celltype_plots/pca.png'))})
            wandb.log({"embedding": wandb.Image(os.path.join(self.out_dir, 'celltype_plots/embedding.png'))})
            for feat in self.features_dict.keys():
                try:
                    wandb.log({f"umap_{feat}": wandb.Image(os.path.join(self.out_dir, f'other_feats/umap_{feat}.png'))})
                    wandb.log({f"pca_{feat}": wandb.Image(os.path.join(self.out_dir, f'other_feats/pca_{feat}.png'))})
                except Exception as e:
                    print(e)
            wandb.finish()