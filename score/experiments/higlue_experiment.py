import anndata as ad
import scanpy as sc
import scglue
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import mode
from sklearn.decomposition import PCA
from score.experiments.experiment import Experiment


class HiGLUEExperiment(Experiment):
    def __init__(self, name, x, y, features, data_generator, 
                 glue_model, prior_graph, rna, hic, n_neighbors=15,
                 **kwargs):
        super().__init__(name, x, y, features, data_generator, **kwargs)
        self.n_components = self.latent_dim
        self.glue = scglue.models.load_model(glue_model)
        self.prior = nx.read_graphml(prior_graph)
        self.rna = ad.read_h5ad(rna)
        self.hic = ad.read_h5ad(hic)
        self.n_neighbors = n_neighbors

        print(self.rna)
        # hic.obs.loc[hic.obs_names.str.startswith('alpha_'), 'celltype'] = 'Alpha'
        # hic.obs.loc[hic.obs_names.str.startswith('beta_'), 'celltype'] = 'Beta'
        # rna = rna[rna.obs['celltype'] != 'Epsilon'].copy()
        # rna = rna[rna.obs['celltype'] != 'Delta'].copy()
        # rna = rna[rna.obs['celltype'] != 'PP'].copy()
        self.rna.var["highly_variable"] = self.rna.var["highly_variable"] & self.rna.var["in_hic"]
        self.hic.var["highly_variable"] = True
        self.hic.obs.loc[self.hic.obs_names.str.startswith('alpha_'), 'celltype'] = 'Alpha'
        self.hic.obs.loc[self.hic.obs_names.str.startswith('beta_'), 'celltype'] = 'Beta'
        self.hic = self.hic[self.hic.obs['depth'] > data_generator.min_depth, :]
        print(self.hic)

        self.celltypes = self.rna.obs['celltype'].unique()
        rna_celltypes = self.rna.obs['celltype'].unique()
        print(rna_celltypes)
        self.rna.obs['celltype'] = pd.Categorical(self.rna.obs['celltype'], categories=rna_celltypes, ordered=True)
        self.hic.obs['celltype'] = pd.Categorical(self.hic.obs['celltype'], categories=self.celltypes, ordered=True)
        # set depth as fraction of total counts
        self.rna.obs['depth'] = self.rna.layers['counts'].sum(axis=1)
        self.rna.obs['depth'] = self.rna.obs['depth'] / self.rna.obs['depth'].max()

        scglue.models.configure_dataset(self.rna, "NB", use_highly_variable=True, use_layer="counts", use_depth="depth", use_rep="X_pca")
        scglue.models.configure_dataset(self.hic, "HiCZINB", use_highly_variable=True, use_layer="counts", use_depth="depth", use_batch="batch")


    def get_embedding(self, iter_n=0, remove_pc1=True):
        self.rna.obsm["X_glue"] = self.glue.encode_data("rna", self.rna)
        self.hic.obsm["X_glue"] = self.glue.encode_data("hic", self.hic)
        self.hic.obs['domain'] = 'hic'
        self.rna.obs['domain'] = 'rna'
        self.hic.obs['dataset'] = 'train'
        if 'old_celltype' not in self.hic.obs.columns:
            self.hic.obs['old_celltype'] = self.hic.obs['celltype']

        self.rna.obs['celltype'] = self.rna.obs['celltype'].apply(lambda s: s.replace('_rna', ''))
        self.rna.obs['old_celltype'] = self.rna.obs['celltype']
        rna_valid = self.rna

        print('Label transfer...')
        scglue.data.transfer_labels(rna_valid, self.hic, "celltype", use_rep="X_glue", n_neighbors=self.n_neighbors)
        combined = ad.concat([self.rna, self.hic])
        sc.pp.neighbors(combined, use_rep="X_glue", metric="cosine", n_neighbors=self.n_neighbors)
        sc.tl.umap(combined)
        print(self.hic.obs['celltype'].value_counts())
        sc.tl.louvain(combined)
        self.hic.obs['louvain'] = combined.obs.loc[self.hic.obs.index, 'louvain']
        self.rna.obs['louvain'] = combined.obs.loc[self.rna.obs.index, 'louvain']
        cluster_celltype_map = {}
        celltype_int_map = {c:i for i, c in enumerate(sorted(combined.obs['celltype'].unique()))}
        print(celltype_int_map)
        for cluster_i in sorted(combined.obs['louvain'].unique()):
            celltypes_in_cluster = np.int32(combined.obs.loc[combined.obs['louvain'] == cluster_i, 'celltype'].map(celltype_int_map).values)
            print(celltypes_in_cluster)
            mode_celltype_int = mode(celltypes_in_cluster, axis=None)[0]
            print(mode_celltype_int)
            mode_celltype = [k for k, v in celltype_int_map.items() if v == mode_celltype_int][0]
            print(mode_celltype)
            cluster_celltype_map[cluster_i] = mode_celltype
        print(cluster_celltype_map)
        self.hic.obs['celltype'] = self.hic.obs['louvain'].map(cluster_celltype_map)
        print(self.hic.obs['celltype'].value_counts())
        combined.obs['celltype'] = combined.obs['louvain'].map(cluster_celltype_map)

        fig = sc.pl.umap(combined[combined.obs["domain"] == "hic"], color=["old_celltype", "celltype", "depth", "batch"], return_fig=True)
        fig.savefig(f'{self.out_dir}/umap_{iter_n}.png')
        plt.close(fig)

        fig = sc.pl.umap(combined[combined.obs["domain"] == "rna"], color=["old_celltype"], return_fig=True)
        fig.savefig(f'{self.out_dir}/umap_rna_{iter_n}.png')
        plt.close(fig)

        rna_mask = combined.obs['domain'] == 'rna'
        combined.obs['celltype'] = combined.obs['celltype'].astype(str)
        combined.obs.loc[rna_mask, 'celltype'] = combined.obs.loc[rna_mask, 'old_celltype'].apply(lambda s: s.replace('_rna', ''))
        combined.obs.loc[rna_mask, 'celltype'] += '_rna'
        fig = sc.pl.umap(combined, color=["celltype"], groups=self.celltypes, size=100, wspace=0.45, hspace=0.45, return_fig=True)
        fig.savefig(f'{self.out_dir}/umap_combined_{iter_n}.png')
        plt.close(fig)

        # take PCA of the combined embedding
        pca = PCA(n_components=min(self.n_components, combined.shape[0], combined.obsm["X_glue"].shape[1]))
        combined.obsm["X_pca"] = pca.fit_transform(combined.obsm["X_glue"])
        hic_only = combined[combined.obs["domain"] == "hic"]
        return hic_only.obsm["X_pca"]
