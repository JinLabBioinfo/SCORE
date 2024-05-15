import os
import time
from score.experiments.experiment import Experiment


class ScHiCToolsExperiment(Experiment):
    def __init__(self, name, y, depths, data_generator, method='innerproduct', embedding_method='MDS', operations=None, n_components=16,
                 data_dir='schictools_data', rewrite=False, resolution_name='200kb', n_strata=50,
                 strata_offset=0, load=False, **kwargs):
        super().__init__(name, None, y, depths, data_generator, **kwargs)
        self.n_components = n_components
        self.dataset_dir = os.path.join(data_dir, data_generator.dataset_name)
        self.data_dir = os.path.join(data_dir, data_generator.dataset_name, resolution_name)
        self.rewrite = rewrite
        self.dataset_name = data_generator.dataset_name
        os.makedirs(os.path.join(data_dir, data_generator.dataset_name), exist_ok=True)
        self.network = None
        self.val_network = None
        if not self.simulate and not load:
            if self.val_dataset is not None:
                self.val_network = self.prepare_schictools_data(self.val_dataset)
            self.network = self.prepare_schictools_data(self.data_generator)

        self.n_strata = n_strata
        self.strata_offset = strata_offset
        self.operations = operations
        self.loaded_data = self.schictools_data
        self.val_loaded_data = self.val_schictools_data
        self.method = method
        self.embedding_method = embedding_method
        if 'viz_innerproduct' in self.other_args.keys():
            self.viz_dist_hist = True

    def get_embedding(self, iter_n=0):
        if 'schicluster' in self.name.lower():
            full_maps = True 
        else:
            full_maps = False
        if self.simulate:
            rep_data_dir = os.path.join(self.data_dir, f'rep{iter_n}')
            self.network, self.chr_lengths = self.data_generator.write_scHiCTools_matrices(out_dir=rep_data_dir, resolution=self.resolution, rewrite=True)
            self.loaded_data = self.load_schictools_data(self.network, full_maps=full_maps)
        elif iter_n == 0 and self.loaded_data is None:
            if self.val_dataset is not None:
                self.loaded_data = self.load_schictools_data(self.val_network + self.network, full_maps=full_maps)
                #self.val_loaded_data = self.load_schictools_data(self.val_network)
            else:
                self.loaded_data = self.load_schictools_data(self.network, full_maps=full_maps)

        if self.viz_dist_hist:
            import anndata as ad
            import scanpy as sc
            import numpy as np
            import pandas as pd
            import seaborn as sns
            import matplotlib.pyplot as plt
            from matplotlib.colors import PowerNorm
            from tqdm import tqdm
            from scipy.stats import zscore, norm
            from scipy.stats import differential_entropy, entropy
            mat = []
            contacts = []
            for k in tqdm(range(self.n_strata)):
                strata_mat = []
                strata_contacts = []
                for ch in self.loaded_data.chromosomes:
                    z = zscore(self.loaded_data.strata[ch][k], axis=1)
                    z[np.isnan(z)] = 0
                    inner = z.dot(z.T) / z.shape[1]
                    #inner[inner > 1] = 1
                    #inner[inner < -1] = -1
                    #distance_mat = np.sqrt(2 - 2 * inner)
                    np.fill_diagonal(inner, 0)
                    strata_mat.append(inner)
                    strata_contacts.append(self.loaded_data.strata[ch][k])
                median_chr_mat = np.median(strata_mat, axis=0)  # chromosome median
                strata_contacts = np.concatenate(strata_contacts, axis=1)
                print(strata_contacts.shape)
                mat.append(median_chr_mat)  # chromosome median
                contact_density = np.sum(strata_contacts, axis=1)
                contacts.append(contact_density)
            #mat = np.median(mat, axis=1)  # cell median
            mat = np.array(mat)
            # set row colors to cell types 
            lut = dict(zip(self.cluster_names, sns.color_palette("tab10", n_colors=len(self.cluster_names))))
            row_colors = pd.Series([self.cluster_names[i] for i in self.y]).map(lut)
            sns.clustermap(np.median(mat, axis=0), cmap='viridis', figsize=(10, 10), row_cluster=True, col_cluster=True, xticklabels=False, yticklabels=False)
            plt.savefig(os.path.join(self.out_dir, 'distance.png'))
            #plt.savefig(os.path.join(self.out_dir, 'distance.pdf'))
            plt.close()
            # convert to probabilities
            #mat = np.log1p(mat)
            mat = (mat - np.min(mat)) / (np.max(mat) - np.min(mat))
            #mat = np.mean(mat, axis=1)
            mat = np.nan_to_num(mat, posinf=0)
            mat = entropy(mat + 1e-10, axis=1, base=2)
            mat = np.nan_to_num(mat, posinf=0)
            print(mat)
            mat = np.transpose(np.array(mat))
            print(mat.shape)
            adata = ad.AnnData(mat)
            adata.obs['cell_type'] = np.array([self.cluster_names[i] for i in self.y])
            # make cell_type categorical sorted by the order of cluster_names
            #adata.obs['cell_type'] = pd.Categorical(adata.obs['cell_type'], categories=sorted(self.cluster_names))
            adata.var_names = [f'{(i * self.data_generator.resolution) / 1e6:.1f}M' for i in range(self.n_strata)]
            sc.pl.heatmap(adata, adata.var_names, groupby='cell_type', swap_axes=False, cmap='Reds_r', show=False, dendrogram=False, figsize=(5, 10))
            plt.tight_layout()
            plt.savefig(os.path.join(self.out_dir, 'distance_hist.png'))
            plt.savefig(os.path.join(self.out_dir, 'distance_hist.pdf'))
            plt.close()

            sc.pl.matrixplot(adata, adata.var_names, groupby='cell_type', cmap='Reds', standard_scale='var')
            plt.tight_layout()
            plt.savefig(os.path.join(self.out_dir, 'distance_matrix.png'))
            plt.savefig(os.path.join(self.out_dir, 'distance_matrix.pdf'))
            plt.close()

            # plot contacts
            contacts = np.array(contacts)
            contacts = np.transpose(contacts)
            total_contacts = np.sum(contacts, axis=1)
            contacts = contacts / total_contacts[:, None]
            adata = ad.AnnData(contacts)
            adata.obs['cell_type'] = np.array([self.cluster_names[i] for i in self.y])
            adata.var_names = [f'{(i * self.data_generator.resolution) / 1e6:.1f}M' for i in range(self.n_strata)]
            sc.pl.heatmap(adata, adata.var_names, groupby='cell_type', swap_axes=False, cmap='Reds', standard_scale='var', show=False, dendrogram=False, figsize=(5, 10))
            plt.tight_layout()
            plt.savefig(os.path.join(self.out_dir, 'contacts_hist.png'))
            plt.savefig(os.path.join(self.out_dir, 'contacts_hist.pdf'))
            plt.close()


        start_time = time.time()
        dim = self.latent_dim
        emb, _ = self.loaded_data.learn_embedding(dim=dim, similarity_method=self.method,
                           embedding_method=self.embedding_method,
                           aggregation='median',
                           print_time=False,
                           return_distance=True,
                           n_strata=self.n_strata)
        

        if self.val_dataset is not None:
            emb = emb[:len(self.val_network)]
        return emb
