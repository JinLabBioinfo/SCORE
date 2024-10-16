import os
import cooler
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import snapatac2 as snap
from scipy.sparse import csr_matrix
from score.experiments.experiment import Experiment


class SnapATACExperiment(Experiment):
    def __init__(self, name, x, y, features, data_generator, n_strata, preprocessing=None, **kwargs):
        super().__init__(name, x, y, features, data_generator, **kwargs)
        self.n_strata = n_strata
        self.n_components = self.latent_dim
        self.preprocessing = preprocessing

    def get_embedding(self, iter_n=0, remove_pc1=True):
        hic = ad.AnnData(csr_matrix(np.uint32(self.x)), obs=pd.DataFrame(index=sorted(self.data_generator.cell_list)))
        print(hic)
        #hic.obs_names = sorted(self.data_generator.cell_list)
        hic.obs_names = hic.obs_names.map(lambda s: s.replace(f'.{self.data_generator.res_name}', ''))
        genomic_pos = self.data_generator.anchor_list.apply(lambda row: f"{row['chr']}:{row['start']}-{row['end']}", axis=1)
        # new_var_names = pd.concat([genomic_pos.iloc[k:] + f'-{k}' for k in range(self.n_strata)])
        # hic.var_names = new_var_names
        # if self.preprocessing is None:
        #     sc.pp.filter_genes(hic, min_counts=1)
        #     sc.pp.filter_genes(hic, min_cells=1)
        hic.obsm['fragment_paired'] = csr_matrix(np.uint32(self.x))
        content_of_scool = cooler.fileops.list_coolers(self.data_generator.scool_file)
        c = cooler.Cooler(f"{self.data_generator.scool_file}::/{content_of_scool[0]}")
        chr_sizes = c.chromsizes
        reference_sequences = pd.DataFrame({'chr': chr_sizes.index, 'size': chr_sizes.values})
        reference_sequences.rename(columns={'chr': 'reference_seq_name', 'size': 'reference_seq_length'}, inplace=True)
        reference_sequences['reference_seq_length'] = reference_sequences['reference_seq_length'].astype(pd.UInt64Dtype())
        hic.uns['reference_sequences'] = reference_sequences
        # reset var and var_names
        # hic.X = None
        # hic.var = pd.DataFrame()
        # hic.var_names = pd.Index([])

        #snap.pp.add_tile_matrix(hic, counting_strategy='fragment', bin_size=10)
        #hic.X = 
        print(hic)
        snap.pp.select_features(hic, n_features=min(hic.shape[1], 500000))
        snap.tl.spectral(hic, n_comps=self.latent_dim)

        # import matplotlib.pyplot as plt
        # query = snap.pp.make_gene_matrix(hic, gene_anno=snap.genome.hg19, counting_strategy='fragment')
        # query.obs['celltype'] = np.array([self.cluster_names[i] for i in self.y])
        # print(query)
        # sc.pp.filter_genes(query, min_cells=2)
        # sc.pp.highly_variable_genes(
        #     query,
        #     n_top_genes = 10000,
        #     flavor="seurat_v3",
        #     #batch_key="batch",
        #     span=1
        # )
        # sc.pl.highly_variable_genes(query)
        # plt.savefig(os.path.join(self.out_dir, f'celltype_plots/highly_variable_genes.png'))
        # plt.close()

        # sc.pp.scale(query)
        # sc.pp.pca(query)
        # sc.pp.neighbors(query)
        # sc.tl.umap(query)
        # fig, umap_ax = plt.subplots(figsize=(6, 6))
        # sc.pl.umap(query, color="celltype", ax=umap_ax, show=False, legend_loc=None, title='', palette=self.color_dict)
        # plt.savefig(os.path.join(self.out_dir, f'celltype_plots/umap_gene_transfer.png'))
        # plt.close()

        # try:
        #     sc.pl.umap(query, color=['INS', 'GCG'])
        #     plt.savefig(os.path.join(self.out_dir, f'celltype_plots/umap_gene_transfer_ins_gcg.png'))
        #     plt.close()
        # except:
        #     pass

        return np.array(hic.obsm['X_spectral'])
