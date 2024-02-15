import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from score.experiments.experiment import Experiment


class ScVIExperiment(Experiment):
    def __init__(self, name, x, y, features, data_generator, n_strata, preprocessing=None, **kwargs):
        super().__init__(name, x, y, features, data_generator, **kwargs)
        self.n_strata = n_strata
        self.preprocessing = preprocessing
        self.n_components = self.latent_dim

    def get_embedding(self, iter_n=0, remove_pc1=True):
        import scvi
        hic = ad.AnnData(self.x)
        hic.obs_names = sorted(self.data_generator.cell_list)
        hic.obs_names = hic.obs_names.map(lambda s: s.replace(f'.{self.data_generator.res_name}', ''))
        genomic_pos = self.data_generator.anchor_list.apply(lambda row: f"{row['chr']}:{row['start']}-{row['end']}", axis=1)
        new_var_names = pd.concat([genomic_pos.iloc[k:] + f'-{k}' for k in range(self.n_strata)])
        hic.var_names = new_var_names
        hic.obs['batch'] = self.features_dict['batch']
        n_top_genes = int(0.1 * hic.shape[1])
        print(n_top_genes)
        sc.pp.highly_variable_genes(
            hic,
            flavor="seurat_v3",
            n_top_genes=n_top_genes,
            batch_key="batch",
            subset=True
        )
        scvi.model.SCVI.setup_anndata(hic, batch_key="batch")
        vae = scvi.model.SCVI(hic, n_layers=2, n_latent=self.n_components, gene_likelihood="nb")
        vae.train()
        latent = vae.get_latent_representation()
        return np.array(latent)
