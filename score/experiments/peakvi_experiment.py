import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from score.experiments.experiment import Experiment


class PeakVIExperiment(Experiment):
    def __init__(self, name, x, y, features, data_generator, n_strata, n_components=64, **kwargs):
        super().__init__(name, x, y, features, data_generator, **kwargs)
        self.n_strata = n_strata
        self.n_components = n_components

    def get_embedding(self, iter_n=0, remove_pc1=True):
        import scvi
        hic = ad.AnnData(self.x)
        hic.obs_names = sorted(self.data_generator.cell_list)
        hic.obs_names = hic.obs_names.map(lambda s: s.replace(f'.{self.data_generator.res_name}', ''))
        genomic_pos = self.data_generator.anchor_list.apply(lambda row: f"{row['chr']}:{row['start']}-{row['end']}", axis=1)
        new_var_names = pd.concat([genomic_pos.iloc[k:] + f'-{k}' for k in range(self.n_strata)])
        hic.var_names = new_var_names

        sc.pp.filter_genes(hic, min_counts=1)
        # compute the threshold: 5% of the cells
        min_cells = int(hic.shape[0] * 0.05)
        sc.pp.filter_genes(hic, min_cells=min_cells)
        scvi.model.PEAKVI.setup_anndata(hic)
        pvi = scvi.model.PEAKVI(hic)
        pvi.train()
        latent = pvi.get_latent_representation()
        return np.array(latent)
