import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from score.experiments.experiment import Experiment


class TestExperiment(Experiment):
    def __init__(self, name, x, y, features, data_generator, n_strata, strata_k,
                 preprocessing=None, **kwargs):
        super().__init__(name, x, y, features, data_generator, **kwargs)
        self.n_strata = n_strata
        self.strata_k = strata_k
        self.n_components = self.latent_dim
        self.preprocessing = preprocessing

    def get_embedding(self, iter_n=0, remove_pc1=False):
        if self.preprocessing is None:  # using raw count data
            hic = ad.AnnData(self.x, dtype='int32')
        else:  # using whatever values are passed in (preprocessed data like normalized probs)
            hic = ad.AnnData(self.x)
        hic.obs_names = sorted(self.data_generator.cell_list)
        hic.obs_names = hic.obs_names.map(lambda s: s.replace(f'.{self.data_generator.res_name}', ''))

        matrix = hic.X  # matrix of shape (n_cells, n_features)
        # n_features is (# bins * # strata)
        
        # return an array of shape (n_cells, n_components)
        # ...

        # example:

        sc.tl.pca(hic, n_comps=self.n_components)
        if remove_pc1:
            return np.array(hic.obsm['X_pca'])[:, 1:]
        else:
            return np.array(hic.obsm['X_pca'])
