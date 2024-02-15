import numpy as np
import anndata as ad
import scanpy as sc

from score.experiments.experiment import Experiment
from score.utils.matrix_ops import lsi


class LSIExperiment(Experiment):
    def __init__(self, name, x, y, features, data_generator, preprocessing=None, **kwargs):
        super().__init__(name, x, y, features, data_generator, **kwargs)
        self.n_components = self.latent_dim
        self.preprocessing = preprocessing

    def get_embedding(self, iter_n=0):
        if self.preprocessing is None:  # using raw count data
            hic = ad.AnnData(self.x, dtype='int32')
        else:  # using whatever values are passed in (preprocessed data like normalized probs)
            hic = ad.AnnData(self.x)
        hic.obs_names = sorted(self.data_generator.cell_list)
        hic.obs_names = hic.obs_names.map(lambda s: s.replace(f'.{self.data_generator.res_name}', ''))
        genomic_pos = self.data_generator.anchor_list.apply(lambda row: f"{row['chr']}:{row['start']}-{row['end']}", axis=1)
        hic.var_names = genomic_pos
        if self.preprocessing is None:
            sc.pp.filter_genes(hic, min_counts=1)
            sc.pp.filter_genes(hic, min_cells=1)
        lsi(hic, n_components=min(self.n_components, hic.shape[1] - 1, hic.shape[0] - 1), n_iter=50, n_oversamples=20)
        return np.array(hic.obsm['X_lsi'])
