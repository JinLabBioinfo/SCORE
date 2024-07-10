import numpy as np
import anndata as ad
from score.experiments.experiment import Experiment
from score.utils.matrix_ops import idf_inner_product


class IDF2DExperiment(Experiment):
    def __init__(self, name, x, y, features, data_generator, n_strata, preprocessing=None, **kwargs):
        super().__init__(name, x, y, features, data_generator, **kwargs)
        self.n_strata = n_strata
        self.n_components = self.latent_dim
        self.preprocessing = preprocessing

    def get_embedding(self, iter_n=0, remove_pc1=True):
        if self.preprocessing is None:  # using raw count data
            hic = ad.AnnData(self.x, dtype='int32')
        else:  # using whatever values are passed in (preprocessed data like normalized probs)
            hic = ad.AnnData(self.x)
        hic.obs_names = sorted(self.data_generator.cell_list)
        hic.obs_names = hic.obs_names.map(lambda s: s.replace(f'.{self.data_generator.res_name}', ''))
        idf_inner_product(hic)
        return np.array(hic.obsm['X_mds'])
