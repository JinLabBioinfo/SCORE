import numpy as np
import pandas as pd
import anndata as ad
from sklearn.decomposition import LatentDirichletAllocation
from score.experiments.experiment import Experiment


class LDAExperiment(Experiment):
    def __init__(self, name, x, y, features, data_generator, n_strata, preprocessing=None, n_components=64, **kwargs):
        super().__init__(name, x, y, features, data_generator, **kwargs)
        self.n_strata = n_strata
        self.preprocessing = preprocessing
        self.n_components = n_components

    def get_embedding(self, iter_n=0, remove_pc1=True):
        hic = ad.AnnData(self.x)
        hic.obs_names = sorted(self.data_generator.cell_list)
        hic.obs_names = hic.obs_names.map(lambda s: s.replace(f'.{self.data_generator.res_name}', ''))
        genomic_pos = self.data_generator.anchor_list.apply(lambda row: f"{row['chr']}:{row['start']}-{row['end']}", axis=1)
        new_var_names = pd.concat([genomic_pos.iloc[k:] + f'-{k}' for k in range(self.n_strata)])
        hic.var_names = new_var_names

        # LDA params from cisTopic paper: https://www.nature.com/articles/s41592-019-0367-1
        alpha = 50 / self.n_components
        beta = 0.1
        lda = LatentDirichletAllocation(n_components=10,
                                        doc_topic_prior=alpha, topic_word_prior=beta,
                                        n_jobs=-1,
                                        random_state=iter_n)
        if self.preprocessing is None:
            binary_mat = np.int32(hic.X > 0)
            zero_mask = np.sum(binary_mat, axis=0) > 0
            z = lda.fit_transform(hic.X[:, zero_mask])
        else:
            q = np.quantile(hic.X, q=0.8)
            binary_mat = np.int32(hic.X > q)
            zero_mask = np.sum(binary_mat, axis=0) > 0
            in_mat = hic.X[:, zero_mask]
            z = lda.fit_transform(in_mat)
        print('LDA perplexity:', lda.bound_)
        hic.obsm['X_lsi'] = z
        return np.array(hic.obsm['X_lsi'])
