import os
import _pickle as cPickle
import argparse
import numpy as np

from sklearn.decomposition import PCA
from score.experiments.experiment import Experiment



class EnsembleExperiment(Experiment):
    def __init__(self, name, x, y, depths, data_generator, **kwargs):
        super().__init__(name, x, y, depths, data_generator, **kwargs)

        self.res_dir = os.path.join(self.save_dir, data_generator.dataset_name, self.resolution_name)

    def get_embedding(self, iter_n=0, n_components=256):
        z = []
        for exp_name in os.listdir(self.res_dir):
            if exp_name == 'ensemble':
                continue
            print(exp_name)
            exp_dir = os.path.join(self.res_dir, exp_name)
            res_dict = os.path.join(exp_dir, 'embedding__0.pickle')
            with open(res_dict, 'rb') as handle:
                res = cPickle.load(handle)
                emb = res['z']
                z.append(emb)
        z = np.concatenate(z, axis=1)
        print(z.shape)
        z_pca = PCA(min(n_components, z.shape[1], z.shape[0])).fit_transform(z)
        return z_pca
