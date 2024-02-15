import os
import sys
import pickle
import argparse
import traceback
import numpy as np
from sklearn.decomposition import PCA

from score.experiments.experiment import Experiment



class HigashiExperiment(Experiment):
    def __init__(self, name, x, y, depths, data_generator, **kwargs):
        super().__init__(name, x, y, depths, data_generator, **kwargs)
        self.emb_dir = 'data/higashi_data/%s_%s/tmp/embed' % (data_generator.dataset_name, data_generator.res_name)
        self.label_file = 'data/higashi_data/%s_%s/label_info.pickle' % (data_generator.dataset_name, data_generator.res_name)
        self.ref = pickle.load(open(self.label_file, "rb"))
        if 'pfc_downsample' in data_generator.dataset_name:
            classes = np.array(['L2/3', 'L4', 'L5', 'L6', 'Ndnf', 'Vip', 'Pvalb', 'Sst', 'Astro', 'ODC', 'OPC', 'MG', 'MP',
                             'Endo'])
        else:
            classes = data_generator.classes
        celltypes = self.ref['cell type']
        cellnames = self.ref['cell name']
        self.y = []
        # for l in celltypes:
        #     self.y.append(np.argwhere(classes == l)[0])
        for cell in cellnames:
            l = self.data_generator.reference.loc[cell, 'cluster']
            self.y.append(np.argwhere(classes == l)[0])
        self.y = np.squeeze(np.array(self.y))
        print(self.y)
        self.batches = np.array(self.ref['batch'])

    def get_embedding(self, iter_n=0):
        if 'scghost' in self.name:
            full_z = np.load(os.path.join(self.emb_dir.replace('embed', 'scghost'), 'pcs.npy'))
            print(full_z.shape)
            size = self.latent_dim
            pca = PCA(n_components=size)
            z = pca.fit_transform(full_z)
        else:
            z = []
            for f in os.listdir(self.emb_dir):
                z_tmp = np.load(os.path.join(self.emb_dir, f))
                if z_tmp.shape[0] == self.data_generator.n_cells:
                    z = z_tmp
            pca = PCA(n_components=self.latent_dim)
            z = pca.fit_transform(z)
        return z
