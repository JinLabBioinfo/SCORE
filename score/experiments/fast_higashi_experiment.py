import os
import sys
import pickle
import argparse
import traceback
import numpy as np

from score.experiments.experiment import Experiment



class FastHigashiExperiment(Experiment):
    def __init__(self, name, x, y, depths, data_generator, depth_norm=False, **kwargs):
        super().__init__(name, x, y, depths, data_generator, **kwargs)
        self.depth_norm = depth_norm
        self.emb_dir = 'data/higashi_data/%s_%s/tmp/embed' % (data_generator.dataset_name, data_generator.res_name)
        self.label_file = 'data/higashi_data/%s_%s/label_info.pickle' % (data_generator.dataset_name, data_generator.res_name)
        self.ref = pickle.load(open(self.label_file, "rb"))
        if 'pfc_downsample' in data_generator.dataset_name:
            classes = np.array(['L2/3', 'L4', 'L5', 'L6', 'Ndnf', 'Vip', 'Pvalb', 'Sst', 'Astro', 'ODC', 'OPC', 'MG', 'MP',
                             'Endo'])
        else:
            classes = data_generator.classes
        print(self.ref)
        cellnames = self.ref['cell name']
        celltypes = self.ref['cell type']
        self.y = []
        # for l in celltypes:
        #     self.y.append(np.argwhere(classes == l)[0])
        for cell in cellnames:
            l = self.data_generator.reference.loc[cell, 'cluster']
            self.y.append(np.argwhere(classes == l)[0])
        self.y = np.squeeze(np.array(self.y))
        print(self.y)
        self.batches = np.array(self.ref['batch'])
        self.model = None  # model needs to be trained before getting embedding

    def get_embedding(self, iter_n=0):
        z = self.model.fetch_cell_embedding(final_dim=min(256, self.data_generator.n_cells), restore_order=True)
        print(z)
        print(z['embed_l2_norm'].shape)
        if self.depth_norm:
            return z['embed_l2_norm_correct_coverage_fh']
        else:
            return z['embed_l2_norm']
