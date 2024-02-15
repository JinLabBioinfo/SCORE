import os
import sys
import pickle
import argparse
import traceback
import numpy as np
import pandas as pd

from score.sc_args import parse_args
from score.experiments.experiment import Experiment



class DVIExperiment(Experiment):
    def __init__(self, name, x, y, depths, data_generator, load_results=False, **kwargs):
        super().__init__(name, x, y, depths, data_generator, **kwargs)
        self.emb_dir = 'threeDVI/results/%s_%s/latentEmbeddings/' % (data_generator.dataset_name, data_generator.res_name)
        self.label_file = 'threeDVI/reference/%s_%s/data_summary.txt' % (data_generator.dataset_name, data_generator.res_name)
        if not load_results:  # written summary files might not exist if we're loading the results from another run
            self.ref = pd.read_csv(self.label_file, sep='\t', header = 0)
            self.ref.sort_values(by='name', inplace=True)
            celltypes = self.ref['cell_type']
            classes = data_generator.classes
            if 'pfc_downsample' in data_generator.dataset_name:
                classes = np.array(['L2/3', 'L4', 'L5', 'L6', 'Ndnf', 'Vip', 'Pvalb', 'Sst', 'Astro', 'ODC', 'OPC', 'MG', 'MP',
                                'Endo'])
            #celltypes = data_generator.reference['cluster']
            self.y = []
            for l in celltypes:
                try:
                    self.y.append(np.argwhere(classes == l)[0])
                except IndexError:
                    self.y.append(0)
            self.y = np.squeeze(np.array(self.y))
            self.batches = np.array(self.ref['batch'])
            self.depths = np.array(self.ref['sparsity'])
            
        

    def get_embedding(self, iter_n=0):
        z = np.loadtxt(os.path.join(self.emb_dir, 'norm3DVI_PCA50.txt'))
        return z


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    x_train, y_train, depths, batches, train_generator, cm = parse_args(parser)
    args = parser.parse_args()

    print(depths)
    print(batches)

    experiment = DVIExperiment('3DVI', x_train, y_train, depths, train_generator)
    experiment.run(load=False)
