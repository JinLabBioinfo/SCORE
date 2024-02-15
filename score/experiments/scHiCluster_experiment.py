import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from score.experiments.experiment import Experiment


class ScHiClusterExperiment(Experiment):
    def __init__(self, name, y, depths, data_generator, method='innerproduct',  operations=['vc_sqrt_norm', 'convolution', 'random_walk'], n_components=16, downsample_depth=None, n_strata=None, strata_offset=0, downsample_amounts=None,
                 data_dir='schictools_data', rewrite=False, resolution_name='200kb', pca_baseline=False, load=False, **kwargs):
        super().__init__(name, None, y, depths, data_generator, **kwargs)
        self.n_components = n_components
        self.dataset_dir = os.path.join(data_dir, data_generator.dataset_name)
        self.data_dir = os.path.join(data_dir, data_generator.dataset_name, resolution_name)
        self.rewrite = rewrite
        self.dataset_name = data_generator.dataset_name
        os.makedirs(os.path.join(data_dir, data_generator.dataset_name), exist_ok=True)
        self.network = None
        self.val_network = None
        if not self.simulate and not load:
            if self.val_dataset is not None:
                self.val_network = self.prepare_schictools_data(self.val_dataset)
            self.network = self.prepare_schictools_data(self.data_generator)

        self.n_strata = int(n_strata) if n_strata is not None else None
        self.strata_offset = strata_offset
        self.operations = operations
        self.loaded_data = self.schictools_data
        self.val_loaded_data = self.val_schictools_data
        self.pca_baseline = pca_baseline
        if self.pca_baseline:
            self.operations = None


    def get_embedding(self, iter_n=0, reload_schictools_data=False):
        if self.simulate:
            rep_data_dir = os.path.join(self.data_dir, f'rep{iter_n}')
            self.network, self.chr_lengths = self.data_generator.write_scHiCTools_matrices(out_dir=rep_data_dir, resolution=self.resolution, rewrite=True)
            self.loaded_data = self.load_schictools_data(self.network, full_maps=True)
            embedding, cluster = self.loaded_data.scHiCluster(dim=self.latent_dim, cutoff=0.9, n_PCs=self.latent_dim, k=4)
        elif iter_n == 0:
            self.loaded_data = self.load_schictools_data(self.network, full_maps=True)
            if self.val_dataset is not None:
                self.val_loaded_data = self.load_schictools_data(self.val_network, full_maps=True)
            embedding, cluster = self.loaded_data.scHiCluster(dim=self.latent_dim, cutoff=0.9, n_PCs=self.latent_dim, k=4, val_data=self.val_loaded_data)
        else:
            embedding = self.load_embedding_from_file()
        #embedding, cluster = self.loaded_data.graph_distance(dim=20, cutoff=0.9, n_PCs=20, k=4)
        #cluster, embedding = raw_pca(self.network, self.chr_lengths, nc=self.data_generator.n_classes)
        return embedding
