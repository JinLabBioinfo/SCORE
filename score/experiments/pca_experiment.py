import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from score.experiments.experiment import Experiment


class PCAExperiment(Experiment):
    def __init__(self, name, x, y, features, data_generator, **kwargs):
        super().__init__(name, x, y, features, data_generator, **kwargs)
        self.n_components = self.latent_dim

    def get_embedding(self, iter_n=0, remove_pc1=True):
        pca = PCA(n_components=min(self.n_components, self.x.shape[0], self.x.shape[1]))
        pca.fit(self.x)
        if self.val_dataset is not None:
            z = pca.transform(self.val_x)
        else:
            z = pca.transform(self.x)
        if remove_pc1:
            return z[..., 1:]
        else:
            return z
