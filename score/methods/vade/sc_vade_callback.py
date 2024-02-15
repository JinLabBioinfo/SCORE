import os
import time
import random
import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from munkres import Munkres
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, homogeneity_score
from sklearn.metrics import silhouette_score
from sklearn import manifold
from tensorflow.keras.callbacks import Callback

nmi = normalized_mutual_info_score
ari = adjusted_rand_score
acc = homogeneity_score


class VisualizeCallback(Callback):

    def __init__(self, x, y, encoder, decoder, autoencoder, gmm, data_generator, n_clusters, n_cell_types, features,
                 n_epochs, batch_size, log_wandb=False, remap_centroids=True, ll_norm=1, save_interval=50, update_interval=100, n_samples=1, n_attempts=20, save_dir='vade/', model_dir='vade_models/',
                 style='default'):
        self.x = x
        self.y = np.squeeze(np.asarray(y).ravel())
        self.encoder = encoder
        self.decoder = decoder
        self.autoencoder = autoencoder
        self.gmm = gmm
        self.component_dist = None
        self.data_generator = data_generator
        self.n_clusters = n_clusters
        self.n_cell_types = n_cell_types
        print('%d cell types across %d clusters' % (self.n_cell_types, self.n_clusters))
        self.features = features
        self.batch_size = batch_size
        self.log_wandb = log_wandb
        self.ll_norm = ll_norm  # product of input dims to normalize log-likelihood between different sized inputs
        self.save_interval = save_interval
        self.update_interval = update_interval
        self.n_samples = n_samples
        self.n_attempts = n_attempts
        self.save_dir = save_dir
        self.model_dir = model_dir
        self.remap_centroids = remap_centroids
        # = data_generator.n_classes
        self.cluster_names = data_generator.classes
        self.clustering_algs = {'k-means': KMeans,
                                'agglomerative': AgglomerativeClustering,
                                'gmm': GaussianMixture,
                                'VaDE': None}
        color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:red']  # list of possible colors
        self.alg_colors = {}  # dict mapping algorithms to colors for plotting
        for i, alg in enumerate(self.clustering_algs.keys()):  # fill dict with colors
            self.alg_colors[alg] = color_list[i]
        if self.data_generator.n_cells < 500:
            self.scatter_size = 6
        else:
            self.scatter_size = 2
        if 'pfc' in self.data_generator.dataset_name:
            self.cluster_names = ['L2/3', 'L4', 'L5', 'L6', 'Ndnf', 'Vip', 'Pvalb', 'Sst', 'Astro', 'ODC', 'OPC', 'MG', 'MP',
                             'Endo']
            sorted_idxs = []
            for i, name in enumerate(data_generator.classes):
                sorted_idxs.append(self.cluster_names.index(name))
            sorted_idxs = np.array(sorted_idxs)

            color_matrix = np.array([[230, 25, 75],
                                     [60, 180, 75],
                                     [255, 225, 25],
                                     [0, 130, 200],
                                     [245, 130, 49],
                                     [145, 30, 180],
                                     [70, 240, 240],
                                     [240, 50, 230],
                                     [210, 245, 60],
                                     [250, 190, 190],
                                     [0, 128, 128],
                                     [230, 190, 255],
                                     [170, 110, 40],
                                     [128, 0, 0]])
            color_matrix = color_matrix[sorted_idxs]
            self.cluster_cmap = matplotlib.colors.ListedColormap(color_matrix / 255.0)
        elif self.n_cell_types == 2:
            self.cluster_cmap = plt.cm.get_cmap("bwr", self.n_cell_types)
        elif self.n_cell_types == 1:
            self.cluster_cmap = plt.cm.get_cmap("plasma", self.n_cell_types)
        elif self.n_cell_types <= 9:
            self.cluster_cmap = plt.cm.get_cmap("jet", self.n_cell_types)
        elif self.n_cell_types <= 20:
            self.cluster_cmap = plt.cm.get_cmap("tab20", self.n_cell_types)
        else:
            self.cluster_cmap = plt.cm.get_cmap("nipy_spectral", self.n_cell_types)

        self.losses_vae = np.empty((n_epochs,))
        self.homo_plot = {}
        self.nmi_plot = {}
        self.ari_plot = {}
        self.sil_plot = {}
        self.pc_homo_plot = {}
        self.pc_nmi_plot = {}
        self.pc_ari_plot = {}
        self.pc_sil_plot = {}
        for alg in self.clustering_algs.keys():
            self.homo_plot[alg] = []
            self.nmi_plot[alg] = []
            self.ari_plot[alg] = []
            self.sil_plot[alg] = []
            self.pc_homo_plot[alg] = []
            self.pc_nmi_plot[alg] = []
            self.pc_ari_plot[alg] = []
            self.pc_sil_plot[alg] = []
        self.start_time = time.time()
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        #plt.style.use(style)
        self.on_epoch_end(epoch=-1)  # init epoch, sets the weights of the GMM layer so we have gradient

    @staticmethod
    def cluster_acc(y_true, y_pred):
        """
        Calculate clustering accuracy. Require scikit-learn installed
        # Arguments
            y: true labels, numpy.array with shape `(n_samples,)`
            y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        # Return
            accuracy, in [0,1]
        """
        y_true = y_true.astype(np.int64)
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        m = Munkres()
        ind = m.compute(w.max() - w)
        ind = np.array(ind)
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    @staticmethod
    def match_means(mus, new_mus):
        n_cols = mus.shape[0]
        w = np.zeros((n_cols, n_cols))
        for i in range(n_cols):
            for j in range(n_cols):
                if j < new_mus.shape[1]:
                    #pi_dist = (pis[i] - new_pis[j]) ** 2  # squared distance between probs
                    mu_mse = np.mean(np.square(mus[i, ...] - new_mus[j, ...]))  # MSE between current and GMM fit mean
                    #sigma_mse = np.mean(np.square(sigmas[..., i] - new_sigmas[..., j]))  # MSE between current and GMM fit covariance
                    w[i, j] = mu_mse
                else:
                    print('Could not fully match clusters params...')
        m = Munkres()
        indices = m.compute(w)
        indices = np.array(indices)
        cols = indices[:, 1]
        print(cols)
        return cols

    def on_epoch_end(self, epoch, logs=None):
        import umap
        import wandb
        import tensorflow as tf
        from tensorflow.linalg import LinearOperatorDiag
        from tensorflow.python.ops.linalg import linear_operator_util
        import tensorflow_probability as tfp
        for dist in self.gmm.submodules:
            if isinstance(dist, tfp.distributions.Categorical) and dist.parameters['name'] == 'prior_categorical':
                logits = dist.parameters['logits']
                p_c = logits - tf.reduce_logsumexp(logits, axis=0, keepdims=True)
                #print('p_c', p_c)
            elif isinstance(dist, tfp.distributions.MultivariateNormalDiag) and 'name' in dist.parameters:
                if dist.parameters['name'] == 'prior':
                    self.component_dist = dist
        if logs is not None and self.log_wandb:
            wandb.log({'elbo': np.mean(logs['loss']) / self.ll_norm,
                    'n_obs': epoch * len(self.data_generator.cell_list)})
        if epoch == 0 or epoch % self.save_interval == 0:  # init GMM params or update based on current embedding
            g = GaussianMixture(n_components=self.n_clusters, n_init=5, random_state=36, covariance_type='diag')
            batch_size = 32
            embedding = None
            for batch_i in range(0, self.x.shape[0], batch_size):
                tmp = self.encoder(self.x[batch_i: batch_i + batch_size]).mean()
                if embedding is None:
                    embedding = tmp
                else:
                    embedding = tf.concat([embedding, tmp], axis=0)
            # save embedding in case we need to load it later
            np.save(self.save_dir + f'embedding_{epoch}.npy', embedding)

            fig, axs = plt.subplots(4, 5, figsize=(28, 25))
            fig.subplots_adjust(wspace=0.25)

            try:
                self.losses_vae[epoch] = np.mean(logs['loss']) / self.ll_norm
            except KeyError:
                print(logs)
                if epoch > 0:
                    self.losses_vae[epoch] = self.losses_vae[epoch - 1]
            except ValueError:
                print(logs)
                if epoch > 0:
                    self.losses_vae[epoch] = self.losses_vae[epoch - 1]
            try:
                pca = PCA(n_components=embedding.shape[1])  # get full PC decomposition
                pc_embeddings = pca.fit_transform(embedding)
                pc_embeddings_no_pc1 = pc_embeddings[:, 1:]
            except ValueError:
                print('NaNs in embedding :(')
                return
            except np.linalg.LinAlgError:
                print('SVD did not converge :(')
                return
            for alg in self.clustering_algs.keys():
                if alg == 'gmm':
                    predicted_labels = g.fit_predict(embedding)
                    if self.remap_centroids and not epoch == 0:
                        cluster_count = np.bincount(predicted_labels)  # count frequency of each label
                        cluster_probs = cluster_count / np.sum(cluster_count)  # turn into prob distribution
                        if len(cluster_probs) < self.n_clusters:
                            cluster_probs = np.concatenate((cluster_probs, np.zeros(self.n_clusters - len(cluster_probs))))
                        new_probs = g.means_  # updated gmm probabilities for new gmm layer weights
                        new_covariances = g.covariances_
                        current_probs = self.component_dist.mean()
                        match_cols = self.match_means(current_probs, new_probs)
                        new_probs = new_probs[match_cols, :]
                        new_covariances = new_covariances[match_cols, :]
                        cluster_probs = cluster_probs[match_cols]
                        for dist in self.gmm.submodules:
                            if isinstance(dist, tfp.distributions.Categorical) and dist.parameters['name'] == 'prior_categorical':
                                dist.logits.assign(np.float32(np.log(cluster_probs + 1e-6)))
                            elif isinstance(dist, tfp.distributions.MultivariateNormalDiag) and dist.parameters['name'] == 'prior':
                                dist.loc.assign(np.float32(new_probs))
                                dist.scale._diag = linear_operator_util.convert_nonref_to_tensor(np.float32(new_covariances), name="diag")
                    pc_gmm = GaussianMixture(n_components=self.n_cell_types, covariance_type='diag')
                    try:
                        pc_predicted_labels = pc_gmm.fit_predict(pc_embeddings_no_pc1)
                    except ValueError:
                        self.pc_homo_plot[alg].append(0)
                        self.pc_nmi_plot[alg].append(0)
                        self.pc_ari_plot[alg].append(0)
                        continue
                elif alg == 'VaDE':
                    z_t = tf.keras.backend.repeat(embedding, self.n_clusters)
                    probs = self.component_dist.log_prob(z_t)
                    probs = probs + p_c
                    prob_hist = tf.reduce_mean(tf.exp(probs - tf.reduce_logsumexp(probs, axis=1, keepdims=True)), axis=0)
                    predicted_labels = np.argmax(probs, axis=-1)
                else:
                    try:
                        cluster = self.clustering_algs[alg](n_clusters=self.n_cell_types)
                        pc_predicted_labels = cluster.fit_predict(pc_embeddings_no_pc1)
                        predicted_labels = cluster.fit_predict(
                            embedding)  # cluster on current embeddings for metric eval
                    except ValueError as e:
                        print(e)
                        continue
                    except TypeError as e:  # skip VaDE for now
                        continue

                self.homo_plot[alg].append(self.cluster_acc(self.y, predicted_labels))
                self.nmi_plot[alg].append(nmi(self.y, predicted_labels))
                self.ari_plot[alg].append(ari(self.y, predicted_labels))
                if len(np.unique(predicted_labels)) > 1:  # silhouette score requires at least 2 unique labels
                    self.sil_plot[alg].append(silhouette_score(embedding, predicted_labels))
                else:
                    self.sil_plot[alg].append(0)

                try:
                    self.pc_homo_plot[alg].append(self.cluster_acc(self.y, pc_predicted_labels))
                    self.pc_nmi_plot[alg].append(nmi(self.y, pc_predicted_labels))
                    self.pc_ari_plot[alg].append(ari(self.y, pc_predicted_labels))
                    if len(np.unique(pc_predicted_labels)) > 1:
                        self.pc_sil_plot[alg].append(silhouette_score(pc_embeddings_no_pc1, pc_predicted_labels))
                    else:
                        self.pc_sil_plot[alg].append(0)
                    if self.log_wandb:
                        wandb.log({
                            'train_ari_' + alg: self.ari_plot[alg][-1],
                            'train_nmi_' + alg: self.nmi_plot[alg][-1],
                            'train_acc_' + alg: self.homo_plot[alg][-1],
                            'train_silhouette_' + alg: self.sil_plot[alg][-1],
                            'train_ari_no_pc1_' + alg: self.pc_ari_plot[alg][-1],
                            'train_nmi_no_pc1_' + alg: self.pc_nmi_plot[alg][-1],
                            'train_acc_no_pc1_' + alg: self.pc_homo_plot[alg][-1],
                            'train_silhouette_no_pc1_' + alg: self.pc_sil_plot[alg][-1]})
                except Exception as e:  # only 2 dimensions
                    print(e)
                    if self.log_wandb:
                        wandb.log({
                            'train_ari_' + alg: self.ari_plot[alg][-1],
                            'train_nmi_' + alg: self.nmi_plot[alg][-1],
                            'train_acc_' + alg: self.homo_plot[alg][-1],
                            'train_silhouette_' + alg: self.sil_plot[alg][-1]})
                    pass

                

            try:
                print(
                    'epoch {}, Loss: {:7.4f}, Acc: {:7.3f}, NMI: {:7.3f}, ARI: {:7.3f} Silhouette: {:7.3f} \tNo PC1 --> Acc: {:7.3f}, NMI: {:7.3f}, ARI: {:7.3f} Silhouette: {:7.3f}'.format(
                        epoch,
                        np.mean(logs['loss']) / self.ll_norm,
                        self.homo_plot['gmm'][-1],
                        self.nmi_plot['gmm'][-1],
                        self.ari_plot['gmm'][-1],
                        self.sil_plot['gmm'][-1],
                        self.pc_homo_plot['gmm'][-1],
                        self.pc_nmi_plot['gmm'][-1],
                        self.pc_ari_plot['gmm'][-1],
                        self.pc_sil_plot['gmm'][-1]
                        ))
            except KeyError:
                pass
            except TypeError:
                pass

            tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
            Z_tsne = tsne.fit_transform(embedding)
            sc1 = axs[1][0].scatter(Z_tsne[:, 0], Z_tsne[:, 1], s=self.scatter_size, c=self.y,
                                    cmap=self.cluster_cmap)
            axs[1][0].set_title('t-SNE Embeddings')
            axs[1][0].set_xlabel('t-SNE 1')
            axs[1][0].set_ylabel('t-SNE 2')
            axs[1][0].set_xticks([])
            axs[1][0].set_yticks([])
            axs[1][0].spines['right'].set_visible(False)
            axs[1][0].spines['top'].set_visible(False)
            divider = make_axes_locatable(axs[1][0])
            cax = divider.append_axes('right', size='15%', pad=0.05)
            cbar1 = fig.colorbar(sc1, cax=cax, orientation='vertical')
            tick_locs = (np.arange(self.n_cell_types) + 0.5) * (self.n_cell_types - 1) / self.n_cell_types
            cbar1.set_ticks(tick_locs)
            try:
                cbar1.set_ticklabels(self.cluster_names)  # vertically oriented colorbar
            except ValueError as e:  # more clusters than labels (usually when 1 unknown cluster)
                pass

            means = self.component_dist.mean()
            means_gmm = g.means_

            reducer = umap.UMAP(transform_seed=36, random_state=36, metric='cosine')
            matrix_reduce = reducer.fit_transform(embedding)
            means_umap = reducer.transform(means)
            means_gmm_umap = reducer.transform(means_gmm)
            axs[1][1].scatter(matrix_reduce[:, 0], matrix_reduce[:, 1], s=self.scatter_size, c=self.y,
                              cmap=self.cluster_cmap)
            axs[1][1].scatter(means_umap[:, 0], means_umap[:, 1], linewidth=1, s=16, c='k', marker='x')
            axs[1][1].scatter(means_gmm_umap[:, 0], means_gmm_umap[:, 1], linewidth=1, s=16, c='r', marker='x')
            axs[1][1].set_title('UMAP Embeddings')
            axs[1][1].set_xlabel('UMAP 1')
            axs[1][1].set_ylabel('UMAP 2')
            axs[1][1].set_xticks([])
            axs[1][1].set_yticks([])
            # Hide the right and top spines
            axs[1][1].spines['right'].set_visible(False)
            axs[1][1].spines['top'].set_visible(False)

            axs[1][2].scatter(matrix_reduce[:, 0], matrix_reduce[:, 1], s=self.scatter_size, c=self.features['batch'],
                              cmap=self.cluster_cmap)
            axs[1][2].scatter(means_umap[:, 0], means_umap[:, 1], linewidth=1, s=16, c='k', marker='x')
            axs[1][2].set_title('UMAP Batch')
            axs[1][2].set_xlabel('UMAP 1')
            axs[1][2].set_ylabel('UMAP 2')
            axs[1][2].set_xticks([])
            axs[1][2].set_yticks([])
            # Hide the right and top spines
            axs[1][2].spines['right'].set_visible(False)
            axs[1][2].spines['top'].set_visible(False)

            axs[0][0].plot(self.losses_vae[:epoch + 1])
            axs[0][0].set_title('VAE Loss')
            axs[0][0].set_xlabel('epochs')

            for alg in self.clustering_algs.keys():
                pc_label = alg + ' (no PC1)'
                axs[0][1].plot(self.homo_plot[alg], label=alg, color=self.alg_colors[alg])
                if alg != 'VaDE':
                    axs[0][1].plot(self.pc_homo_plot[alg], label=pc_label, color=self.alg_colors[alg], linestyle='--')
                axs[0][1].set_title('Cluster Accuracy')
                axs[0][1].set_xlabel('epochs')
                axs[0][1].set_ylim(0, 1)
                axs[0][1].legend(loc='best')

                axs[0][2].plot(self.ari_plot[alg], label=alg, color=self.alg_colors[alg])
                if alg != 'VaDE':
                    axs[0][2].plot(self.pc_ari_plot[alg], label=pc_label, color=self.alg_colors[alg], linestyle='--')
                axs[0][2].set_title('ARI')
                axs[0][2].set_xlabel('epochs')
                axs[0][2].set_ylim(0, 1)
                axs[0][2].legend(loc='best')

                axs[0][3].plot(self.sil_plot[alg], label=alg, color=self.alg_colors[alg])
                if alg != 'VaDE':
                    axs[0][3].plot(self.pc_sil_plot[alg], label=pc_label, color=self.alg_colors[alg], linestyle='--')
                axs[0][3].set_title('Silhouette Score')
                axs[0][3].set_xlabel('epochs')
                axs[0][3].set_ylim(0, 1)
                axs[0][3].legend(loc='best')

            sc1 = axs[2][0].scatter(Z_tsne[:, 0], Z_tsne[:, 1], s=self.scatter_size, c=self.features['depth'],
                                    cmap=plt.cm.get_cmap("plasma"))
            axs[2][0].set_title('t-SNE Read Depth')
            axs[2][0].set_xlabel('t-SNE 1')
            axs[2][0].set_ylabel('t-SNE 2')
            axs[2][0].set_xticks([])
            axs[2][0].set_yticks([])
            axs[2][0].spines['right'].set_visible(False)
            axs[2][0].spines['top'].set_visible(False)
            divider = make_axes_locatable(axs[2][0])
            cax = divider.append_axes('right', size='15%', pad=0.05)
            cbar1 = fig.colorbar(sc1, cax=cax, orientation='vertical')

            axs[2][1].scatter(matrix_reduce[:, 0], matrix_reduce[:, 1], s=self.scatter_size, c=self.features['depth'],
                              cmap=plt.cm.get_cmap("plasma"))
            axs[2][1].set_title('UMAP Read Depth')
            axs[2][1].set_xlabel('UMAP 1')
            axs[2][1].set_ylabel('UMAP 2')
            axs[2][1].set_xticks([])
            axs[2][1].set_yticks([])
            # Hide the right and top spines
            axs[2][1].spines['right'].set_visible(False)
            axs[2][1].spines['top'].set_visible(False)

            example_index = random.randint(0, self.data_generator.n_cells)
            #example_cell, _, _, _ = self.data_generator.get_cell_by_index(example_index, downsample=False)
            #example_cell = np.expand_dims(example_cell, 0)
            example_cell = self.data_generator.example_cell
            cell_tile = example_cell[0, ..., 0]
            example_width = cell_tile.shape[0] * 5
            start_idx = 1024
            example_idxs = slice(start_idx, start_idx + example_width)
            cell_tile = cell_tile[:, start_idx:start_idx + example_width]
            cell_tile = np.flipud(cell_tile)
            #'''
            #z_sample = self.encoder(self.data_generator.example_cell)
            # if self.data_generator.batch_remove or self.data_generator.contrastive_batches:
            #     x_recon = self.decoder(self.encoder(example_cell)).mean()
            # else:
            x_recon = self.autoencoder(example_cell).mean()
            x_recon = x_recon[0, ..., 0]
            reconstructed_cell_tile = x_recon[:, start_idx:start_idx + example_width]

            example_mat = np.zeros((example_width, example_width))
            recon_mat = np.zeros((example_width, example_width))
            for k in range(x_recon.shape[0]):
                diag_i = np.arange(k, cell_tile.shape[1])
                diag_j = np.arange(0, cell_tile.shape[1] - k)
                example_mat[diag_i, diag_j] = example_cell[0, k, example_idxs, 0][k:]
                example_mat[diag_j, diag_i] = example_cell[0, k, example_idxs, 0][k:]
                recon_mat[diag_i, diag_j] = x_recon[k, example_idxs][k:]
                recon_mat[diag_j, diag_i] = x_recon[k, example_idxs][k:]

            mat_ax = axs[2][2]
            mat_ax.imshow(np.log1p(example_mat), cmap='Reds')
            mat_ax.set_xticks([])
            mat_ax.set_yticks([])
            mat_ax.spines['right'].set_visible(False)
            mat_ax.spines['top'].set_visible(False)
            mat_ax.spines['left'].set_visible(False)
            mat_ax.spines['bottom'].set_visible(False)
            mat_ax.set_title('Input Matrix')

            mat_ax = axs[2][3]
            mat_ax.imshow(np.log1p(recon_mat), cmap='Reds')
            mat_ax.set_xticks([])
            mat_ax.set_yticks([])
            mat_ax.spines['right'].set_visible(False)
            mat_ax.spines['top'].set_visible(False)
            mat_ax.spines['left'].set_visible(False)
            mat_ax.spines['bottom'].set_visible(False)
            mat_ax.set_title('Pred Matrix')

            #reconstructed_cell_tile = np.flipud(reconstructed_cell_tile)
            cell_heatmap = np.vstack((cell_tile, np.zeros(cell_tile.shape[1]), reconstructed_cell_tile))
            axs[1][3].imshow(np.log1p(cell_heatmap), cmap='Reds')
            axs[1][3].set_xticks([])
            axs[1][3].set_yticks([])
            axs[1][3].spines['right'].set_visible(False)
            axs[1][3].spines['top'].set_visible(False)
            axs[1][3].spines['left'].set_visible(False)
            axs[1][3].spines['bottom'].set_visible(False)
            #'''

            means_pc = pca.transform(means)
            means_gmm_pc = pca.transform(means_gmm)
            axs[3][2].scatter(pc_embeddings[:, 0], pc_embeddings[:, 1], s=self.scatter_size, c=self.y,
                              cmap=self.cluster_cmap)
            axs[3][2].scatter(means_pc[:, 0], means_pc[:, 1], linewidth=1, s=16, c='k', marker='x')
            axs[3][2].scatter(means_gmm_pc[:, 0], means_gmm_pc[:, 1], linewidth=1, s=16, c='r', marker='x')
            axs[3][2].set_title('PCA Embedding')
            axs[3][2].set_xlabel('PC 1')
            axs[3][2].set_ylabel('PC 2')
            axs[3][2].set_xticks([])
            axs[3][2].set_yticks([])
            # Hide the right and top spines
            axs[3][2].spines['right'].set_visible(False)
            axs[3][2].spines['top'].set_visible(False)

            axs[3][3].scatter(pc_embeddings[:, 0], pc_embeddings[:, 1], s=self.scatter_size, c=self.features['depth'],
                              cmap=plt.cm.get_cmap("inferno"))
            axs[3][3].set_title('PCA depth')
            axs[3][3].set_xlabel('PC 1')
            axs[3][3].set_ylabel('PC 2')
            axs[3][3].set_xticks([])
            axs[3][3].set_yticks([])
            # Hide the right and top spines
            axs[3][3].spines['right'].set_visible(False)
            axs[3][3].spines['top'].set_visible(False)

            try:
                depth_tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
                pc_tsne = depth_tsne.fit_transform(
                    pc_embeddings_no_pc1)  # tsne without first pc (should correspond to depth)
                sc1 = axs[3][0].scatter(pc_tsne[:, 0], pc_tsne[:, 1], s=self.scatter_size, c=self.y,
                                        cmap=self.cluster_cmap)
                axs[3][0].set_title('t-SNE (w/o PC-1)')
                axs[3][0].set_xlabel('t-SNE 1')
                axs[3][0].set_ylabel('t-SNE 2')
                axs[3][0].set_xticks([])
                axs[3][0].set_yticks([])
                axs[3][0].spines['right'].set_visible(False)
                axs[3][0].spines['top'].set_visible(False)

                depth_reducer = umap.UMAP(transform_seed=36, random_state=36, metric='cosine')
                depth_matrix_reduce = depth_reducer.fit_transform(pc_embeddings_no_pc1)
                sc1 = axs[3][1].scatter(depth_matrix_reduce[:, 0], depth_matrix_reduce[:, 1], s=self.scatter_size, c=self.y,
                                        cmap=self.cluster_cmap)
                axs[3][1].set_title('UMAP (w/o PC-1)')
                axs[3][1].set_xlabel('UMAP 1')
                axs[3][1].set_ylabel('UMAP 2')
                axs[3][1].set_xticks([])
                axs[3][1].set_yticks([])
                axs[3][1].spines['right'].set_visible(False)
                axs[3][1].spines['top'].set_visible(False)
            except ValueError:  # not enough dimensions in latent space
                pass

            # plot first layer filters
            # get the symbolic outputs of each "key" layer (we gave them unique names).
            conv_layer = self.encoder.layers[1]  # first layer is InputLayer
            filters, biases = conv_layer.get_weights()

            def band_to_mat(b):
                size = b.shape[1] + b.shape[0]
                mat = np.zeros((size, size))
                for i, band in enumerate(b):
                    idxs = np.arange(band.size)
                    mat[idxs + i, idxs] = band 
                    mat[idxs, idxs + i] = band
                return mat

            for i in range(4):
                filter_ax = axs[i][4]
                try:
                    img = filters[:, :, 0, i]
                    #mat = band_to_mat(gaussian_filter(img, sigma=0.1))
                    filter_ax.imshow(gaussian_filter(img, sigma=0.2), cmap='RdGy_r')
                    filter_ax.set_xticks([])
                    filter_ax.set_yticks([])
                except IndexError:  # no more filters
                    break

            train_time = str(datetime.timedelta(seconds=(int(time.time() - self.start_time))))
            n_matrices = (epoch + 1) * self.data_generator.n_cells
            fig.suptitle('Trained on ' + '{:,}'.format(n_matrices) + ' cells\n' + train_time)

            plt.savefig('%s%d.png' % (self.save_dir, epoch))
            plt.close()

            # plot centroids and covariance ellipses for each cluster in component distribution
            means = self.component_dist.mean()
            means_gmm = g.means_
            means_umap = reducer.transform(means)
            means_gmm_umap = reducer.transform(means_gmm)
            covs = self.component_dist.covariance()
            print(covs.shape)
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.scatter(matrix_reduce[:, 0], matrix_reduce[:, 1], s=self.scatter_size, c=self.y, cmap=self.cluster_cmap, alpha=0.25)
            ax.scatter(means_umap[:, 0], means_umap[:, 1], linewidth=1, s=16, c='k', marker='x')
            ax.scatter(means_gmm_umap[:, 0], means_gmm_umap[:, 1], linewidth=1, s=16, c='r', marker='x')
            for i in range(self.n_clusters):
                cov = covs[i]
                cov = np.diag(cov)
                cov = np.sqrt(cov)
                cov = np.expand_dims(cov, 0)
                cov_umap = reducer.transform(cov)
                # set color to cluster color
                ax.add_patch(matplotlib.patches.Ellipse(means_umap[i], cov_umap[0, 1], cov_umap[0, 0], alpha=0.25))
            ax.set_title('GMM Means and Covariances')
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            plt.savefig(self.save_dir + f'gmm_{epoch}.png')
            plt.close()

        else:
            if epoch >= 0:
                try:
                    self.losses_vae[epoch] = np.mean(logs['loss']) / self.ll_norm
                except KeyError:
                    print(logs)
                    if epoch > 0:
                        self.losses_vae[epoch] = self.losses_vae[epoch - 1]
                except ValueError:
                    print(logs)
                    if epoch > 0:
                        self.losses_vae[epoch] = self.losses_vae[epoch - 1]
