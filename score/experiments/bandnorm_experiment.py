import os
import sys
import json
import pickle
import argparse
import traceback
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from score.sc_args import parse_args
from score.experiments.experiment import Experiment



class BandNormExperiment(Experiment):
    def __init__(self, name, x, y, depths, data_generator, args, depth_norm=True, base_dir='BandNorm', **kwargs):
        import anndata as ad
        super().__init__(name, x, y, depths, data_generator, **kwargs)
        self.res_file = '%s/output/%s_%s.tsv' % (base_dir, data_generator.dataset_name, data_generator.res_name)
        self.rna = None
        if args.rna_anndata is not None:
            print('Loading RNAseq reference data...')
            self.rna = ad.read_h5ad(args.rna_anndata)
            try:
                self.rna.obs['celltype'] = self.rna.obs['cluster']
            except Exception as e:
                print(e)
            try:
                self.rna.obs['batch'] = self.rna.obs['individual']
            except Exception as e:
                print(e)
            try:
                self.rna.X = self.rna.layers["counts"]
            except Exception as e:
                pass
            self.rna.layers["counts"] = self.rna.X.copy()
        #if depth_norm:
        #    self.res_file = self.res_file.replace('.tsv', '_depth_norm.tsv')
        #self.ref = pd.read_csv(self.label_file, sep='\t', names=['cell', 'Cell type', 'batch'])
        

    def get_embedding(self, iter_n=0):
        import scanpy as sc
        import anndata as ad
        n_components = self.latent_dim
        self.ref = pd.read_csv(self.res_file, sep='\t').transpose()
        print(self.ref)
        classes = np.array(self.cluster_names)
        celltypes = []
        filtered_cells = []
        for cellname in self.ref.index:
            try:
                celltypes.append(str(self.data_generator.reference.loc[cellname.replace('.int.bed', ''), 'cluster']))
            except KeyError:  # cell was filtered out...
                filtered_cells.append(cellname)
        self.ref.drop(filtered_cells, inplace=True)
        self.y = []
        for l in celltypes:
            self.y.append(np.argwhere(classes == l)[0])
        self.y = np.squeeze(np.array(self.y))
        self.batches = np.zeros(len(self.y))
        if len(self.features_dict['depth']) == 0:
            self.depths = np.ones(len(self.y))
        else:
            self.depths = np.array(self.features_dict['depth'])
  
        if self.rna is not None:
            try:
                self.rna.X = self.rna.layers["counts"]
            except Exception as e:
                pass
            adata = ad.AnnData(self.ref)
            var_names = self.rna.var_names.intersection(adata.var_names)
            print(len(var_names))
            
            self.rna = self.rna[:, var_names]
            print(self.rna)
            adata = adata[:, var_names]
            print(adata)
            adata.obs['celltype'] = celltypes
            adata.obs['batch'] = self.batches
            adata.obs['depth'] = self.depths
            self.rna.obs['old_celltype'] = self.rna.obs['celltype']
            self.rna.obs['celltype'] = self.rna.obs['celltype'].apply(lambda s: s + '_rna')

            celltypes = sorted(adata.obs['celltype'].unique())
            rna_celltypes = sorted(self.rna.obs['celltype'].unique())
            n_clusters = max(len(celltypes), len(rna_celltypes))
            colors = list(plt.cm.tab20(np.int32(np.linspace(0, n_clusters + 0.99, n_clusters))))
            if self.data_generator.color_config is not None:
                try:
                    with open(os.path.join('data/dataset_colors', self.data_generator.color_config), 'r') as f:
                        color_map = json.load(f)
                except FileNotFoundError:  
                    pass 
                color_map = color_map['colors']
                for c in color_map.keys():
                    color_map[c] = np.array([x / 255.0 for x in color_map[c]])
            else:
                color_map = {celltype: colors[i] for i, celltype in enumerate(celltypes)}
            rna_color_map = {celltype: colors[i] for i, celltype in enumerate(rna_celltypes)}
            try:  # pfc colors
                rna_color_map['AST-FB_rna'] = np.array([207, 243, 57]) / 255
                rna_color_map['AST-PP_rna'] = np.array([176, 242, 57]) / 255
                rna_color_map['Endothelial_rna'] = color_map['Endo']
                rna_color_map['IN-PV_rna'] = color_map['Pvalb']
                rna_color_map['IN-SST_rna'] = color_map['Sst']
                rna_color_map['IN-SV2C_rna'] = color_map['Ndnf']
                rna_color_map['IN-VIP_rna'] = color_map['Vip']
                rna_color_map['L2/3_rna'] = color_map['L2/3']
                rna_color_map['L4_rna'] = color_map['L4']
                rna_color_map['L5/6_rna'] = color_map['L6']
                rna_color_map['L5/6-CC_rna'] = color_map['L5']
                rna_color_map['Microglia_rna'] = color_map['MG']
                rna_color_map['Oligodendrocytes_rna'] = color_map['ODC']
                rna_color_map['OPC_rna'] = color_map['OPC']
                rna_color_map['Neu-NRGN-II_rna'] = np.array([0, 221, 84]) / 255
            except KeyError:
                pass
            color_map = {**color_map, **rna_color_map}
            color_map['Other'] = 'gray'

            print('Embedding scGAD gene scores...')
            adata.var.loc[:, 'highly_variable'] = True
            # # #sc.pp.highly_variable_genes(rna, min_mean=0.0125, max_mean=3, min_disp=0.5)
            #sc.pp.normalize_total(adata)
            #sc.pp.log1p(adata)
            #sc.pp.scale(adata)
            sc.tl.pca(adata, n_comps=n_components, svd_solver="auto")
            sc.pp.neighbors(adata, n_pcs=n_components, metric="cosine")
            sc.tl.umap(adata)

            fig = sc.pl.umap(adata, color=["celltype", "batch", "depth"], return_fig=True, wspace=0.6, palette=color_map)
            fig.tight_layout()
            fig.savefig(f"{self.out_dir}/pfc_scgad_scanpy_umap.png")
            plt.close()

            fig, ax = plt.subplots(figsize=(6, 6))
            sc.pl.umap(adata, color=['celltype'], ax=ax, palette=color_map, show=False, legend_loc=None, title='')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')
            fig.savefig(f"{self.out_dir}/pfc_hic_only_no_legend.png", dpi=400)
            plt.close()

            print('Embedding RNA reference...')
            #sc.pp.highly_variable_genes(self.rna, n_top_genes=10000, flavor="seurat_v3")
            self.rna.var.loc[:, 'highly_variable'] = True
            sc.pp.normalize_total(self.rna)
            sc.pp.log1p(self.rna)
            sc.pp.scale(self.rna)
            sc.tl.pca(self.rna, n_comps=n_components, svd_solver="auto")
            sc.pp.neighbors(self.rna, n_pcs=n_components, metric="cosine")
            sc.tl.umap(self.rna)

            fig = sc.pl.umap(self.rna, color=["celltype"], return_fig=True, wspace=0.6, palette=rna_color_map)
            fig.tight_layout()
            fig.savefig(f"{self.out_dir}/pfc_rna_reference_umap.png")
            plt.close()

            fig, ax = plt.subplots(figsize=(6, 6))
            sc.pl.umap(self.rna, color=['celltype'], ax=ax, show=False, legend_loc=None, title='', palette=rna_color_map)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')
            fig.savefig(f"{self.out_dir}/pfc_rna_only_no_legend.png", dpi=400)
            plt.close()
            
            print('Coembedding...')
            self.rna.obs['dataset'] = 'RNA'
            adata.obs['dataset'] = 'scHi-C'
            adata_concat = self.rna.concatenate(adata)
            sc.tl.pca(adata_concat)
            #sc.external.pp.bbknn(adata_concat, batch_key='dataset')
            #sc.external.pp.scanorama_integrate(adata_concat, 'dataset', knn=10)
            #sc.pp.neighbors(adata_concat, metric="cosine", use_rep='X_scanorama')
            integrate_key = 'X_pca'
            if 'harmony' in self.name:
                integrate_key = 'X_pca_harmony'
                sc.external.pp.harmony_integrate(adata_concat, 'dataset', random_state=iter_n)
            elif 'scanorama' in self.name:
                integrate_key = 'X_scanorama'
                np.random.seed(iter_n)
                sc.external.pp.scanorama_integrate(adata_concat, 'dataset')
            elif 'bbknn' in self.name:
                sc.external.pp.bbknn(adata_concat, metric='cosine', batch_key='dataset')
            elif 'mnn' in self.name:
                adata_concat = sc.external.pp.mnn_correct(self.rna, adata, batch_key='dataset')
                sc.tl.pca(adata_concat, svd_solver = 'arpack', use_highly_variable = False)
            elif 'combat' in self.name:
                # make sure dataset is a categorical variable
                adata_concat.obs['dataset'] = adata_concat.obs['dataset'].astype('category')
                sc.pp.combat(adata_concat, key='dataset')
                sc.tl.pca(adata_concat, svd_solver = 'arpack', use_highly_variable = False)
            elif 'scvi' in self.name:
                import scvi
                scvi.model.SCVI.setup_anndata(adata_concat, layer="counts", batch_key="dataset")
                scvi_model = scvi.model.SCVI(adata_concat, n_layers=2, n_latent=30)
                scvi_model.train()
                integrate_key = "X_scVI"
                adata_concat.obsm[integrate_key] = scvi_model.get_latent_representation()
            else:
                integrate_key = 'X_pca_harmony'
                sc.external.pp.harmony_integrate(adata_concat, 'dataset', random_state=iter_n)
            if 'bbknn' not in self.name:
                sc.pp.neighbors(adata_concat, metric="cosine", use_rep=integrate_key)
            sc.tl.umap(adata_concat)
            fig = sc.pl.umap(adata_concat, color=['celltype', 'dataset'], return_fig=True)
            fig.tight_layout()
            fig.savefig(f"{self.out_dir}/pfc_coembed_{iter_n}.png")
            plt.close()

            fig = sc.pl.umap(adata_concat[adata_concat.obs['dataset'] == 'scHi-C'], color=['celltype', 'depth'], return_fig=True, palette=color_map)
            fig.tight_layout()
            fig.savefig(f"{self.out_dir}/pfc_coembed_hic_only_{iter_n}.png")
            plt.close()

            fig = sc.pl.umap(adata_concat[adata_concat.obs['dataset'] == 'RNA'], color=['celltype'], wspace=0.35, return_fig=True, palette=rna_color_map)
            #fig.tight_layout()
            fig.savefig(f"{self.out_dir}/pfc_coembed_rna_only_{iter_n}.png")
            plt.close()

            fig, ax = plt.subplots(figsize=(6, 6))
            sc.pl.umap(adata_concat[adata_concat.obs['dataset'] == 'RNA'], color=['celltype'], ax=ax, show=False, legend_loc=None, title='', palette=rna_color_map)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')
            fig.savefig(f"{self.out_dir}/pfc_coembed_rna_only_no_legend_{iter_n}.png", dpi=400)
            plt.close()


            fig = sc.pl.umap(adata_concat, color=["celltype"], palette=color_map, groups=celltypes, wspace=0.35, size=80, return_fig=True)
            fig.tight_layout()
            fig.savefig(f"{self.out_dir}/pfc_coembed_grayed_out_{iter_n}.png")
            fig.savefig(f"{self.out_dir}/pfc_coembed_grayed_out_{iter_n}.pdf")
            plt.close()

            fig, ax = plt.subplots(figsize=(6, 6))
            sc.pl.umap(adata_concat, color=['celltype'], palette=color_map, groups=celltypes, size=80, ax=ax, show=False, legend_loc=None, title='')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')
            fig.savefig(f"{self.out_dir}/pfc_coembed_grayed_out_no_legend_{iter_n}.png", dpi=400)
            plt.close()

            sc.tl.rank_genes_groups(adata, groupby='celltype', method='wilcoxon')
            fig = sc.pl.rank_genes_groups_dotplot(adata, n_genes=4, return_fig=True)
            fig.savefig(f"{self.out_dir}/pfc_scgad_scanpy_dotplot.png")
            plt.close()

            try: # pfc neurons
                neurons = adata_concat[adata_concat.obs['celltype'].isin(['L2/3', 'L2/3_rna', 'L4', 'L4_rna', 'L5', 'L6', 'L5/6_rna', 'L5/6-CC_rna',
                                                                        'Pvalb', 'Sst', 'IN-PV_rna', 'IN-SST_rna', 'Ndnf', 'Vip', 'IN-SV2C_rna', 'IN-VIP_rna'])].copy()
                sc.pp.neighbors(neurons, metric="cosine", use_rep=integrate_key)
                sc.tl.umap(neurons)

                fig = sc.pl.umap(neurons, color=['celltype', 'dataset'], return_fig=True)
                fig.tight_layout()
                fig.savefig(f"{self.out_dir}/pfc_coembed_neurons.png")
                plt.close()

                fig = sc.pl.umap(neurons[neurons.obs['dataset'] == 'scHi-C'], color=['celltype', 'depth'], return_fig=True, palette='tab20')
                fig.tight_layout()
                fig.savefig(f"{self.out_dir}/pfc_coembed_neurons_hic_only.png")
                plt.close()

                fig, ax = plt.subplots(figsize=(6, 6))
                sc.pl.umap(neurons[neurons.obs['dataset'] == 'RNA'], color=['celltype'], ax=ax, show=False, legend_loc=None, title='', palette=rna_color_map)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.axis('off')
                fig.savefig(f"{self.out_dir}/pfc_coembed_neurons_rna_only_no_legend.png", dpi=400)
                plt.close()

                fig = sc.pl.umap(neurons, color=["celltype"], palette=color_map, groups=celltypes, wspace=0.35, size=80, return_fig=True)
                fig.tight_layout()
                fig.savefig(f"{self.out_dir}/pfc_neurons_coembed_grayed_out.png")
                fig.savefig(f"{self.out_dir}/pfc_neurons_coembed_grayed_out.pdf")
                plt.close()

                fig, ax = plt.subplots(figsize=(6, 6))
                sc.pl.umap(neurons, color=['celltype'], palette=color_map, groups=celltypes, ax=ax, show=False, legend_loc=None, title='', size=80)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.axis('off')
                fig.savefig(f"{self.out_dir}/pfc_neurons_coembed_grayed_out_no_legend.png", dpi=400)
                plt.close()
            except Exception as e:
                print(e)
                pass

            fig = sc.pl.rank_genes_groups_matrixplot(adata, n_genes=4, vmin=-1, vmax=1, cmap='bwr', return_fig=True)
            fig.savefig(f"{self.out_dir}/pfc_scgad_scanpy_matrixplot.png")
            plt.close()


            print('Done!')
            z_pca = np.array(adata_concat[adata_concat.obs['dataset'] == 'scHi-C'].obsm[integrate_key])
        else:
            z = self.ref.to_numpy()
            z_pca = PCA(min(n_components, z.shape[1], z.shape[0])).fit_transform(z)
        return z_pca


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth_norm', action='store_true')
    x_train, y_train, depths, batches, train_generator, cm = parse_args(parser)
    args = parser.parse_args()
    depth_norm = args.depth_norm
    exp_name = "BandNorm"

    features = {'batch': batches, 'depth': depths}

    if depth_norm:
        exp_name += '_depth_norm'
    experiment = BandNormExperiment(exp_name, x_train, y_train, features, train_generator, depth_norm=depth_norm)
    experiment.run(load=False)
