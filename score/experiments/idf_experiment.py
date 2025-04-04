import os
import numpy as np
import anndata as ad
from score.experiments.experiment import Experiment
from score.utils.matrix_ops import idf_inner_product, per_strata_idf_inner_product


def analyze_power_law(data):
    import matplotlib.pyplot as plt 
    from scipy.stats import linregress
    num_observations, num_features = data.shape
    
    # Calculate feature-wise frequencies
    feature_frequencies = []
    for feature in range(num_features):
        unique, counts = np.unique(data[:, feature], return_counts=True)
        feature_frequencies.extend(counts[unique > 1])  # Exclude counts of 0 and 1
    
    # Sort frequencies in descending order
    frequencies = np.sort(feature_frequencies)[::-1]
    
    # Create log-log plot
    plt.figure(figsize=(10, 6))
    plt.loglog(range(1, len(frequencies) + 1), frequencies, 'bo-', markersize=3, label='Data')
    
    # Fit power law
    x = np.log(range(1, len(frequencies) + 1))
    y = np.log(frequencies)
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    
    # Plot fitted line
    fit_line = np.exp(intercept) * np.power(range(1, len(frequencies) + 1), slope)
    plt.loglog(range(1, len(frequencies) + 1), fit_line, 'r-', label=f'Fitted Power Law (slope = {slope:.2f})')
    
    plt.title('Feature-Wise Frequency Distribution (Log-Log Scale)')
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    
    print(f"Power Law Exponent (slope): {-slope:.4f}")
    print(f"R-squared: {r_value**2:.4f}")


class IDF2DExperiment(Experiment):
    def __init__(self, name, x, y, features, data_generator, n_strata, strata_k,
                 preprocessing=None, **kwargs):
        super().__init__(name, x, y, features, data_generator, **kwargs)
        self.n_strata = n_strata
        self.n_components = self.latent_dim
        self.preprocessing = preprocessing
        self.strata_k = strata_k

    def get_embedding(self, iter_n=0, remove_pc1=True):
        import matplotlib.pyplot as plt 
        import seaborn as sns

        if self.preprocessing is None:  # using raw count data
            hic = ad.AnnData(self.x, dtype='int32')
        else:  # using whatever values are passed in (preprocessed data like normalized probs)
            hic = ad.AnnData(self.x)
        hic.obs_names = sorted(self.data_generator.cell_list)
        hic.obs_names = hic.obs_names.map(lambda s: s.replace(f'.{self.data_generator.res_name}', ''))
        idf_inner_product(hic, self.strata_k)
        #per_strata_idf_inner_product(hic, self.strata_k)
        # visualize per-strata distributions before and after idf
        os.makedirs(os.path.join(self.out_dir, 'celltype_plots'), exist_ok=True)
        from scipy.stats import linregress, kstest
        import scanpy as sc
        import powerlaw
        import math
        adata_use = hic

        x_nonzero = adata_use.X[adata_use.X > 1].flatten()
        hist, bins = np.histogram(x_nonzero, bins=np.arange(0, x_nonzero.max(), 1), density=True)
        plt.scatter(np.arange(len(hist)), np.sort(hist)[::-1])
        plt.yscale('log')
        plt.xscale('log')
        plt.title("All strata")
        plt.savefig(os.path.join(self.out_dir, f'celltype_plots/log_log_raw_counts_nonzero_all.png'))
        plt.close()

        X = adata_use.X
        idf = np.log(X.shape[0] / (np.sum(X > 0, axis=0) + 1))
        X_idf = X * idf
        X = np.nan_to_num(X)
        X_idf = np.nan_to_num(X_idf)
        n_strata = len(np.unique(self.strata_k))
        #fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        colors = plt.cm.get_cmap('Spectral')(np.linspace(0, 1, n_strata))
        epon_vals = []
        expon_ps = []
        lognorm_vals = []
        lognorm_ps = []
        k_stats = []
        p_values = []
        r_values = []
        linreg_pvals = []
        width = int(math.sqrt(n_strata))
        height = int(math.ceil(n_strata / width))
        fig, axs = plt.subplots(height, width, figsize=(14, 14))
        for k, ax in enumerate(axs.flatten()):
            print(k)
            # strata_X = X[:, self.strata_k == k]
            # analyze_power_law(strata_X)
            # plt.savefig(os.path.join(self.out_dir, f'celltype_plots/power_law_{k}.png'))
            # plt.close()
            strata_X = X[:, self.strata_k == k]
            # strata_X_idf = X_idf[:, self.strata_k == k].flatten()
            strata_x_nonzero = strata_X[strata_X > 1].flatten()
            # strata_x_idf_nonzero = strata_X_idf[strata_X_idf > 0]
            try:
                hist, bins = np.histogram(strata_x_nonzero, bins=np.arange(0, strata_x_nonzero.max(), 1), density=True)
            except:
                continue
            ax.scatter(np.arange(len(hist)), np.sort(hist)[::-1])
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_title(f"{k}")

            # plt.scatter(np.arange(len(hist)), np.sort(hist)[::-1])
            # plt.yscale('log')
            # plt.xscale('log')
            # plt.xlabel("Frequency rank")
            # plt.ylabel("Density")
            # plt.title(f"Strata {k}")
            # plt.savefig(os.path.join(self.out_dir, f'celltype_plots/log_log_raw_counts_nonzero_{k}.png'))
            # plt.close()

            # # Calculate feature-wise frequencies
            # feature_frequencies = []
            # for cell in range(strata_X.shape[0]):
            #     unique, counts = np.unique(strata_X[cell, :], return_counts=True)
            #     feature_frequencies.extend(counts[unique > 0])  # Exclude counts of 0 and 1
            
            # # Sort frequencies in descending order
            # hist = np.sort(feature_frequencies)[::-1]
            # bins = np.arange(0, len(hist), 1)

            strata_adata = adata_use[:, self.strata_k == k].copy()
            sc.pp.calculate_qc_metrics(strata_adata, inplace=True)

            hist = np.sort(strata_adata.var['total_counts'].values)[::-1]
            bins = np.arange(0, len(hist), 1)
            #hist, bins = np.histogram(freqs, bins=np.arange(0, freqs.max(), 1), density=True)

            

            # plot log log plot to see if it is a power law
            #plt.scatter(bins, hist)
            # run a linear regression to see if it is a power law
            slope, intercept, r_value, p_value, std_err = linregress(np.log(np.arange(len(hist)) + 1), np.log(np.sort(hist)[::-1] + 1))
            r_values.append(r_value)
            linreg_pvals.append(p_value)
            print(f"Strata {k} slope: {slope}, intercept: {intercept}, r_value: {r_value}, p_value: {p_value}, std_err: {std_err}")
            # run ks test to see if it is a power law
            
            fit = powerlaw.Fit(hist, discrete=True)
            ks_stat, ks_p = kstest(hist, "powerlaw", args=(fit.power_law.alpha, fit.power_law.xmin), alternative='two-sided')
            k_stats.append(ks_stat)
            p_values.append(ks_p)
            print(f"Strata {k} ks_stat: {ks_stat}, ks_p: {ks_p}")
            exponential, expon_p = fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)
            log_normal, log_normal_p = fit.distribution_compare('power_law', 'lognormal', normalized_ratio=True)
            epon_vals.append(exponential)
            expon_ps.append(expon_p)
            lognorm_vals.append(log_normal)
            lognorm_ps.append(log_normal_p)
            print(f"Strata {k} exponential: {exponential}, p: {expon_p}, lognormal: {log_normal}, p: {log_normal_p}")
            # plt.yscale('log')
            # plt.xscale('log')
            # plt.xlabel("Frequency rank")
            # plt.ylabel("Density")
            # plt.title(f"Strata {k}, exponential: {exponential:.2f}, p: {expon_p:.2f}, lognormal: {log_normal:.2f}, p: {log_normal_p:.2f}")
            # plt.savefig(os.path.join(self.out_dir, f'celltype_plots/log_log_raw_counts_{k}.png'))
            # plt.close()

            # plot histogram of raw counts
            # sns.histplot(strata_X.flatten(), bins=100, kde=True, log_scale=True)
            # plt.title(f"Strata {k} raw counts")
            # plt.savefig(os.path.join(self.out_dir, f'celltype_plots/strata_{k}_raw_counts.png'))
            # plt.close()
            #sns.kdeplot(strata_x_nonzero, log_scale=True, ax=axs[0], bw_adjust=2, cut=0, color=colors[k])
            #sns.histplot(strata_x_nonzero, kde=False, log_scale=True, ax=axs[0], color=colors[k], discrete=True, element="step", fill=False)

            # plot histogram of idf counts
            # sns.histplot(strata_X_idf.flatten(), bins=100, kde=True, log_scale=True)
            # plt.title(f"Strata {k} idf counts")
            # plt.savefig(os.path.join(self.out_dir, f'celltype_plots/strata_{k}_idf_counts.png'))
            # plt.close()
            #sns.kdeplot(strata_x_idf_nonzero, log_scale=True, ax=axs[1], label=f"{k}", bw_adjust=2, cut=0, color=colors[k])
            #sns.histplot(strata_x_idf_nonzero, kde=False, log_scale=True, ax=axs[1], color=colors[k], discrete=True, element="step", fill=False)
        # axs[0].set_title("Raw counts")
        # axs[1].set_title("IDF counts")
        # leg = axs[1].legend(loc="upper left", bbox_to_anchor=(1,1), prop={'size': 6})
        # # get colors from matplotlib colors cmap
        

        # for i, j in enumerate(leg.legendHandles):
        #     j.set_color(colors[i])
        # plt.tight_layout()
        # 
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, f'celltype_plots/all_strata_power_law.png'))
        plt.close()
        # plot exponential and lognormal p values  
        plt.plot(epon_vals)
        plt.xlabel("Strata")
        plt.ylabel("likelihood ratio test statistic")
        plt.title("likelihood ratio test for exponential vs power law")
        plt.savefig(os.path.join(self.out_dir, f'celltype_plots/exponential_p_values.png'))
        plt.close()

        plt.plot(lognorm_vals)
        plt.xlabel("Strata")
        plt.ylabel("likelihood ratio test statistic")
        plt.title("likelihood ratio test for lognormal vs power law")
        plt.savefig(os.path.join(self.out_dir, f'celltype_plots/lognormal_p_values.png'))
        plt.close()

        # plot ks statistics
        plt.plot(k_stats)
        plt.xlabel("Strata")
        plt.ylabel("KS statistic")
        plt.title("KS statistic for power law fit")
        plt.savefig(os.path.join(self.out_dir, f'celltype_plots/ks_statistics.png'))
        plt.close()

        # plot r values
        linreg_pvals = np.array(linreg_pvals)
        # find indexes of strata where p value is less than 0.05
        significant_strata = np.where(linreg_pvals < 0.05)[0]
        plt.plot(np.array(r_values) ** 2)
        plt.xlabel("Strata")
        plt.ylabel("R squared value")
        plt.title("R squared value for power law fit")
        plt.savefig(os.path.join(self.out_dir, f'celltype_plots/r_squared_values.png'))
        plt.close()

        # plot p values
        plt.plot(linreg_pvals)
        plt.xlabel("Strata")
        plt.ylabel("p value")
        plt.title("p value for power law fit")
        plt.savefig(os.path.join(self.out_dir, f'celltype_plots/p_values.png'))
        plt.close()


        return np.array(hic.obsm['X_mds'])
