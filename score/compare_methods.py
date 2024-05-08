import os 
import json 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['figure.autolayout'] = False

from score.utils.utils import res_name_to_int, resolution_to_name


def symmetrize_res_matrix(heatmap):
    heatmap[np.tril_indices(heatmap.shape[0])] = 0
    heatmap = heatmap + heatmap.T
    return heatmap


def compare_methods_wilcoxon(dataset_name, res_name, main_metric='ari', palette='tab20', alpha=0.05, emb_plots=True, emb_rows=3, emb_cols=3, dpi=600, max_str_len=30, results_dir='results'):
    out_dir = os.path.join(results_dir, dataset_name, f"{main_metric}_compare")
    os.makedirs(out_dir, exist_ok=True)
    
    all_metrics = {}
    df = {'method': [], 'full_method': [], 'cluster_alg': [], 'metric': [], 'value': [], 'preprocessing': [], 'resolution': [], 'res_name': [], 'distance': [], 'n_strata': []}
    resolutions = os.listdir(os.path.join(results_dir, dataset_name))
    for res in resolutions:
        if res.lower().endswith('m') or res.lower().endswith('kb'):
            resolution_value = res_name_to_int(res)
            embedding_algs = os.listdir(os.path.join(results_dir, dataset_name, res))
            for method in embedding_algs:
                exp_dir = os.path.join(results_dir, dataset_name, res, method.lower())
                full_method_name = method + '@' + res
                strata_options = 'default'
                if ':' in method:  # preprocessing was used
                    method_name, preprocessing = method.split(':')
                    if '<' in preprocessing:
                        preprocessing, strata_options = preprocessing.split('<')
                else:
                    method_name = method
                    if '<' in method_name:
                        method_name, strata_options = method_name.split('<')
                    preprocessing = 'none'
                if strata_options != 'default':
                    try:
                        n_strata = int(strata_options)
                        distance = n_strata * resolution_value
                    except ValueError as e:
                        print(e)
                        distance = res_name_to_int(strata_options)
                        n_strata = int(distance / resolution_value)
                else:
                    distance = 'default'
                    n_strata = 'default'
                try:
                    with open(os.path.join(exp_dir, f"{method.lower()}.json"), 'r') as f:
                        metrics = json.load(f)
                        all_metrics[method] = metrics
                    for key in metrics.keys():
                        key_tmp = key.split('_')
                        if 'croc' in key:
                            continue
                        cluster_alg, metric_name = key_tmp[1], key_tmp[0]
                        for val in metrics[key]:
                            df['method'].append(method_name)
                            df['full_method'].append(full_method_name)
                            df['preprocessing'].append(preprocessing.replace(',','\n'))
                            df['res_name'].append(res)
                            df['resolution'].append(resolution_value)
                            df['distance'].append(distance)
                            df['n_strata'].append(n_strata)
                            df['cluster_alg'].append(cluster_alg)
                            df['metric'].append(metric_name)
                            try:
                                df['value'].append(val / 60 if metric_name == 'wall' else val)
                            except TypeError as e:
                                print(e)
                                df['value'].append(val)
                except FileNotFoundError as e:
                    print(e)

    df = pd.DataFrame.from_dict(df)

    df.to_csv(os.path.join(out_dir, 'results.csv'))

    # ttest_res = pg.pairwise_ttests(dv='value', between='full_method', alternative='greater', padjust='bonf', data=df[df['metric'] == main_metric])
    # ttest_res_preprocessing = pg.pairwise_ttests(dv='value', between='preprocessing', alternative='greater', padjust='bonf', data=df[df['metric'] == main_metric])
    # ttest_res_preprocessing.to_csv(os.path.join(out_dir, 'preprocessing_stats.csv'))

    results = {'method': [], 'full_method': [], 'preprocessing': [], 'resolution': [], 'res_name': [], 'distance': [], 'n_strata': [], 'best': [], 'p': [], 'effect_size': [], 'cluster_alg': [], 'power': [], 'U-val': [], 'results_dir': []}
    for cluster_alg in df['cluster_alg'].unique():
        print(cluster_alg)
        if cluster_alg not in ['k-means', 'gmm', 'agglomerative', 'leiden', 'louvain']:
            continue
        for res in resolutions:
            if res.lower().endswith('m') or res.lower().endswith('kb'):
                resolution_value = res_name_to_int(res)
                embedding_algs = os.listdir(os.path.join(results_dir, dataset_name, res))    
                for method in embedding_algs:
                    full_method_name = method + '@' + res
                    strata_options = 'default'
                    if ':' in method:  # preprocessing was used
                        method_name, preprocessing = method.split(':')
                        if '<' in preprocessing:
                            preprocessing, strata_options = preprocessing.split('<')
                    else:
                        method_name = method
                        if '<' in method_name:
                            method_name, strata_options = method_name.split('<')
                        preprocessing = 'none'
                    if strata_options != 'default':
                        try:  # distance specified in n_strata
                            n_strata = int(strata_options)
                            distance = n_strata * resolution_value
                        except ValueError as e:  # distance specified in kb or Mb
                            distance = res_name_to_int(strata_options)
                            n_strata = int(distance / resolution_value)
                    else:
                        distance = 'default'
                        n_strata = 'default'
                    df_tmp = df.copy()
                    df_tmp['preprocessing'] = df_tmp['preprocessing'].apply(lambda s: s.replace('\n', ','))
                    df_tmp['method_compare'] = 'B_other'
                    compare_mask = (df_tmp['method'] == method_name) & (df_tmp['preprocessing'] == preprocessing) & (df_tmp['n_strata'] == n_strata) & (df_tmp['resolution'] == resolution_value)
                    df_tmp.loc[compare_mask, 'method_compare'] = 'A_' + method + ':'  + preprocessing + '<' + str(n_strata) + '@' + res
                    in_df = df_tmp.loc[(df_tmp['metric'] == main_metric) & (df_tmp['cluster_alg'] == cluster_alg)]
                    compare_mask = (in_df['method'] == method_name) & (in_df['preprocessing'] == preprocessing) & (in_df['n_strata'] == n_strata) & (in_df['resolution'] == resolution_value)
                    best_val = in_df.loc[(in_df['method'] == method_name) & (in_df['preprocessing'] == preprocessing), 'value'].max()
                    # try:
                    #     ttest_res = pg.pairwise_ttests(dv='value', between='method_compare', alternative='greater', effsize='cohen', parametric=False, padjust='bonf', data=in_df)
                    #     results['effect_size'].append(float(ttest_res['cohen'][0]))
                    #     results['best'].append(best_val)
                    #     results['p'].append(float(ttest_res['p-unc'][0]))
                    #     results['cluster_alg'].append(cluster_alg)
                    #     results['U-val'].append(float(ttest_res['U-val'][0]))
                    #     results['method'].append(method_name)
                    #     results['full_method'].append(full_method_name)
                    #     results['preprocessing'].append(preprocessing.replace(',','\n'))
                    #     results['resolution'].append(resolution_value)
                    #     results['res_name'].append(res)
                    #     results['distance'].append(distance)
                    #     results['n_strata'].append(n_strata)
                    #     results['results_dir'].append(os.path.join(results_dir, dataset_name, res, method))

                    #     n_other = np.sum(in_df['method_compare'] == 'B_other')
                    #     n_method = len(in_df) - n_other
                    #     power = pg.power_ttest2n(nx=n_other, ny=n_method, d=results['effect_size'][-1], power=None, alpha=0.05, alternative='greater')
                    #     results['power'].append(power)
                    # except ValueError as e:
                    #     print(e, cluster_alg)
                    # except KeyError as e:
                    #     print(e, cluster_alg, method)
                    #     break
    results = pd.DataFrame.from_dict(results)
    agg_results = results.groupby(['full_method']).mean()
    print(agg_results)
    method_agg_results = results.groupby(['method']).mean()
    print(method_agg_results)
    preprocessing_agg_results = results.groupby(['preprocessing']).mean()
    print(preprocessing_agg_results)
    resolution_agg_results = results.groupby(['res_name']).mean()
    print(resolution_agg_results)
    distance_agg_results = results.groupby(['distance']).mean()
    print(distance_agg_results)

    results.to_csv(os.path.join(out_dir, 'test_statistics.csv'))
    shorten_names = lambda x: x.replace(',', '\n').replace(':', '\n')[:max_str_len] + '...' if len(x) > max_str_len else x.replace(',', '\n').replace(':', '\n')
    results['method'] = results['method'].apply(shorten_names)
    results['cluster_alg'] = results['cluster_alg'].apply(shorten_names)
    results['full_method'] = results['full_method'].apply(shorten_names)
    df['full_method'] = df['full_method'].apply(shorten_names)

    full_method_order = agg_results.index[agg_results['effect_size'].argsort()]
    full_method_order = [shorten_names(m) for m in full_method_order]
    method_order = agg_results.index[method_agg_results['effect_size'].argsort()]
    method_order = [shorten_names(m) for m in method_order]
    preprocessing_order = preprocessing_agg_results.index[preprocessing_agg_results['effect_size'].argsort()]


    try:
        sns.catplot(data=results, x='preprocessing', y='effect_size', kind='bar',
                    aspect=2, order=preprocessing_order, palette=palette, legend_out=True)
        plt.axhline(0, linestyle='--', c='k')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'compare_to_all_effect_size_preprocessing.png'))
        plt.savefig(os.path.join(out_dir, 'compare_to_all_effect_size_preprocessing.pdf'))
        plt.close()
    except Exception as e:
        print(e)

    try:
        sns.catplot(data=results, x='full_method', y='effect_size', kind='bar',
                    aspect=2, order=full_method_order, palette=palette, legend_out=True)
        plt.axhline(0, linestyle='--', c='k')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'compare_to_all_effect_size.png'))
        plt.savefig(os.path.join(out_dir, 'compare_to_all_effect_size.pdf'))
        plt.close()
    except Exception as e:
        print(e)

    try:
        res_order = resolution_agg_results.index[resolution_agg_results['effect_size'].argsort()]
        sns.catplot(data=results, x='res_name', y='effect_size', kind='bar',
                    aspect=2, palette=palette, legend_out=True, order=res_order)
        plt.axhline(0, linestyle='--', c='k')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'compare_to_all_effect_size_resolution.png'))
        plt.savefig(os.path.join(out_dir, 'compare_to_all_effect_size_resolution.pdf'))
        plt.close()
    except Exception as e:
        print(e)

    try:
        g = sns.catplot(data=results, x='method', y='p', kind='bar', hue='cluster_alg', 
                        aspect=3, order=method_order, palette='Set2', legend_out=True)
        plt.axhline(0, linestyle='--', c='k')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'compare_to_all_p.png'))
        plt.savefig(os.path.join(out_dir, 'compare_to_all_p.pdf'))
        ax = g.ax
        significant_bars = []
        for rect in ax.patches:
            height = rect.get_height()
            significant_bars.append(height <= alpha / len(agg_results))
        plt.close()
    except Exception as e:
        print(e)
    try:
        g = sns.catplot(data=results, x='method', y='effect_size', hue='cluster_alg', kind='bar', palette=palette,
                        aspect=3, order=method_order, legend_out=True)
        ax = g.ax
        plt.axhline(0, linestyle='--', c='k')
        plt.xticks(rotation=90)
        for rect, is_significant in zip(ax.patches, significant_bars):
            if is_significant:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width() / 2., height + height * 0.2,
                        '*', fontsize='xx-large', fontweight='extra bold',
                        ha='center', va='bottom', color='k')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'compare_to_all_effect_size_hue_alg.png'))
        plt.savefig(os.path.join(out_dir, 'compare_to_all_effect_size_hue_alg.pdf'))
        plt.close()
    except Exception as e:
        print(e)
    try:
        g = sns.catplot(data=results, x='method', y='p', kind='bar', hue='preprocessing', 
                        hue_order=preprocessing_order, aspect=3, order=method_order, palette=palette, legend_out=True)
        plt.axhline(0, linestyle='--', c='k')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'compare_to_all_p_hue_preprocessing.png'))
        plt.savefig(os.path.join(out_dir, 'compare_to_all_p_hue_preprocessing.pdf'))

        ax = g.ax
        significant_bars = []
        for rect in ax.patches:
            height = rect.get_height()
            significant_bars.append(height <= alpha)
        plt.close() 
    except Exception as e:
        print(e)           
    try:
        g = sns.catplot(data=results, x='method', y='effect_size', hue='preprocessing', palette=palette,
                        hue_order=preprocessing_order, kind='bar', aspect=3, order=method_order, legend_out=True)
        ax = g.ax
        plt.axhline(0, linestyle='--', c='k')
        plt.xticks(rotation=90)
        for rect, is_significant in zip(ax.patches, significant_bars):
            if is_significant:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width() / 2., height + height * 0.2,
                        '*', fontsize='xx-large', fontweight='extra bold',
                        ha='center', va='bottom', color='k')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'compare_to_all_effect_size_hue_preprocessing.png'))
        plt.savefig(os.path.join(out_dir, 'compare_to_all_effect_size_hue_preprocessing.pdf'))
        plt.close()
    except Exception as e:
        print(e)

    df.rename(columns={'value': main_metric}, inplace=True)
    metric_mask = df['metric'] == main_metric
    cluster_mask = df['cluster_alg'].isin(['k-means', 'agglomerative', 'gmm', 'leiden', 'louvain'])
    df['method'] = df['method'].apply(shorten_names)
    df['cluster_alg'] = df['cluster_alg'].apply(shorten_names)

    print(df)
    try:
        sns.catplot(data=df[df['metric'] == 'wall'], x='cluster_alg', y=main_metric, hue='full_method', 
                    kind='box', palette=palette, legend_out=True)
        plt.ylabel('wall time (s)')
        plt.savefig(os.path.join(out_dir, 'walltime.png'))
        plt.savefig(os.path.join(out_dir, 'walltime.pdf'))
        plt.close()
    except Exception as e:
        print(e)
    try:
        sns.catplot(data=df[metric_mask & cluster_mask], x='cluster_alg', y=main_metric, kind='box', 
                    aspect=3, palette=palette)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'accuracy_box_alg.png'))
        plt.savefig(os.path.join(out_dir, 'accuracy_box_alg.pdf'))
        plt.close()
    except Exception as e:
        print(e)
    try:
        sns.lineplot(data=df[metric_mask & cluster_mask], x='distance', y=main_metric, hue='method',
                    palette=palette)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'accuracy_distance.png'))
        plt.savefig(os.path.join(out_dir, 'accuracy_distance.pdf'))
        plt.close()
    except Exception as e:
        print(e)
    try:
        sns.lineplot(data=df[metric_mask & cluster_mask], x='resolution', y=main_metric, hue='method',
                    palette=palette)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'accuracy_resolution.png'))
        plt.savefig(os.path.join(out_dir, 'accuracy_resolution.pdf'))
        plt.close()
    except Exception as e:
        print(e)
    try:
        sns.catplot(data=df[metric_mask & cluster_mask], x='full_method', y=main_metric, kind='box', order=full_method_order,
                    aspect=2, palette=palette)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'accuracy_box_method.png'))
        plt.savefig(os.path.join(out_dir, 'accuracy_box_method.pdf'))
        plt.close()
    except Exception as e:
        print(e)
    try:
        sns.catplot(data=df[metric_mask & cluster_mask], x='preprocessing', y=main_metric, kind='box', order=preprocessing_order,
                    aspect=2, palette=palette)
        plt.tight_layout()
        plt.xticks(rotation=90)
        plt.savefig(os.path.join(out_dir, 'accuracy_box_preprocessing.png'))
        plt.savefig(os.path.join(out_dir, 'accuracy_box_preprocessing.pdf'))
        plt.close()
    except Exception as e:
        print(e)
    try:
        sns.catplot(data=df[metric_mask & cluster_mask], x='method', y=main_metric, hue='preprocessing', order=method_order, hue_order=preprocessing_order, palette=palette,
                    aspect=3,  kind='box', legend_out=True)
        plt.savefig(os.path.join(out_dir, 'accuracy_box_hue_by_preprocessing.png'))
        plt.savefig(os.path.join(out_dir, 'accuracy_box_hue_by_preprocessing.pdf'))
        plt.close()
    except Exception as e:
        print(e)

    full_method_order = agg_results.index[agg_results['effect_size'].argsort()]
    full_method_order = [s.split('@')[0] for s in full_method_order]

    if emb_plots:
        try:
            # plot distance and resolution embeddings
            for sweep in ['distance', 'resolution']:
                for emb_type in ['umap', 'tsne', 'pca']:
                    for method_name in results['method'].unique():
                        distances = sorted(results[sweep].unique())
                        if len(distances) == 1:
                            fig, axs = plt.subplots(1, 2)
                        else:
                            n_cols = int(len(distances) / 2) + 1
                            fig, axs = plt.subplots(2, n_cols, figsize=(n_cols * 3, 6))
                        for distance, ax in zip(distances, axs.reshape(-1)):
                            if distance == 'default':
                                dist_name = 'default'
                            else:
                                dist_name = resolution_to_name(distance)
                            method_dir = results[(results['method'] == method_name) & (results[sweep] == distance)]['results_dir'].values[0]
                            img = plt.imread(os.path.join(method_dir, f'celltype_plots/{emb_type}.png'))
                            ax.imshow(img)
                            ax.set_title(dist_name)
                        for ax in axs.reshape(-1):
                            ax.set_xticks([])
                            ax.set_yticks([])
                        plt.tight_layout()
                        plt.savefig(os.path.join(out_dir, f'{method_name}_{sweep}_{emb_type}.pdf'), dpi=dpi)
                        plt.savefig(os.path.join(out_dir, f'{method_name}_{sweep}_{emb_type}.png'), dpi=dpi)
                        plt.close()



            for cluster_alg in ['', 'k-means', 'agglomerative', 'gmm', 'leiden', 'louvain']:
                fig, axs = plt.subplots(emb_rows, emb_cols, figsize=(12, 8))
                order = agg_results.index[agg_results['best'].argsort()]
                for i, ax in enumerate(axs.reshape(-1)):
                    try:
                        algo = full_method_order[-(i + 1)]
                    except IndexError as e:
                        print(e)
                        ax.axis('off')
                        continue
                    exp_dir = os.path.join(results_dir, dataset_name, res_name, algo.lower())
                    try:
                        if cluster_alg == '':
                            img = plt.imread(os.path.join(exp_dir, 'celltype_plots/umap.png'))
                        else:
                            img = plt.imread(os.path.join(exp_dir, f'clustering/umap_{cluster_alg}.png'))
                        ax.imshow(img)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.set_title(shorten_names(algo))
                    except FileNotFoundError as e:
                        print(e)
                        ax.axis('off')
                        continue

                plt.tight_layout(h_pad=0.2, w_pad=0.01)
                plt.savefig(os.path.join(out_dir, f'umap_{cluster_alg}.pdf'), dpi=dpi)
                plt.savefig(os.path.join(out_dir, f'umap_{cluster_alg}.png'), dpi=dpi)
                plt.close()

                fig, axs = plt.subplots(emb_rows, emb_cols, figsize=(12, 8))

                for i, ax in enumerate(axs.reshape(-1)):
                    try:
                        algo = full_method_order[-(i + 1)]
                    except IndexError:
                        ax.axis('off')
                        continue
                    exp_dir = os.path.join(results_dir, dataset_name, res_name, algo.lower())
                    try:
                        if cluster_alg == '':
                            img = plt.imread(os.path.join(exp_dir, 'celltype_plots/pca.png'))
                        else:
                            img = plt.imread(os.path.join(exp_dir, f'clustering/pca_{cluster_alg}.png'))
                        ax.imshow(img)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.set_title(shorten_names(algo))
                    except FileNotFoundError as e:
                        print(e)
                        ax.axis('off')
                        continue

                plt.tight_layout(h_pad=0.2, w_pad=0.01)
                plt.savefig(os.path.join(out_dir, f'pca_{cluster_alg}.pdf'), dpi=dpi)
                plt.savefig(os.path.join(out_dir, f'pca_{cluster_alg}.png'), dpi=dpi)
                plt.close()

                fig, axs = plt.subplots(emb_rows, emb_cols, figsize=(12, 8))

                for i, ax in enumerate(axs.reshape(-1)):
                    try:
                        algo = full_method_order[-(i + 1)]
                    except IndexError:
                        ax.axis('off')
                        continue
                    exp_dir = os.path.join(results_dir, dataset_name, res_name, algo.lower())
                    try:
                        if cluster_alg == '':
                            img = plt.imread(os.path.join(exp_dir, 'celltype_plots/tsne.png'))
                        else:
                            img = plt.imread(os.path.join(exp_dir, f'clustering/tsne_{cluster_alg}.png'))
                        ax.imshow(img)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.set_title(shorten_names(algo))
                    except FileNotFoundError as e:
                        print(e)
                        ax.axis('off')
                        continue

                plt.tight_layout(h_pad=0.2, w_pad=0.01)
                plt.savefig(os.path.join(out_dir, f'tsne_{cluster_alg}.pdf'), dpi=dpi)
                plt.savefig(os.path.join(out_dir, f'tsne_{cluster_alg}.png'), dpi=dpi)
                plt.close()
        except Exception as e:
            print(e)
