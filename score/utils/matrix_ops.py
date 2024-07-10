import os
import cooler
import numpy as np
import pandas as pd
import networkx as nx

from tqdm import tqdm
from multiprocessing import Pool
from sklearn.preprocessing import normalize
from scipy.linalg import block_diag
from scipy.sparse import issparse, coo_matrix, csr_matrix
from matplotlib.colors import PowerNorm
from score.utils.utils import anchor_to_locus, anchor_list_to_dict, sorted_nicely


def rebin(a, shape):
        sh = shape[0], a.shape[0] // shape[0], shape[1], a.shape[1] // shape[1]
        return a.reshape(sh).sum(-1).sum(1)


def convolution(mat, kernel_shape=3):
    from scipy.signal import convolve2d
    conv = np.ones((kernel_shape, kernel_shape)) / (kernel_shape ** 2)
    mat = convolve2d(mat, conv, 'same')
    return mat


def random_walk(mat, random_walk_ratio=1.0, t=1):
    sm = np.sum(mat, axis=1)
    sm = np.where(sm == 0, 1, sm)
    sm = np.tile(sm, (len(mat), 1)).T
    walk = mat / sm
    for i in range(t):
        mat = random_walk_ratio * mat.dot(walk) + (1 - random_walk_ratio) * mat
    return mat


def OE_norm(mat):
    new_mat = np.zeros(mat.shape)
    averages = np.array([np.mean(mat[i:, :len(mat) - i]) for i in range(len(mat))])
    averages = np.where(averages == 0, 1, averages)
    for i in range(len(mat)):
        for j in range(len(mat)):
            d = abs(i - j)
            new_mat[i, j] = mat[i, j] / averages[d]
    return new_mat


def KR_norm(mat, maximum_error_rate=1e-3):
    # https://github.com/liu-bioinfo-lab/scHiCTools/blob/3d3379c2af82f0d8d17f66fa1e61a35a0f30074a/scHiCTools/load/processing_utils.py#L130
    bias = np.mean(mat) * maximum_error_rate
    # Remove all-zero rows and columns
    sm = np.sum(mat, axis=0)
    zeros = []
    for i in range(len(sm)):
        if sm[i] == 0:
            zeros.append(i)
    new_mat = np.delete(mat, zeros, axis=0)
    new_mat = np.delete(new_mat, zeros, axis=1)

    x = np.random.random(size=len(new_mat))
    k = 0
    while True:
        k += 1
        aa = np.diag(x).dot(new_mat) + np.diag(new_mat.dot(x))
        try:
            aa = np.linalg.inv(aa)
        except np.linalg.LinAlgError:
            new_x = np.zeros(x.shape)
            break
        bb = np.diag(x).dot(new_mat).dot(x) - np.ones(x.shape)
        delta = aa.dot(bb)
        new_x = x - delta

        max_error = np.max(np.abs(delta))
        # print(f'Iteration: {k}, Max Error: {max_error}')
        if max_error < bias:
            break
        else:
            x = new_x

    # Normalization
    dg = np.diag(new_x)
    new_mat = dg.dot(new_mat).dot(dg)

    # Put all-zero rows and columns back
    for zero in zeros:
        new_mat = np.insert(new_mat, zero, 0, axis=0)
        new_mat = np.insert(new_mat, zero, 0, axis=1)
    return new_mat


def VC_SQRT_norm(mat):
    sm = np.sum(mat, axis=0)
    sm = np.where(sm == 0, 1, sm)
    sm = np.sqrt(sm)
    sm_v = np.tile(sm, (len(sm), 1))
    sm_c = sm_v.T
    new_mat = mat / sm_c / sm_v
    return new_mat


def network_enhance(mat, kNN=20, iteration=1, alpha=0.9, **kwargs):
    argsort = np.argsort(-mat, axis=1)
    new_mat = np.zeros(mat.shape)
    for i in range(len(mat)):
        for j in range(kNN):
            pos = argsort[i, j]
            new_mat[i, pos] = mat[i, pos]

    sm = np.sum(new_mat, axis=1)
    sm = np.where(sm == 0, 1, sm)
    sm = np.tile(sm, (len(mat), 1)).T
    walk = new_mat / sm

    for k in range(iteration):
        new_mat = alpha * walk.T.dot(new_mat).dot(walk) + (1 - alpha) * new_mat
    return new_mat


def tfidf(X):
    idf = X.shape[0] / X.sum(axis=0)
    if issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf


def lsi(adata, n_components=20, use_highly_variable=None, **kwargs):
    # from https://scglue.readthedocs.io/en/latest/_modules/scglue/data.html#lsi
    from sklearn.utils.extmath import randomized_svd
    from sklearn.preprocessing import normalize
    from sklearn.decomposition import PCA
    if "random_state" not in kwargs:
        kwargs["random_state"] = 0  # Keep deterministic as the default behavior
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)
    X = np.nan_to_num(X)
    X_norm = normalize(X, norm="l2")
    X_lsi = PCA(n_components=n_components).fit_transform(X_norm)
    adata.obsm["X_lsi"] = X_lsi


def idf_inner_product(adata, n_components=32, use_highly_variable=None):
    from sklearn.manifold import MDS
    from sklearn.preprocessing import normalize
    from sklearn.decomposition import PCA
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = np.array(adata_use.X)
    idf = np.log(X.shape[0] / (np.sum(X > 0, axis=0) + 1))
    X = X * idf
    X = np.nan_to_num(X)
    X_norm = normalize(X, norm="l2")
    # inner = X_norm.dot(X_norm.T)
    # inner[inner > 1] = 1
    # inner[inner < -1] = -1
    # distance_mat = np.sqrt(2 - 2 * inner)
    # D = np.diag((sim - np.eye(sim.shape[0])) * np.ones_like(sim))
    # W = sim / np.sqrt(D[:, None] * D[None, :])
    # W = np.nan_to_num(W)
    # W
    #z = MDS(n_components=n_components, dissimilarity='precomputed').fit_transform(distance_mat)
    z = PCA(n_components=n_components).fit_transform(X_norm)
    #z = MDS(n_components=n_components).fit_transform(X_norm)
    adata.obsm["X_mds"] = z


def network_centrality(mat):
    G = nx.from_numpy_array(mat)
    centrality = nx.edge_betweenness_centrality(G, weight='weight')
    new_mat = np.zeros(mat.shape)
    for edge in centrality.keys():
        new_mat[edge[0], edge[1]] = centrality[edge]
    return new_mat


def graph_laplacian(mat):
    G = nx.from_numpy_array(mat)
    L = nx.normalized_laplacian_matrix(G).A
    return L


def graph_google(mat, alpha=0.85):
    G = nx.from_numpy_array(mat)
    L = nx.google_matrix(G, alpha=alpha)
    return L


def graph_modularity(mat):
    G = nx.from_numpy_array(mat)
    L = nx.modularity_matrix(G).A
    return L


def graph_resource_allocation(mat):
    G = nx.from_numpy_array(mat)
    resources = nx.resource_allocation_index(G)
    new_mat = np.zeros(mat.shape)
    for edge in resources:
        new_mat[edge[0], edge[1]] = edge[2]
    return new_mat


def graph_jaccard(mat):
    G = nx.from_numpy_array(mat)
    jaccard = nx.jaccard_coefficient(G)
    new_mat = np.zeros(mat.shape)
    for edge in jaccard:
        new_mat[edge[0], edge[1]] = edge[2]
    return new_mat

def graph_adamic_adar(mat):
    G = nx.from_numpy_array(mat)
    adamic_adar = nx.adamic_adar_index(G)
    new_mat = np.zeros(mat.shape)
    for edge in adamic_adar:
        new_mat[edge[0], edge[1]] = edge[2]
    return new_mat


def graph_preferential_attachment(mat):
    G = nx.from_numpy_array(mat)
    preferential_attachment = nx.preferential_attachment(G)
    new_mat = np.zeros(mat.shape)
    for edge in preferential_attachment:
        new_mat[edge[0], edge[1]] = edge[2]
    return new_mat


def graph_mst(mat):
    G = nx.from_numpy_array(mat)
    T = nx.minimum_spanning_tree(G)
    return nx.adjacency_matrix(T).A


def graph_coloring(mat):
    G = nx.from_numpy_array(mat)
    L = nx.line_graph(G)
    coloring = nx.coloring.greedy_color(L)
    new_mat = np.zeros(mat.shape)
    for edge in coloring.keys():
        new_mat[edge[0], edge[1]] = coloring[edge]
    return new_mat


def graph_min_edge_cut(mat):
    G = nx.from_numpy_array(mat)
    new_mat = np.zeros(mat.shape)
    for c in nx.connected_components(G):
        if len(c) > 1:
            G_c = G.subgraph(c).copy()
            edge_cut = nx.minimum_edge_cut(G_c)
            for edge in edge_cut:
                new_mat[edge[0], edge[1]] += 1
    return new_mat


def quantile_cutoff(mat, q=0.99):
    q_cutoff = np.quantile(mat, q)
    mat[mat < q_cutoff] = 0
    return mat


def get_processed_matrix(dataset, cell, cell_i, preprocessing, chr_only=None, rw_iter=1, rw_ratio=1.0):
    #loops = dataset.get_cell_pixels(cell)
    chr_list = list(pd.unique(dataset.anchor_list['chr']))
    chr_mats = []
    for chr_name in chr_list:
        if chr_only is not None and chr_name != chr_only:
            continue
        # chr_anchors = dataset.anchor_list.loc[dataset.anchor_list['chr'] == chr_name]
        # chr_anchors.reset_index(drop=True, inplace=True)
        # chr_anchor_dict = anchor_list_to_dict(chr_anchors['anchor'].values)
        # chr_contacts = loops.loc[loops['a1'].isin(chr_anchors['anchor']) & loops['a2'].isin(chr_anchors['anchor'])].copy()
        # if len(chr_contacts) > 0:
        #     chr_contacts['chr1'] = chr_name
        #     chr_contacts['chr2'] = chr_name

        #     rows = np.vectorize(anchor_to_locus(chr_anchor_dict))(
        #         chr_contacts['a1'].values)  # convert anchor names to row indices
        #     cols = np.vectorize(anchor_to_locus(chr_anchor_dict))(
        #         chr_contacts['a2'].values)  # convert anchor names to column indices
        #     matrix = coo_matrix((chr_contacts['obs'], (rows, cols)),
        #                 shape=(len(chr_anchors), len(chr_anchors)))
            
        #     mat = matrix.A
        try:
            c = cooler.Cooler(f"{dataset.scool_file}::/cells/{cell}")
        except Exception as e:
            try:
                new_cellname = cell.replace(dataset.res_name, f"comb.{dataset.res_name}")
                c = cooler.Cooler(f"{dataset.scool_file}::/cells/{new_cellname}")
            except Exception as e:
                try:
                    new_cellname = cell.replace(dataset.res_name, f"3C.{dataset.res_name}")
                    c = cooler.Cooler(f"{dataset.scool_file}::/cells/{new_cellname}")
                except Exception as e:
                    print(e)
                    pass
        try:
            mat = c.matrix(balance=False).fetch(chr_name)
        except IndexError as e:  # no interactions in this chrom
            chr_size = len(c.bins().fetch(chr_name))
            mat = np.zeros((chr_size, chr_size))
        if preprocessing is not None:
            for op in preprocessing:
                if op.lower() == 'convolution':
                    mat = convolution(mat)
                elif op.lower() == 'random_walk':
                    mat = random_walk(mat, t=int(rw_iter), random_walk_ratio=float(rw_ratio))
                elif op.lower() == 'vc_sqrt_norm' or op.lower() == 'vcsqrt_norm':
                    mat = VC_SQRT_norm(mat)
                elif op.lower() == 'oe_norm':
                    mat = OE_norm(mat)
                elif op.lower() == 'kr_norm':
                    mat = KR_norm(mat)
                elif op.lower() == 'network_enhance':
                    mat = network_enhance(mat)
                elif op.lower() == 'laplacian':
                    mat = graph_laplacian(mat)
                elif op.lower() == 'modularity':
                    mat = graph_modularity(mat)
                elif op.lower() == 'google':
                    mat = graph_google(mat)
                elif op.lower() == 'network_centrality':
                    mat = network_centrality(mat)
                elif op.lower() == 'resource_allocation':
                    mat = graph_resource_allocation(mat)
                elif op.lower() == 'jaccard':
                    mat = graph_jaccard(mat)
                elif op.lower() == 'adamic_adar':
                    mat = graph_adamic_adar(mat)
                elif op.lower() == 'preferential_attachment':
                    mat = graph_preferential_attachment(mat)
                elif op.lower() == 'coloring':
                    mat = graph_coloring(mat)
                elif op.lower() == 'edge_cut':
                    mat = graph_min_edge_cut(mat)
                elif op.lower() == 'mst':
                    mat = graph_mst(mat)
                elif 'quantile' in op.lower():
                    if '_' not in op:
                        q = 0.99
                    else:
                        q = float(op.split('_')[1])
                    mat = quantile_cutoff(mat, q=q)
                elif 'idf' in op.lower():
                    pass  # wait until all matrices are concatenated to do this
                else:
                    print('Unrecognized preprocessing op', op)
        # else:
        #     mat = np.zeros((len(chr_anchors), len(chr_anchors)))
        
        chr_mats.append(mat)
    mat = csr_matrix(block_diag(*chr_mats))
    
    return cell_i, cell, mat

def sum_sparse(m):
    # https://stackoverflow.com/questions/43427615/does-numpy-csr-matrix-mean-function-do-the-mean-on-all-of-the-matrix-how-can
    x = np.zeros(m[0].shape)
    for a in m:
        ri = np.repeat(np.arange(a.shape[0]),np.diff(a.indptr))
        x[ri,a.indices] += a.data
    return x

def viz_preprocessing(dataset, preprocessing, out_dir=None, gamma=0.3, small_window_sizes=[0.01, 0.05, 0.1, 0.25, 0.5], chr_name='chr11'):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.ndimage import median_filter
    from scipy.stats import rankdata
    
    print(preprocessing)
    if out_dir is None:
        sub_out_dir = 'raw'
        if preprocessing is not None:
            sub_out_dir = ','.join(preprocessing)
        out_dir = os.path.join('results', dataset.dataset_name, f'{dataset.res_name}_heatmaps', sub_out_dir)
    os.makedirs(out_dir, exist_ok=True)
    mats = {}
    results = []
    with Pool(8) as p:
        for cell_i, cell in enumerate(sorted(dataset.cell_list)):
            results.append(p.apply_async(get_processed_matrix, args=(dataset, cell, cell_i, preprocessing, chr_name)))
        for res in tqdm(results):
            cell_i, cell, mat = res.get(timeout=60)
            mats[cell] = mat
    if preprocessing is not None:
        if 'idf' in preprocessing:
            full_mat = []
            for cell_i, cell in enumerate(sorted(dataset.cell_list)):
                full_mat.append(mats[cell].A)
            full_mat = np.stack(full_mat, axis=0)
            print(full_mat.shape)
            flat_mat = full_mat.reshape(full_mat.shape[0], -1)
            idf = np.log((flat_mat.shape[0] + 1) / (np.sum(flat_mat > 0) + 1))
            flat_mat = flat_mat * idf
            flat_mat = np.nan_to_num(flat_mat)
            flat_mat = normalize(flat_mat, norm='l2')
            full_mat = flat_mat.reshape(full_mat.shape)
            mats = {}
            for cell_i, cell in enumerate(sorted(dataset.cell_list)):
                mats[cell] = csr_matrix(full_mat[cell_i])
    if preprocessing is not None:  # also get raw matrices to compare
        raw_mats = {}
        results = []
        with Pool(8) as p:
            for cell_i, cell in enumerate(sorted(dataset.cell_list)):
                results.append(p.apply_async(get_processed_matrix, args=(dataset, cell, cell_i, ['vc_sqrt_norm'], chr_name)))
            for res in tqdm(results):
                cell_i, cell, mat = res.get(timeout=60)
                raw_mats[cell] = mat
    else:
        raw_mats = mats
    full_mats = []
    merged_mats = {}
    merged_raw_mats = {}
    for c in dataset.reference['cluster'].unique():
        merged_mats[c] = []
        merged_raw_mats[c] = []
    for cell_i, cell in enumerate(sorted(dataset.cell_list)): 
        cluster = dataset.reference.loc[cell, 'cluster']
        merged_mats[cluster].append(mats[cell])
        merged_raw_mats[cluster].append(raw_mats[cell])
    for c in merged_mats.keys():
        print(c)
        mat = sum_sparse(merged_mats[c]) / len(merged_mats[c])
        raw_mat = sum_sparse(merged_raw_mats[c]) / len(merged_raw_mats[c])
        mat = mat + mat.T
        raw_mat = raw_mat + raw_mat.T
        ratio = (rankdata(mat).reshape(mat.shape) + 1) / (rankdata(raw_mat).reshape(mat.shape) + 1)
        #ratio = np.log2((mat + 1e-3) / (raw_mat + 1e-3))
        #ratio = median_filter(ratio, size=3)
        ratio_cmap_max = 1
        im = plt.imshow(ratio, cmap='bwr', vmin=-ratio_cmap_max, vmax=ratio_cmap_max)
        cbar = plt.colorbar(im)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{c.replace('/', '_')}_{chr_name}_ratio.png"))
        # remove axes and just save the image
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.set_title('')
        # remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        # remove colorbar
        cbar.remove()
        plt.axis('off')
        plt.savefig(os.path.join(out_dir, f"{c.replace('/', '_')}_{chr_name}_ratio_no_axes.png"), bbox_inches='tight',transparent=True, pad_inches=0)
        plt.close()
  
        im = plt.imshow(mat, cmap='Reds', norm=PowerNorm(gamma=gamma))
        cbar = plt.colorbar(im)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{c.replace('/', '_')}_{chr_name}.png"))
        # remove axes and just save the image
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.set_title('')
        # remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        # remove colorbar
        cbar.remove()
        plt.axis('off')
        plt.savefig(os.path.join(out_dir, f"{c.replace('/', '_')}_{chr_name}_no_axes.png"), bbox_inches='tight',transparent=True, pad_inches=0)
        plt.close()
        for small_window_frac in small_window_sizes:
            small_mat = mat
            small_ratio_mat = ratio
            small_window_size = int(small_window_frac * small_mat.shape[0])
            start_idx = (small_mat.shape[0] / 2)
            small_mat = small_mat[int(start_idx - small_window_size / 2):int(start_idx + small_window_size / 2), int(start_idx - small_window_size / 2):int(start_idx + small_window_size / 2)]
            small_ratio_mat = small_ratio_mat[int(start_idx - small_window_size / 2):int(start_idx + small_window_size / 2), int(start_idx - small_window_size / 2):int(start_idx + small_window_size / 2)]
            im = plt.imshow(small_ratio_mat, cmap='bwr', vmin=-ratio_cmap_max, vmax=ratio_cmap_max)
            cbar = plt.colorbar(im)
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{c.replace('/', '_')}_{chr_name}_{small_window_size}kb_ratio.png"))
            # remove axes and just save the image
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            ax.set_title('')
            # remove spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            # remove colorbar
            cbar.remove()
            plt.axis('off')
            plt.savefig(os.path.join(out_dir, f"{c.replace('/', '_')}_{chr_name}_{small_window_size}kb_ratio_no_axes.png"), bbox_inches='tight',transparent=True, pad_inches=0)
            plt.close()
            
            im = plt.imshow(small_mat, cmap='Reds', norm=PowerNorm(gamma=gamma))
            cbar = plt.colorbar(im)
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{c.replace('/', '_')}_{chr_name}_{small_window_size}kb.png"))
            # remove axes and just save the image
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            ax.set_title('')
            # remove spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            # remove colorbar
            cbar.remove()
            plt.axis('off')
            plt.savefig(os.path.join(out_dir, f"{c.replace('/', '_')}_{chr_name}_{small_window_size}kb_no_axes.png"), bbox_inches='tight',transparent=True, pad_inches=0)
            plt.close()
            #break
            #full_mats.append(mats[cell])


def get_loops_from_single_cell(dataset, cell_name, loops, preprocessing=None):
    c = cooler.Cooler(f"{dataset.scool_file}::/cells/{cell_name}")
    cell_counts = []
    total_reads = c.pixels()[:]['count'].sum()
    for loop in loops:
        chr1, start1, end1, chr2, start2, end2 = loop.split('_')
        a1 = f"{chr1}:{start1}-{end1}"
        a2 = f"{chr2}:{start2}-{end2}"
        loop_count = np.sum(c.matrix().fetch(a1, a2)) / total_reads
        cell_counts.append(loop_count)
    return cell_name, cell_counts


def get_loops_data(dataset, loop_file='/mnt/rds/genetics01/JinLab/dmp131/sc-thrillpark/sc-thrillpark/examples/glue/data/important_loops.txt'):
    loops_list = np.loadtxt(loop_file, dtype=str)
    loops = set()
    for l in loops_list:
        loops.add(l)
    loops = list(loops)
    print(loops)
    full_mats = {}
    results = []
    with Pool(10) as p:
        for cell_i, cell_name in enumerate(sorted(dataset.cell_list)):
            results.append(p.apply_async(get_loops_from_single_cell, args=(dataset, cell_name, loops)))
        for res in tqdm(results):
            cell_name, cell_counts = res.get(timeout=60)
            full_mats[cell_name] = cell_counts
    mats = []
    for cell_i, cell_name in enumerate(sorted(dataset.cell_list)): 
        mats.append(full_mats[cell_name])
    return np.array(mats)

def get_loop_strata_data(dataset, loop_file='/mnt/rds/genetics01/JinLab/dmp131/sc-thrillpark/sc-thrillpark/examples/glue/data/important_loops.txt'):
    loops_list = np.loadtxt(loop_file, dtype=str)
    loops = set()
    for l in loops_list:
        loops.add(l)
    loops = list(loops)
    print(loops)
    strata_n = []
    for l in loops:
        chr1, start1, end1, chr2, start2, end2 = l.split('_')
        strata_n.append(int(abs(start1 - start2) / 10000))
    full_mats = {}
    results = []
    with Pool(8) as p:
        for cell_i, cell_name in enumerate(sorted(dataset.cell_list)):
            results.append(p.apply_async(get_loops_from_single_cell, args=(dataset, cell_name, loops)))
        for res in tqdm(results):
            cell_name, cell_counts = res.get(timeout=60)
            full_mats[cell_name] = cell_counts
    mats = []
    for cell_i, cell_name in enumerate(sorted(dataset.cell_list)): 
        mats.append(full_mats[cell_name])
    return np.array(mats), strata_n
            

def get_flattened_matrices(dataset, n_strata, preprocessing=None, agg_fn=None, chr_only=None, rw_iter=1, rw_ratio=1.0, strata_offset=0):
    if strata_offset is None:
        strata_offset = 0
    mats = {}
    results = []
    with Pool(8) as p:
        for cell_i, cell in enumerate(sorted(dataset.cell_list)):
            results.append(p.apply_async(get_processed_matrix, args=(dataset, cell, cell_i, preprocessing, chr_only, rw_iter, rw_ratio)))
        for res in tqdm(results):
            cell_i, cell, mat = res.get(timeout=1000)
            if chr_only is not None:
                new_mat = []
                for i in range(n_strata + strata_offset):
                    new_mat.append(mat.diagonal(k=i))
                new_mat = np.concatenate(new_mat)
                mats[cell] = new_mat
            else:
                mats[cell] = mat

    full_mats = []
    for cell_i, cell in enumerate(sorted(dataset.cell_list)): 
        full_mats.append(mats[cell])

    if agg_fn is not None:  # aggregating to 1D
        strata_mask = np.zeros_like(full_mats[0].A)
        for k in range(n_strata):
            strata_mask += np.eye(strata_mask.shape[0], k=k, dtype=full_mats[0].dtype)
        mat = []
        for cell_i, cell in enumerate(sorted(dataset.cell_list)):
            tmp_mat = full_mats[cell_i].A
            tmp_mat[strata_mask == 0] = 0
            mat.append(agg_fn(tmp_mat, axis=0))
    else:  # unraveling 2D strata to 1D vector
        mat = []
        strata_k = []
        for cell_i, cell in enumerate(sorted(dataset.cell_list)):
            if chr_only is not None:
                counts = full_mats[cell_i]
                mat.append(counts)
            else:
                counts = []
                for k in range(n_strata + strata_offset):
                    new_strata = list(full_mats[cell_i].diagonal(k=k))
                    counts += new_strata
                    if cell_i == 0:
                        strata_k += [k] * len(new_strata)
                mat.append(counts)
    x = np.array(mat)
    return x