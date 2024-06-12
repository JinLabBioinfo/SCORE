import os
import math
import time
import pickle
import random
import cooler
import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from scipy.sparse import csr_matrix, coo_matrix, save_npz
from scipy.ndimage import rotate
from tqdm import tqdm
from multiprocessing import Pool

from score.utils.utils import anchor_to_locus, anchor_list_to_dict, sorted_nicely
from score.utils.matrix_ops import rebin, convolution, random_walk, OE_norm, VC_SQRT_norm, network_enhance, graph_google


class DataGenerator():
    def __init__(self, sparse_matrices, anchor_list, anchor_dict, data_dir, reference, full_reference=None, scool_file=None, scool_contents=None, assembly='hg19', res_name='1M', resolution=1000000,
                 n_clusters=None, class_names=None, dataset_name=None, active_regions=None, preprocessing=[],
                 batch_size=64, normalize=False, standardize=False, binarize=False, downsample=False,
                 simulate_from_bulk=False, bulk_cooler_dir=None, simulate_n=None, simulate_depth=None, real_n=0,
                 depth_norm=False, distance_norm=False, shuffle=True, use_raw_data=True, no_viz=False,
                 rotated_cells=False, resize_amount=1, limit2Mb=8, rotated_offset=0, depth=4, verbose=True, filter_cells_by_depth=True, ignore_chr_filter=False, color_config=None):
        # dictionary of rotated cells represented as sparse matrices
        self.sparse_matrices = sparse_matrices
        self.cell_anchor_vectors = None
        self.anchor_list = anchor_list  # DataFrame of anchors across whole genome
        self.resize_amount = resize_amount  # fraction to resize each cell
        # compute length of matrix diagonal
        # this becomes the width of the rotated matrix
        matrix_len = int(len(self.anchor_list) * self.resize_amount *
                         (math.sqrt(2) if rotated_cells else 1))
        # find next closest power of 2 so we can more easily define downsampling and upsampling
        # the matrix is padded with zeros to reach this new length
        next = matrix_len + (2 ** depth - matrix_len % (2 ** depth))
        # amount to pad matrix to reach new length
        self.matrix_pad = int(next - matrix_len)
        self.matrix_len = int(matrix_len)
        # dictionary m apping each anchor name to its genomic index
        self.anchor_dict = anchor_dict
        self.data_dir = data_dir  # directory containing all cell anchor to anchor files
        self.reference = reference  # DataFrame of cell, cluster, and depth information
        if full_reference is None:
            self.full_reference = reference
        else:
            self.full_reference = full_reference
        self.scool_file = scool_file
        self.scool_contents = scool_contents
        self.assembly = assembly
        self.res_name = res_name
        self.resolution = resolution
        self.verbose = verbose
        self.depth = depth  # depth of autoencoder model
        # optionally filter representations to only active regions (or any regions provided)
        self.active_regions = active_regions
        self.active_regions_to_idxs(self.active_regions)
        if dataset_name is None:
            self.dataset_name = 'pfc'
        else:
            self.dataset_name = dataset_name
        if filter_cells_by_depth and not simulate_from_bulk:
            # remove cells which do not fit sequencing depth criteria
            self.filter_cells(ignore_chr_filter=ignore_chr_filter)
        # list of preprocessing operations to apply to each matrix generated
        self.preprocessing = preprocessing
        self.max_read_depth = self.reference['depth'].max()
        self.batch_size = batch_size  # size of each batch during training
        # list of all cell file names
        self.cell_list = list(self.reference.index)
        self.n_cells = len(self.reference)  # total cells to train on
        if class_names is None:
            # list of cluster names
            self.classes = np.array(self.reference['cluster'].unique())
        else:
            self.classes = np.array(class_names)
        self.n_classes = len(self.classes)  # number of classes/clusters
        if n_clusters is None:
            self.n_clusters = self.n_classes
        else:
            self.n_clusters = n_clusters
        self.normalize = normalize  # option to normalize cell matrices to range 0-1
        self.standardize = standardize  # option to convert cells to mean zero unit variance
        self.binarize = binarize
        self.downsample = downsample
        self.simulate_from_bulk = simulate_from_bulk
        self.bulk_cooler_dir = bulk_cooler_dir
        self.simulate_n = simulate_n
        self.simulate_depth = simulate_depth
        self.real_n = real_n  # number of real cells in combined real/simulated datasets
        self.depth_norm = depth_norm  # option to normalize by read depth
        self.distance_norm = distance_norm
        self.shuffle = shuffle  # shuffle the order for generating matrices after each epoch
        # option to use either observed (raw) reads or bias-correction ratio values
        self.use_raw_data = use_raw_data
        self.no_viz = no_viz  # option to skip visualizations
        # option to use rotated matrix representation instead of compressed band
        self.rotated_cells = rotated_cells
        # height at which there are no more values, used at height of rotated matrix
        self.limit2Mb = limit2Mb
        self.rotated_offset = rotated_offset
        # shape of model input layer
        self.input_shape = (
            self.limit2Mb, self.matrix_len + self.matrix_pad, 1)

        # file mapping cluster names to RGB values (only for visualization)
        self.color_config = color_config

        self.example_cell = None

        self.epoch = 0

    def active_regions_to_idxs(self, active_regions, padding=100000):
        self.active_idxs = None
        if active_regions is not None:
            self.active_idxs = []
            for _, row in tqdm(self.active_regions.iterrows(), total=len(self.active_regions)):
                chrom = row['chr']
                chr_mask = self.anchor_list['chr'] == chrom
                pos_mask = (self.anchor_list['start'] >= (
                    row['start'] - padding)) & (self.anchor_list['end'] <= (row['end'] + padding))
                idx = list(self.anchor_list.loc[chr_mask & pos_mask].index)
                self.active_idxs += idx
            self.active_idxs = np.unique(self.active_idxs)
            matrix_len = len(self.active_idxs)
            next_pow2 = matrix_len + \
                (2 ** self.depth - matrix_len % (2 ** self.depth))
            self.matrix_pad = int(next_pow2 - matrix_len)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        if self.batch_size == -1:
            return 1
        else:
            return int(self.n_cells / self.batch_size) + 1

    def get_cell_by_index(self, index, downsample=False, preprocessing=None):
        """Generate one batch of data"""
        if index >= len(self.cell_list):
            index = index % len(self.cell_list)
        cell_name = self.cell_list[index]
        return self.__data_generation(cell_name, downsample=downsample, autoencoder_gen=False, preprocessing=preprocessing)

    def __data_generation(self, cell_name, downsample=False, autoencoder_gen=False, preprocessing=None):
        """Generates data containing batch_size samples"""
        cluster = str(self.reference.loc[cell_name, 'cluster'])
        depth = float(self.reference.loc[cell_name, 'depth'])
        try:
            batch = int(self.reference.loc[cell_name, 'batch'])
        except KeyError:
            batch = 1
        label = np.argwhere(self.classes == cluster)[0]
        cell = self.get_compressed_band_cell(cell_name, preprocessing=preprocessing)
        if cell is None:  # cell has no reads
            # it will be skipped when loading (need better solution when using generator)
            return cell, label, depth, batch
        else:
            cell = cell.A
            cell = np.expand_dims(cell, -1)  # add channel and batch dimension
            if downsample:
                new_x = self.downsample_mat(cell)
                cell_downsample = np.expand_dims(new_x, -1)
            else:
                cell_downsample = cell
            if self.binarize:
                cell = cell > 0
                cell = np.array(cell)
            if self.depth_norm:
                depth = int(self.reference.loc[cell_name, 'depth'])
                cell /= depth
            if self.standardize:
                cell = (cell - cell.mean()) / cell.std()
            if self.normalize:
                cell /= cell.max()
            if self.example_cell is None:
                self.example_cell = np.expand_dims(cell, 0)
            #cell = np.nan_to_num(cell)
            if autoencoder_gen:
                if downsample:
                    return cell_downsample, cell
                else:
                    return cell, cell
            else:
                return cell, label, depth, batch, cell_name

    def __getitem__(self, index):
        """Generate one batch of data"""
        batch_i = index
        x_batch = []
        y_batch = []
        if self.batch_size > 0:
            iter = range(batch_i * self.batch_size, batch_i *
                         self.batch_size + self.batch_size)
        else:
            iter = range(0, len(self.cell_list))
        for i in iter:
            if i >= len(self.cell_list):
                i = i % self.batch_size
            cell_name = self.cell_list[i]
            x, y = self.__data_generation(
                cell_name, downsample=self.downsample, autoencoder_gen=True)
            x_batch.append(x)
            y_batch.append(y)
        return np.array(x_batch), np.array(y_batch)

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:  # randomize cell order for next epoch
            random.shuffle(self.cell_list)

    def get_simulated_pixels(self, cell_name, bulk_loops, chr_offsets, anchor_ref, downsample_percent):
        weights = bulk_loops['obs'].values / bulk_loops['obs'].sum()
        # sample with replacement (why we reset obs to 1)
        loops = bulk_loops.sample(
            frac=downsample_percent, weights=weights, replace=True, ignore_index=True)
        anchor_dict = anchor_ref['start'].to_dict()
        anchor_chr_dict = anchor_ref['chr'].to_dict()
        loops = self.bin_pixels(loops, anchor_dict, anchor_chr_dict,
                                chr_offsets, self.resolution, key='count', use_chr_offset=True)
        loops.rename(columns={'bin1': 'a1', 'bin2': 'a2',
                     'count': 'obs'}, inplace=True)
        return loops, cell_name

    def get_cell_pixels(self, cell_name, alt_dir=None, return_cellname=False):
        if self.data_dir is None and alt_dir is None:  # read from .scool
            try:
                c = cooler.Cooler(f"{self.scool_file}::/cells/{cell_name}")
            except Exception as e:
                try:
                    new_cellname = cell_name.replace(self.res_name, f"comb.{self.res_name}")
                    c = cooler.Cooler(f"{self.scool_file}::/cells/{new_cellname}")
                except Exception as e:
                    try:
                        new_cellname = cell_name.replace(self.res_name, f"3C.{self.res_name}")
                        c = cooler.Cooler(f"{self.scool_file}::/cells/{new_cellname}")
                    except Exception as e:
                        print(e)
                        pass
            loops = c.pixels()[:]
            loops.rename(
                columns={'bin1_id': 'a1', 'bin2_id': 'a2', 'count': 'obs'}, inplace=True)
        else:
            use_dir = self.data_dir if alt_dir is None else alt_dir
            try:
                if self.res_name == 'frag':
                    frag_file = 'frag_loop.' + \
                        cell_name.split('.')[0] + '.cis.filter'
                    loops = pd.read_csv(os.path.join(use_dir, frag_file), delimiter='\t',
                                        names=['a1', 'a2', 'obs', 'exp'], usecols=['a1', 'a2', 'obs'])
                else:
                    loops = pd.read_csv(os.path.join(use_dir, cell_name), delimiter='\t',
                                        names=['a1', 'a2', 'obs', 'exp'], usecols=['a1', 'a2', 'obs'])
            except Exception as e:
                try:
                    loops = pd.read_csv(os.path.join(use_dir, cell_name), delimiter='\t',
                                        names=['a1', 'a2', 'obs'])
                except Exception as e:
                    loops = pd.read_csv(os.path.join(use_dir, cell_name.replace(self.res_name, f"3C.{self.res_name}")), delimiter='\t',
                                        names=['a1', 'a2', 'obs'])
                                        
            loops.dropna(inplace=True)
        loops['a1'] = loops['a1'].astype(str)
        loops['a2'] = loops['a2'].astype(str)
        if return_cellname:
            return loops, cell_name
        else:
            return loops

    def get_mitotic_reads(self, cell_name, anchor_chr_dict, anchor_dict, chr_offsets, mitotic_min, mitotic_max, local_min, local_max, from_frags):
        if from_frags:
            filename = 'frag_loop.' + cell_name.split('.')[0] + '.cis.filter'
            try:
                loops = self.get_cell_pixels(filename)
            except FileNotFoundError:
                loops = self.get_cell_pixels(filename.replace('.filter', ''))
            trans = self.get_cell_pixels(
                filename.replace('.cis.filter', '.trans'))
        else:
            loops = self.get_cell_pixels(cell_name)
            trans = loops  # ignore trans reads when not using frags
        loops['chr1'] = loops['a1'].map(anchor_chr_dict)
        loops['chr2'] = loops['a2'].map(anchor_chr_dict)
        loops['a1_start'] = loops['a1'].map(anchor_dict)
        loops['a1_start'] = loops.apply(
            lambda row: row['a1_start'] + chr_offsets[row['chr1']], axis=1)
        loops['a2_start'] = loops['a2'].map(anchor_dict)
        loops['a2_start'] = loops.apply(
            lambda row: row['a2_start'] + chr_offsets[row['chr2']], axis=1)
        loops['dist'] = (loops['a1_start'] - loops['a2_start']).abs()
        mitotic_mask = (loops['dist'] > mitotic_min) & (
            loops['dist'] < mitotic_max)
        local_mask = (loops['dist'] > local_min) & (loops['dist'] < local_max)
        self_mask = loops['dist'] <= local_min
        total = loops['obs'].sum() + trans['obs'].sum()
        mitotic = loops.loc[mitotic_mask, 'obs'].sum() / total
        local = loops.loc[local_mask, 'obs'].sum() / total
        trans = trans['obs'].sum()
        self_loops = loops.loc[self_mask, 'obs'].sum()
        return cell_name, total, mitotic, local, trans, self_loops

    def check_mitotic_cells(self, local_min=25000, local_max=2000000, mitotic_min=2000000, mitotic_max=12000000, out_dir='data/read_info', from_frags=True):
        import seaborn as sns
        import matplotlib.pyplot as plt
        os.makedirs(out_dir, exist_ok=True)
        if f'mitotic_{self.dataset_name}.csv' in os.listdir(out_dir):
            df = pd.read_csv(os.path.join(
                out_dir, f'mitotic_{self.dataset_name}.csv'))
        else:
            df = {'total': [], 'mitotic': [], 'local': [],
                  'trans': [], 'self': [], 'cell': [], 'cluster': []}
            anchor_ref = self.anchor_list.set_index('anchor')
            anchor_dict = anchor_ref['start'].to_dict()
            anchor_chr_dict = anchor_ref['chr'].to_dict()

            chr_offsets = {}
            current_offset = 0
            for chrom in sorted_nicely(pd.unique(anchor_ref['chr'])):
                chr_offsets[chrom] = current_offset
                chrom_mask = anchor_ref['chr'] == chrom
                current_offset += anchor_ref.loc[chrom_mask, 'end'].max()
            with Pool(6) as p:
                results = []
                for cell_name in tqdm(self.cell_list):
                    results.append(p.apply_async(self.get_mitotic_reads, args=(
                        cell_name, anchor_chr_dict, anchor_dict, chr_offsets, mitotic_min, mitotic_max, local_min, local_max, from_frags)))
                for res in tqdm(results):
                    cell_name, total, mitotic, local, trans, self_loops = res.get(
                        timeout=600)
                    df['total'].append(total)
                    df['mitotic'].append(mitotic)
                    df['local'].append(local)
                    df['trans'].append(trans)
                    df['self'].append(self_loops)
                    df['cell'].append(cell_name)
                    df['cluster'].append(
                        str(self.reference.loc[cell_name, 'cluster']))

            df = pd.DataFrame.from_dict(df)
            df.to_csv(os.path.join(
                out_dir, f'mitotic_{self.dataset_name}.csv'), index=False)
        df['cis'] = df['total'] - df['trans']
        df['cis_trans_ratio'] = df['cis'] / df['trans']
        df['log2_cis_trans_ratio'] = np.log2(df['cis_trans_ratio'])
        df['log10_self'] = np.log10(df['self'])
        df['self_ratio'] = df['self'] / df['total']

        if 'cluster' in df.columns:
            sns.scatterplot(data=df, x='mitotic', y='local', hue='cluster')
            plt.savefig(os.path.join(
                out_dir, f'mitotic_{self.dataset_name}.png'))
            plt.close()
        else:
            sns.scatterplot(data=df, x='mitotic', y='local', hue='local')
            plt.savefig(os.path.join(
                out_dir, f'mitotic_{self.dataset_name}.png'))
            plt.close()

        sns.histplot(data=df, x='self_ratio')
        plt.savefig(os.path.join(
            out_dir, f'self_ligation_ratio_{self.dataset_name}.png'))
        plt.close()

        df['cell'] = df['cell'].apply(lambda s: '.'.join(s.split('.')[:-1]))
        df.to_csv(os.path.join(
            out_dir, f'read_info_{self.dataset_name}'), sep='\t', index=False)
        return df

    def update_from_scool(self, scool_file, keep_depth=False):
        # updates dataset based on existing cooler file
        # used for simulated data where we need to update the dataset to a new simulated version
        print('Setting dataset to', scool_file)
        self.scool_file = scool_file
        content_of_scool = cooler.fileops.list_coolers(scool_file)
        self.scool_contents = content_of_scool
        c = cooler.Cooler(f"{scool_file}::{content_of_scool[0]}")
        anchor_list = c.bins()[:]
        anchor_list = anchor_list[['chrom', 'start', 'end']]
        anchor_list['anchor'] = np.arange(len(anchor_list))
        anchor_list['anchor'] = anchor_list['anchor'].astype(str)
        anchor_list.rename(columns={'chrom': 'chr'}, inplace=True)
        # convert to anchor --> index dictionary
        anchor_dict = anchor_list_to_dict(anchor_list['anchor'].values)
        self.anchor_list = anchor_list
        self.anchor_dict = anchor_dict
        if not keep_depth:
            for cell_name in tqdm(self.cell_list):
                self.reference.loc[cell_name, 'depth'] = self.get_cell_pixels(cell_name)[
                    'obs'].sum()
        matrix_len = int(len(self.anchor_list) * self.resize_amount *
                         (math.sqrt(2) if self.rotated_cells else 1))
        # find next closest power of 2 so we can more easily define downsampling and upsampling
        # the matrix is padded with zeros to reach this new length
        next = matrix_len + (2 ** self.depth - matrix_len % (2 ** self.depth))
        # amount to pad matrix to reach new length
        self.matrix_pad = int(next - matrix_len)
        self.matrix_len = int(matrix_len)

    def distance_summary(self, out_dir='data/dataset_summaries', distance_res=100000):
        import seaborn as sns
        import matplotlib.pyplot as plt
        os.makedirs(out_dir, exist_ok=True)
        chr_list = list(pd.unique(self.anchor_list['chr']))
        try:
            chr_list.remove('chrM')
        except Exception as e:
            pass
        chr_lengths = {}
        chr_anchor_dfs = {}
        chr_anchor_dicts = {}
        read_counts = []
        anchor_ref = self.anchor_list.set_index('anchor')
        chr_dict = anchor_ref['chr'].to_dict()
        pos_dict = anchor_ref['start'].to_dict()
        res = []
        for cell_name in tqdm(self.cell_list):
            loops = self.get_cell_pixels(cell_name)
            loops['pos1'] = loops['a1'].map(pos_dict)
            loops['pos2'] = loops['a2'].map(pos_dict)
            loops['pos1'] = loops['pos1'].astype(int)
            loops['pos2'] = loops['pos2'].astype(int)
            loops['dist'] = (loops['pos1'] - loops['pos2']).abs() / distance_res
            loops['dist'] = loops['dist'].astype(int)
            #loops['freq'] = loops['obs'] / loops['obs'].sum()
            loops = loops[loops['dist'] < 1000]  # 100Mb cutoff
            # compute per distance avg frequency
            dist_freq = loops.groupby('dist')[['obs']].sum()
            dist_freq['freq'] = dist_freq['obs'] / dist_freq['obs'].sum()
            dist_freq = dist_freq.reset_index()
            #dist_freq['cell'] = cell_name
            for col in self.reference.columns:
                dist_freq[col] = self.reference.loc[cell_name, col]
            res.append(dist_freq)
        res = pd.concat(res).reset_index(drop=True)
        res['log_dist'] = np.log1p(res['dist'])
        res['log_freq'] = np.log1p(res['freq'])
        res.to_csv(os.path.join(out_dir, f'{self.dataset_name}_dist.tsv'), sep='\t', index=False)

        fig, ax = plt.subplots(figsize=(4, 8))
        sns.lineplot(data=res, x='dist', y='freq', hue='cluster', ax=ax)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.savefig(os.path.join(out_dir, f'{self.dataset_name}_dist.png'))
        plt.savefig(os.path.join(out_dir, f'{self.dataset_name}_dist.pdf'))
        plt.close()
        
        return res



    def get_avg_cis_reads(self, out_dir='data/dataset_summaries'):
        os.makedirs(out_dir, exist_ok=True)
        chr_list = list(pd.unique(self.anchor_list['chr']))
        try:
            chr_list.remove('chrM')
        except Exception as e:
            pass
        chr_lengths = {}
        chr_anchor_dfs = {}
        chr_anchor_dicts = {}
        read_counts = []
        for cell_name in tqdm(self.cell_list):
            loops = self.get_cell_pixels(cell_name)
            cis_reads = 0
            for chr_name in chr_list:
                if chr_name in chr_anchor_dicts.keys():  # reload chr data to save time
                    chr_anchor_dict = chr_anchor_dicts[chr_name]
                    chr_anchors = chr_anchor_dfs[chr_name]
                    chr_lengths[chr_name] = len(chr_anchors)
                else:
                    chr_anchors = self.anchor_list.loc[self.anchor_list['chr'] == chr_name]
                    chr_anchors.reset_index(drop=True, inplace=True)
                    chr_anchor_dfs[chr_name] = chr_anchors
                    chr_anchor_dict = anchor_list_to_dict(
                        chr_anchors['anchor'].values)
                    chr_anchor_dicts[chr_name] = chr_anchor_dict
                    chr_lengths[chr_name] = len(chr_anchors)
                chr_contacts = loops[loops['a1'].isin(
                    chr_anchors['anchor']) & loops['a2'].isin(chr_anchors['anchor'])]
                # remove any contacts a2 a1 where a1 a2 is already present
                chr_contacts = chr_contacts[chr_contacts['a1'] <= chr_contacts['a2']]
                cis_reads += chr_contacts['obs'].sum()
            read_counts.append(cis_reads)
        print('Avg:', np.mean(read_counts))
        print('Median:', np.median(read_counts))
        print('Std:', np.std(read_counts))
        # ref = pd.DataFrame.from_dict({'cell': self.cell_list, 'depth': read_counts})
        new_ref = self.reference.copy()
        cells = np.array(self.cell_list)
        new_ref.loc[cells, 'depth'] = read_counts
        new_ref.to_csv(os.path.join(out_dir, f'{self.dataset_name}_depth.tsv'), sep='\t', index=True)


    def insulation_vectors(self, out_dir='data/dataset_summaries', loops_offset=0, window_size=9):
        if f'{self.dataset_name}_insulation_{self.res_name}.npy' in os.listdir(out_dir):
            boundaries = np.load(os.path.join(out_dir, f'{self.dataset_name}_insulation_{self.res_name}.npy'))
            boundaries = np.nan_to_num(boundaries)
        else:
            from cooltools import insulation
            os.makedirs(out_dir, exist_ok=True)
            chr_list = list(pd.unique(self.anchor_list['chr']))
            try:
                chr_list.remove('chrM')
            except Exception as e:
                pass
            cool_files = cooler.fileops.list_scool_cells(self.scool_file)
            cool_files = [f"{self.scool_file}::{cell}" for cell in cool_files]
            cooler.merge_coolers('tmp_bulk.cool', cool_files, mergebuf=40000000)
            bulk = cooler.Cooler('tmp_bulk.cool')
            cooler.balance_cooler(bulk, cis_only=True, store=True)
            bins = bulk.bins()[:]
            boundaries = []
            for cell_name in tqdm(self.cell_list):
                dummy_bins = []
                for chrom in bulk.chromnames:
                    chrom_bins = bins[bins['chrom'] == chrom].copy()
                    chrom_bins['start'] = np.arange(0, len(chrom_bins) * int(self.resolution), int(self.resolution))
                    chrom_bins['end'] = chrom_bins['start'] + int(self.resolution)
                    dummy_bins.append(chrom_bins)
                dummy_bins = pd.concat(dummy_bins).reset_index(drop=True)
                
                dummy_pixels = self.get_cell_pixels(cell_name)
                dummy_pixels.rename(columns={'a1': 'bin1_id', 'a2': 'bin2_id', 'obs': 'count'}, inplace=True)
                dummy_pixels['bin1_id'] = dummy_pixels['bin1_id'].astype(int)
                dummy_pixels['bin2_id'] = dummy_pixels['bin2_id'].astype(int)
                dummy_pixels['bin2_id'] = dummy_pixels['bin2_id'] - loops_offset
                cooler.create_cooler(f'tmp_dummy.cool', dummy_bins[['chrom', 'start', 'end', 'weight']], dummy_pixels)
                dummy_cool = cooler.Cooler(f'tmp_dummy.cool')
                insulation_table = insulation(dummy_cool, [self.resolution * window_size], verbose=False, ignore_diags=0)
                score = insulation_table[f'log2_insulation_score_{self.resolution * window_size}'].values
                boundaries.append(score)
            os.remove('tmp_dummy.cool')
            os.remove('tmp_bulk.cool')
            boundaries = np.array(boundaries)
            boundaries = np.nan_to_num(boundaries)
            np.save(os.path.join(out_dir, f'{self.dataset_name}_insulation_{self.res_name}.npy'), boundaries)
        return boundaries


    def compartment_vectors(self, out_dir='data/dataset_summaries', loops_offset=0):
        if f'{self.dataset_name}_compartments.npy' in os.listdir(out_dir):
            compartments = np.load(os.path.join(out_dir, f'{self.dataset_name}_compartments.npy'))
            compartments = np.nan_to_num(compartments)
        else:
            import bioframe
            import cooltools
            os.makedirs(out_dir, exist_ok=True)
            chr_list = list(pd.unique(self.anchor_list['chr']))
            try:
                chr_list.remove('chrM')
            except Exception as e:
                pass
            c = cooler.Cooler(f"{self.scool_file}::/cells/{self.cell_list[0]}")
            bins = c.bins()[:]
            cool_files = cooler.fileops.list_scool_cells(self.scool_file)
            cool_files = [f"{self.scool_file}::{cell}" for cell in cool_files]
            cooler.merge_coolers('tmp_bulk.cool', cool_files, mergebuf=40000000)
            bulk = cooler.Cooler('tmp_bulk.cool')
            cooler.balance_cooler(bulk, cis_only=True, store=True)
            bins = bulk.bins()[:]
            genome = bioframe.load_fasta('/mnt/jinstore/JinLab01/LAB/Genome_references/cellranger_atac_hg19ref/hg19/fasta/genome.fa')
            gc_cov = bioframe.frac_gc(bins[['chrom', 'start', 'end']], genome)
            print(gc_cov)
            compartments = []
            for cell_name in tqdm(self.cell_list):
                c = cooler.Cooler(f"{self.scool_file}::/cells/{cell_name}")
                dummy_bins = []
                for chrom in c.chromnames:
                    chrom_bins = bins[bins['chrom'] == chrom].copy()
                    chrom_bins['start'] = np.arange(0, len(chrom_bins) * int(self.resolution), int(self.resolution))
                    chrom_bins['end'] = chrom_bins['start'] + int(self.resolution)
                    dummy_bins.append(chrom_bins)
                dummy_bins = pd.concat(dummy_bins)
                gc_cov['start'] = dummy_bins['start']
                gc_cov['end'] = dummy_bins['end']
                
                dummy_pixels = c.pixels()[:]
                dummy_pixels['bin2_id'] = dummy_pixels['bin2_id'] - loops_offset
                cooler.create_cooler(f'tmp_dummy.cool', dummy_bins[['chrom', 'start', 'end', 'weight']], dummy_pixels)
                dummy_cool = cooler.Cooler(f'tmp_dummy.cool')
                #cooler.balance_cooler(c, cis_only=True, store=True)
                view_df = pd.DataFrame({'chrom': dummy_cool.chromnames,
                                'start': 0,
                                'end': dummy_cool.chromsizes.values,
                                'name': dummy_cool.chromnames}
                            )
                # obtain first 3 eigenvectors
                cis_eigs = cooltools.eigs_cis(
                                        dummy_cool,
                                        gc_cov,
                                        n_eigs=3,
                                        view_df=view_df
                                        )
                eigenvector_track = cis_eigs[1][['chrom','start','end','E1', 'E2']]
                eigs = eigenvector_track['E1'].values
                compartments.append(eigs)
            os.remove('tmp_dummy.cool')
            os.remove('tmp_bulk.cool')
            compartments = np.array(compartments)
            compartments = np.nan_to_num(compartments)
            np.save(os.path.join(out_dir, f'{self.dataset_name}_compartments.npy'), compartments)
        return compartments
    
    def write_scool(self, out_file, simulate=False, append_simulated=False, n_proc=1, downsample_frac=None):
        if simulate:
            coolers = []
            bulk_loops = []
            downsample_fracs = []
            cool_files = os.listdir(self.bulk_cooler_dir)
            for file in cool_files:
                if file.endswith('.mcool'):
                    c = cooler.Cooler(os.path.join(
                        self.bulk_cooler_dir, file + '::resolutions/10000'))
                else:
                    c = cooler.Cooler(os.path.join(self.bulk_cooler_dir, file))
                coolers.append(c)
                pixels = c.pixels()[:]
                pixels.rename(
                    columns={'bin1_id': 'a1', 'bin2_id': 'a2', 'count': 'obs'}, inplace=True)
                pixels.dropna(inplace=True)
                bulk_loops.append(pixels)
                downsample_fracs.append(
                    self.simulate_depth / pixels['obs'].sum())
        if simulate and not append_simulated:
            chr_offsets, uniform_bins = self.get_uniform_bins(self.resolution)
            uniform_bins.to_csv(
                f'bins_{self.dataset_name}.tsv', sep='\t', index=False)
            bins = pd.DataFrame()
            bins['chrom'] = uniform_bins['chr']
            bins['start'] = uniform_bins['start']
            bins['end'] = uniform_bins['end']
            bins['weight'] = 1

            anchor_ref = self.anchor_list.set_index(
                'anchor').dropna().reset_index(drop=True)
        else:
            bins = pd.DataFrame()
            bins['chrom'] = self.anchor_list['chr']
            bins['start'] = self.anchor_list['start']
            bins['end'] = self.anchor_list['end']
            bins['weight'] = 1
            if append_simulated:
                chr_offsets = {}
                prev_offset = 0
                for chr_name in sorted_nicely(bins['chrom'].unique()):
                    chr_offsets[chr_name] = prev_offset
                    prev_offset += int(
                        self.anchor_list.loc[self.anchor_list['chr'] == chr_name, 'end'].max() / self.resolution)
                print(chr_offsets)
                anchor_ref = self.anchor_list.set_index(
                    'anchor').dropna().reset_index(drop=True)

        loop_list = []
        cell_list = []
        simulated_i = 0
        def generate_pixels(chunksize=1000):
            name_pixel_dict = {}
            get_row_indices = np.vectorize(anchor_to_locus(self.anchor_dict))
            for cell_i, cell_name in tqdm(enumerate(sorted(self.cell_list)), total=len(self.cell_list)):
                if simulate and self.reference.loc[cell_name, 'type'] == 'simulated':
                    bulk_loops_idx = int(simulated_i / self.simulate_n)
                    simulated_i += 1
                    loops, _ = self.get_simulated_pixels(
                        cell_name, bulk_loops[bulk_loops_idx], chr_offsets, anchor_ref, downsample_fracs[bulk_loops_idx])
                else:
                    try:
                        loops = self.get_cell_pixels(cell_name)
                    except FileNotFoundError:
                        print(f'File not found for {cell_name}, skipping...')
                        continue
                    except Exception as e:
                        print(e, cell_name)
                        continue
                if downsample_frac is not None:  # downsample each cell by some percentage
                    weights = loops['obs']
                    #weights.loc[weights <= 0] = 0
                    total = loops['obs'].sum()
                    loops['new_count'] = 1
                    loops = loops.sample(
                        n=int(total * downsample_frac), replace=True, weights=weights)
                    loops = loops[['a1', 'a2', 'new_count']]
                    loops = loops.groupby(['a1', 'a2']).sum().reset_index()
                    loops.rename(columns={'new_count': 'obs'}, inplace=True)

                #loop_list.append(loops)
                #cell_list.append(cell_name)

            #for cell_name, loops in tqdm(sorted(zip(cell_list, loop_list), key=lambda x: x[0])):
                if simulate and not append_simulated:
                    self.anchor_dict = anchor_list_to_dict(
                        uniform_bins['anchor'].values)
                    a1_mask = loops['a1'].isin(uniform_bins['anchor'])
                    a2_mask = loops['a2'].isin(uniform_bins['anchor'])
                    loops = loops[a1_mask & a2_mask]
                else:
                    a1_mask = loops['a1'].isin(self.anchor_list['anchor'])
                    a2_mask = loops['a2'].isin(self.anchor_list['anchor'])
                    loops = loops[a1_mask & a2_mask]
                if len(loops) == 0:  # cell has no reads
                    if self.verbose:
                        print('No reads, skipping...')
                        continue
                rows = get_row_indices(
                    loops['a1'].values)  # convert anchor names to row indices
                cols = get_row_indices(
                    loops['a2'].values)  # convert anchor names to column indices
                pixels = pd.DataFrame()
                pixels['bin1_id'] = rows
                pixels['bin2_id'] = cols
                # if 'pfc' in self.dataset_name:
                #     pixels['bin2_id'] = pixels['bin2_id'] - 1
                bad_loops_mask = pixels.apply(
                    lambda row: row['bin1_id'] > row['bin2_id'], axis=1)
                bad_pixels = pixels[bad_loops_mask].to_numpy()
                pixels.loc[bad_loops_mask, 'bin1_id'] = bad_pixels[..., 1]
                pixels.loc[bad_loops_mask, 'bin2_id'] = bad_pixels[..., 0]
                pixels['count'] = loops['obs']
                pixels = pixels.groupby(['bin1_id', 'bin2_id'])['count'].max().reset_index()
                try:
                    # remove any unnecessary zeros
                    pixels = pixels[pixels['count'] > 0].reset_index()
                except KeyError:
                    print(pixels)
                name_pixel_dict[cell_name] = pixels
                if len(name_pixel_dict) >= chunksize:
                    yield name_pixel_dict
                    name_pixel_dict = {}
            yield name_pixel_dict
        if downsample_frac is not None:
            out_file = out_file.replace(
                f'{self.res_name}.scool', f'{downsample_frac:.2f}_{self.res_name}.scool')
        try:
            os.remove(out_file)
        except FileNotFoundError:
            pass
        for i, name_pixel_dict in enumerate(generate_pixels()):
            print(i)
            cooler.create_scool(out_file, bins, name_pixel_dict, mode='a')

    def write_binned_scool(self, out_file, factor, new_res_name):
        bins = None 
        name_pixel_dict = {}
        for cool_content in tqdm(self.scool_contents):
            cellname = cool_content.split('/')[-1]
            base_uri = f"{self.scool_file}::{cool_content}"
            out_uri = f"tmp.cool"
            c = cooler.Cooler(base_uri)
            cooler.coarsen_cooler(base_uri, out_uri, factor=factor, chunksize=10000000)
            c = cooler.Cooler(out_uri)
            if bins is None:
                bins = c.bins()[:]
            name_pixel_dict[cellname.replace(self.res_name, new_res_name)] = c.pixels()[:]
        os.remove(out_uri)
        cooler.create_scool(out_file, bins, name_pixel_dict)

    def write_pseudo_bulk_coolers(self, out_dir='data/coolers'):
        os.makedirs(out_dir, exist_ok=True)
        for celltype in self.reference['cluster'].unique():
            print(celltype)
            cools = []
            for cell_name in tqdm(self.cell_list):
                if self.reference.loc[cell_name, 'cluster'] == celltype:
                    cools.append(f"{self.scool_file}::/cells/{cell_name}")
            cooler.merge_coolers(os.path.join(out_dir, f'{celltype.replace("/", "_")}_{self.res_name}.cool'), cools, mergebuf=1000000)
            c = cooler.Cooler(os.path.join(out_dir, f'{celltype.replace("/", "_")}_{self.res_name}.cool'))
            cooler.balance_cooler(c, chunksize=10000000, cis_only=True, store=True)

    def write_3dvi_ref(self, out_dir):
        cell_types = []
        cell_names = []
        depths = []
        batches = []
        sparsity = []
        for cell_i, cell_name in tqdm(enumerate(self.cell_list)):
            cluster = str(self.reference.loc[cell_name, 'cluster'])
            batch = str(self.reference.loc[cell_name, 'batch'])
            cell_names.append(cell_name + '.int.bed')

            anchor_to_anchor = self.get_cell_pixels(cell_name)
            anchor_to_anchor.rename(
                columns={'a1': 'pos1', 'a2': 'pos2', 'obs': 'count'}, inplace=True)

            cell_types.append(cluster)
            batches.append(batch)
            sparsity.append(len(anchor_to_anchor))
            depths.append(anchor_to_anchor['count'].sum())

        label_info = {
            'name': cell_names,
            'batch': batches,
            'cell_type': cell_types,
            'depth': depths,
            'sparsity': sparsity
        }
        ref = pd.DataFrame.from_dict(label_info)
        os.makedirs(out_dir, exist_ok=True)
        ref.to_csv(os.path.join(out_dir, 'data_summary.txt'),
                   sep='\t', index=False)

    def write_lda_matrix(self, cell_name, chr_list, chr_anchor_dicts, chr_anchor_dfs, norm=True, rw=False, keep_trans=False):
        anchor_to_anchor = self.get_cell_pixels(cell_name)
        anchor_to_anchor.rename(columns={'obs': 'count'}, inplace=True)
        chr_map = self.anchor_list.set_index('anchor')['chr'].to_dict()
        anchor_to_anchor['chr1'] = anchor_to_anchor['a1'].map(chr_map)
        anchor_to_anchor['chr2'] = anchor_to_anchor['a2'].map(chr_map)
        if not keep_trans:
            anchor_to_anchor = anchor_to_anchor[anchor_to_anchor['chr1'] == anchor_to_anchor['chr2']].reset_index(drop=True)
        anchor_to_anchor['a1'] = anchor_to_anchor['a1'].apply(lambda s: s.split('_')[-1])
        anchor_to_anchor['a2'] = anchor_to_anchor['a2'].apply(lambda s: s.split('_')[-1])
        lda_df = anchor_to_anchor[['a1', 'a2', 'count', 'chr1', 'chr2']]
        return cell_name, lda_df

    def write_lda_data(self, out_dir, norm=True, rw=False):
        cell_types = []
        cell_names = []
        depths = []
        batches = []
        chr_list = list(pd.unique(self.anchor_list['chr']))
        chr_anchor_dfs = {}
        chr_anchor_dicts = {}
        results = []
        for cell_name in tqdm(sorted(self.cell_list)):
            cluster = str(self.reference.loc[cell_name, 'cluster'])
            depth = float(self.reference.loc[cell_name, 'depth'])
            batch = str(self.reference.loc[cell_name, 'batch'])
            cell_name, lda_df = self.write_lda_matrix(cell_name, chr_list, chr_anchor_dicts, chr_anchor_dfs, norm, rw)
            cell_types.append(cluster.replace(' ', '_'))
            depths.append(depth)
            batches.append(batch)
            cell_names.append(cell_name)
            lda_df.to_csv(os.path.join(out_dir, '%s.matrix' %
                              cell_name), sep='\t', index=False, header=False)

        label_info = {
            'cell': cell_names,
            'cell type': cell_types,
            'batch': ['batch_%s' % b for b in batches],
        }
        label_df = pd.DataFrame.from_dict(label_info)
        label_df.to_csv(os.path.join(out_dir, 'cell_labels.tsv'),
                        sep='\t', header=False, index=False)

    def run_preprocessing_on_sparse(self, loops, operations, chr_anchor_map, chr_anchor_dict_map,q=0.8):
        new_loops = []
        for chrom in sorted_nicely(self.anchor_list['chr'].unique()):
            chr_anchors = chr_anchor_map[chrom]
            min_anchor = chr_anchors['anchor'].astype(int).min()
            chr_anchor_dict = chr_anchor_dict_map[chrom]
            chrom_mask = loops['a1'].isin(chr_anchors['anchor']) & loops['a2'].isin(chr_anchors['anchor'])
            chrom_loops = loops[chrom_mask]
            if len(chrom_loops) == 0:
                continue
            # build sparse matrix
            rows = np.vectorize(anchor_to_locus(chr_anchor_dict))(
                chrom_loops['a1'].values)
            cols = np.vectorize(anchor_to_locus(chr_anchor_dict))(
                chrom_loops['a2'].values)
            matrix = coo_matrix((chrom_loops['obs'], (rows, cols)),
                                shape=(len(chr_anchors), len(chr_anchors)))
            matrix = matrix.A
            for operation in operations:
                operation = operation.lower()
                if operation == 'vc_sqrt_norm':
                    matrix = VC_SQRT_norm(matrix)
                elif operation == 'convolution':
                    matrix = convolution(matrix)
                elif operation == 'random_walk':
                    matrix = random_walk(matrix)
                else:
                    raise ValueError(f'Invalid operation {operation}')
            # matrix[matrix < np.quantile(matrix, q)] = 0
            matrix = coo_matrix(matrix)
            new_chrom_loops = pd.DataFrame()
            new_chrom_loops['a1'] = matrix.row + min_anchor
            new_chrom_loops['a1'] = new_chrom_loops['a1'].astype(str)
            new_chrom_loops['a2'] = matrix.col + min_anchor
            new_chrom_loops['a2'] = new_chrom_loops['a2'].astype(str)
            new_chrom_loops['obs'] = matrix.data
            new_loops.append(new_chrom_loops)
        new_loops = pd.concat(new_loops)
        return new_loops

    def write_higashi_data(self, out_file, out_label_file, loops_file=None, n_strata=None, strata_offset=0, operations=None, preprocessing_resample_depth=1000):
        try:
            os.remove(out_file)
        except FileNotFoundError:
            pass
        if loops_file is not None:
            loops_list = np.loadtxt(loops_file, dtype=str)
            loops = set()
            for l in loops_list:
                loops.add(l)
            loops = list(loops)
            a1s = []
            a2s = []
            for loop in loops:
                chr1, start1, end1, chr2, start2, end2 = loop.split('_')
                chr1_mask = self.anchor_list['chr'] == chr1
                chr2_mask = self.anchor_list['chr'] == chr2
                start1_mask = self.anchor_list['start'] <= int(start1)
                end1_mask = self.anchor_list['end'] >= int(end1)
                start2_mask = self.anchor_list['start'] <= int(start2)
                end2_mask = self.anchor_list['end'] >= int(end2)
                a1_idx = self.anchor_list.loc[chr1_mask & start1_mask & end1_mask, 'anchor'].values[0]
                a2_idx = self.anchor_list.loc[chr2_mask & start2_mask & end2_mask, 'anchor'].values[0]
                a1s.append(a1_idx)
                a2s.append(a2_idx)
        cellnames = []
        cell_types = []
        depths = []
        batches = []
        anchor_ref = self.anchor_list.set_index('anchor')
        chr_dict = anchor_ref['chr'].to_dict()
        pos_dict = anchor_ref['start'].to_dict()
        for cell_i, cell_name in tqdm(enumerate(sorted(self.cell_list))):
            cluster = str(self.reference.loc[cell_name, 'cluster'])
            depth = float(self.reference.loc[cell_name, 'depth'])
            batch = str(self.reference.loc[cell_name, 'batch'])

            anchor_to_anchor = self.get_cell_pixels(cell_name)
            if operations is not None:
                chr_anchor_map = {}
                chr_anchor_dict_map = {}
                for chrom in sorted_nicely(self.anchor_list['chr'].unique()):
                    chr_anchors = self.anchor_list.loc[self.anchor_list['chr'] == chrom]
                    chr_anchor_map[chrom] = chr_anchors
                    chr_anchor_dict = anchor_list_to_dict(
                        chr_anchors['anchor'].values)
                    chr_anchor_dict_map[chrom] = chr_anchor_dict
                anchor_to_anchor = self.run_preprocessing_on_sparse(anchor_to_anchor, operations, chr_anchor_map, chr_anchor_dict_map)
                anchor_to_anchor['obs'] = anchor_to_anchor['obs'] * preprocessing_resample_depth
                anchor_to_anchor['obs'] = anchor_to_anchor['obs'].astype(int)
            if loops_file is not None:
                mask = ((anchor_to_anchor['a1'].isin(a1s)) & (anchor_to_anchor['a2'].isin(a2s)) | (anchor_to_anchor['a2'].isin(a1s)) & (anchor_to_anchor['a1'].isin(a2s)))
                anchor_to_anchor = anchor_to_anchor[mask]
            anchor_to_anchor.rename(
                columns={'a1': 'pos1', 'a2': 'pos2', 'obs': 'count'}, inplace=True)

            anchor_to_anchor['cell_id'] = cell_i
            anchor_to_anchor['chrom1'] = anchor_to_anchor['pos1'].map(chr_dict)
            anchor_to_anchor['chrom2'] = anchor_to_anchor['pos2'].map(chr_dict)
            anchor_to_anchor = anchor_to_anchor[anchor_to_anchor['chrom1'] != 'chrM']
            anchor_to_anchor = anchor_to_anchor[anchor_to_anchor['chrom2'] != 'chrM']
            anchor_to_anchor = anchor_to_anchor[anchor_to_anchor['chrom1'] != 'chrY']
            anchor_to_anchor = anchor_to_anchor[anchor_to_anchor['chrom2'] != 'chrY']
            anchor_to_anchor['pos1'] = anchor_to_anchor['pos1'].map(pos_dict)
            anchor_to_anchor['pos2'] = anchor_to_anchor['pos2'].map(pos_dict)
            if strata_offset > 0:
                anchor_to_anchor['dist'] = (anchor_to_anchor['pos1'] - anchor_to_anchor['pos2']).abs()
                anchor_to_anchor['strata'] = anchor_to_anchor['dist'] / self.resolution
                anchor_to_anchor['strata'] = anchor_to_anchor['strata'].astype(int)
                anchor_to_anchor = anchor_to_anchor[anchor_to_anchor['strata'] >= int(strata_offset)]
                if len(anchor_to_anchor) == 0:
                    continue
                if n_strata is None:
                    anchor_to_anchor.drop(columns=['dist', 'strata'], inplace=True)
            if n_strata is not None:
                if strata_offset == 0:
                    anchor_to_anchor['dist'] = (anchor_to_anchor['pos1'] - anchor_to_anchor['pos2']).abs()
                    anchor_to_anchor['strata'] = anchor_to_anchor['dist'] / self.resolution
                    anchor_to_anchor['strata'] = anchor_to_anchor['strata'].astype(int)
                anchor_to_anchor = anchor_to_anchor[anchor_to_anchor['strata'] <= n_strata + int(strata_offset)]
                if len(anchor_to_anchor) == 0:
                    continue
                anchor_to_anchor.drop(columns=['dist', 'strata'], inplace=True)
            cellnames.append(cell_name)
            cell_types.append(cluster)
            depths.append(depth)
            batches.append(batch)
            anchor_to_anchor.dropna(inplace=True)
            anchor_to_anchor['pos1'] = anchor_to_anchor['pos1'].astype(int)
            anchor_to_anchor['pos2'] = anchor_to_anchor['pos2'].astype(int)
            anchor_to_anchor.to_csv(out_file, sep='\t', index=False, mode='a', header=cell_i == 0)

        label_info = {
            'cell name': cellnames,
            'cell type': cell_types,
            'depth': depths,
            'batch': batches,
        }
        with open(out_label_file, "wb") as f:
            pickle.dump(label_info, f)

    def write_cell_bin_matrix(self, out_file=None, max_dist=None, preprocessing=None):
        mat = []
        cell_names = []
        heatmap = []
        chr_anchor_dfs = {}
        chr_anchor_dicts = {}
        print('Aggregating contact matrices to 1D vectors...')
        if preprocessing is not None:
            print(f'Preprocessing with {preprocessing}...')
        for cell_i, cell_name in tqdm(enumerate(sorted(self.cell_list)), total=len(self.cell_list)):
            try:
                loops = self.get_cell_pixels(cell_name)
            except Exception as e:
                print(cell_name, e)
                continue
            if len(loops) == 0:  # cell has no reads
                print('No reads, skipping...')
                continue
            if preprocessing is None:
                rows = np.vectorize(anchor_to_locus(self.anchor_dict))(
                    loops['a1'].values)  # convert anchor names to row indices
                cols = np.vectorize(anchor_to_locus(self.anchor_dict))(
                    loops['a2'].values)  # convert anchor names to column indices

                if max_dist is not None:
                    dist = np.abs(rows - cols)
                    mask = dist < max_dist
                    rows = rows[mask]
                    cols = cols[mask]
                    loops = loops.loc[mask]

                matrix = csr_matrix((loops['obs'], (rows, cols)),
                                    shape=(len(self.anchor_list), len(self.anchor_list)))
                mat.append(matrix.sum(axis=0))
                cell_names.append(cell_name)
            else:
                chr_mats = {}
                chr_list = list(pd.unique(self.anchor_list['chr']))
                results = []
                with Pool(8) as p:
                    for chr_name in chr_list:
                        results.append(p.apply_async(self.preprocess_mat_by_chr, args=(
                            chr_name, loops, chr_anchor_dfs, chr_anchor_dicts, preprocessing)))
                    tmp_mat = None
                    for res in results:
                        tmp_mat, chr_name = res.get(timeout=10)
                        chr_mats[chr_name] = tmp_mat
                block_mats = []
                for chr_name in chr_list:
                    if chr_name in chr_mats.keys():
                        block_mats.append(chr_mats[chr_name].A)
                    else:
                        block_mats.append(
                            np.zeros((len(chr_anchor_dfs[chr_name]), len(chr_anchor_dfs[chr_name]))))
                matrix = csr_matrix(block_diag(*block_mats))
                mat.append(matrix.sum(axis=0))
                cell_names.append(cell_name)
        mat = np.array(mat)
        mat = mat[:, 0, :]
        header = self.anchor_list
        header = header[header['end'] > 0]
        if out_file is not None:
            header.to_csv(out_file + '_bins', sep='\t',
                          header=False, index=False)
            np.savetxt(out_file, mat, header=' '.join(
                header['anchor'].to_list()), fmt='%d')
            np.savetxt(out_file + '_cells', np.array(cell_names), fmt='%s')
        else:
            return mat

    def write_toki_matrices(self, cell_name, toki_dir='toki_data'):
        chr_list = list(pd.unique(self.anchor_list['chr']))
        chr_lengths = {}
        chr_anchor_dfs = {}
        chr_anchor_dicts = {}
        loops = self.get_cell_pixels(cell_name)
        for chr_name in chr_list:
            if chr_name in chr_anchor_dicts.keys():  # reload chr data to save time
                chr_anchor_dict = chr_anchor_dicts[chr_name]
                chr_anchors = chr_anchor_dfs[chr_name]
                chr_lengths[chr_name] = len(chr_anchors)
            else:
                chr_anchors = self.anchor_list.loc[self.anchor_list['chr'] == chr_name]
                chr_anchors.reset_index(drop=True, inplace=True)
                chr_anchor_dfs[chr_name] = chr_anchors
                chr_anchor_dict = anchor_list_to_dict(
                    chr_anchors['anchor'].values)
                chr_anchor_dicts[chr_name] = chr_anchor_dict
                chr_lengths[chr_name] = len(chr_anchors)
            chr_contacts = loops[loops['a1'].isin(
                chr_anchors['anchor']) & loops['a2'].isin(chr_anchors['anchor'])]
            if len(chr_contacts) > 0:
                rows = np.vectorize(anchor_to_locus(chr_anchor_dict))(
                    chr_contacts['a1'].values)  # convert anchor names to row indices
                cols = np.vectorize(anchor_to_locus(chr_anchor_dict))(
                    chr_contacts['a2'].values)  # convert anchor names to column indices
                matrix = coo_matrix((chr_contacts['obs'], (rows, cols)),
                                    shape=(len(chr_anchors), len(chr_anchors)))
                matrix = matrix + matrix.transpose()
                np.savetxt(os.path.join(toki_dir, cell_name +
                           '_' + chr_name), matrix.A, fmt='%d')
            else:
                #print('No reads found for', cell_name, chr_name)
                matrix = np.zeros((len(chr_anchors), len(chr_anchors)))
                np.savetxt(os.path.join(toki_dir, cell_name +
                           '_' + chr_name), matrix, fmt='%d')
        return chr_lengths

    def summarize_tads(self, cell_name, chr_name, tads):
        loops = self.get_cell_pixels(cell_name)
        total = loops['obs'].sum()
        if chr_name is not None:
            chr_anchors = self.anchor_list.loc[self.anchor_list['chr'] == chr_name].copy(
            )
            chr_anchors.reset_index(drop=True, inplace=True)
            # if 'chr' in loops['a1'].iloc[0]:
            #     chr_anchors['anchor'] = chr_anchors['anchor'].apply(lambda s: chr_name + '_' + s)
            chr_anchor_dict = anchor_list_to_dict(chr_anchors['anchor'].values)
            chr_contacts = loops[loops['a1'].isin(
                chr_anchors['anchor']) & loops['a2'].isin(chr_anchors['anchor'])].copy()
        else:
            chr_anchors = self.anchor_list.copy()
            chr_anchor_dict = anchor_list_to_dict(chr_anchors['anchor'].values)
            chr_contacts = loops.copy()
        # print(chr_contacts)
        res = np.zeros(len(chr_anchors))

        if len(chr_contacts) > 0:
            rows = np.vectorize(anchor_to_locus(chr_anchor_dict))(
                chr_contacts['a1'].values)  # convert anchor names to row indices
            cols = np.vectorize(anchor_to_locus(chr_anchor_dict))(
                chr_contacts['a2'].values)  # convert anchor names to column indices
            chr_contacts['a1_idx'] = rows
            chr_contacts['a2_idx'] = cols
            tad_start = 0
            for tad_idx in tads:
                if tad_idx > tad_start:
                    a1_mask = (chr_contacts['a1_idx'] >= tad_start) & (
                        chr_contacts['a1_idx'] <= tad_idx)
                    a2_mask = (chr_contacts['a2_idx'] >= tad_start) & (
                        chr_contacts['a2_idx'] <= tad_idx)
                    tad_contacts = chr_contacts.loc[a1_mask & a2_mask]
                    # compute contact density within TAD (whole cell should sum to 1)
                    res[tad_start:tad_idx] = tad_contacts['obs'].sum() / \
                        total / (tad_idx - tad_start)
                tad_start = tad_idx
        return res

    def write_ins_matrices(self, cell_name, assembly='hg19', toki_dir='ins_data', split_chrs=True):
        chr_list = list(pd.unique(self.anchor_list['chr']))
        try:
            chr_list.remove('chrM')
        except Exception as e:
            pass
        chr_lengths = {}
        chr_anchor_dfs = {}
        chr_anchor_dicts = {}
        loops = self.get_cell_pixels(cell_name)
        offset_bin = False  # account for 0 or 1-based indexing
        round_bins = False  # smooth bins to uniform sizes with rounding
        if self.anchor_list['start'].min() == 0:
            offset_bin = True
        if 'pfc' in self.dataset_name:  # we know pfc data is mapped to non-uniform bins
            round_bins = True
        # check if other datasets have this mapping too
        anchor_length_var = (self.anchor_list['start'] - self.anchor_list['end']).abs().var()
        if anchor_length_var > 0:
            round_bins = True
        if split_chrs:
            for chr_name in chr_list:
                if chr_name in chr_anchor_dicts.keys():  # reload chr data to save time
                    chr_anchor_dict = chr_anchor_dicts[chr_name]
                    chr_anchors = chr_anchor_dfs[chr_name]
                    chr_lengths[chr_name] = len(chr_anchors)
                else:
                    chr_anchors = self.anchor_list.loc[self.anchor_list['chr'] == chr_name].copy(
                    )
                    chr_anchors.reset_index(drop=True, inplace=True)
                    if offset_bin:
                        chr_anchors['start'] += 1
                        chr_anchors['end'] += 1
                    chr_anchor_dfs[chr_name] = chr_anchors
                    chr_anchor_dict = anchor_list_to_dict(
                        chr_anchors['anchor'].values)
                    chr_anchor_dicts[chr_name] = chr_anchor_dict
                    chr_lengths[chr_name] = len(chr_anchors)
                if round_bins:
                    chr_anchors['header'] = chr_anchors.apply(lambda r: '%s|%s|%s:%d-%d' % (r['anchor'], assembly, r['chr'],
                                                                                            int(r['anchor'].split(
                                                                                                '_')[-1]) * self.resolution,
                                                                                            int(r['anchor'].split('_')[-1]) * self.resolution + self.resolution), axis=1)
                else:
                    chr_anchors['header'] = chr_anchors.apply(
                        lambda r: '%s|%s|%s:%d-%d' % (r['anchor'], assembly, r['chr'], r['start'], r['end']), axis=1)
                chr_contacts = loops[loops['a1'].isin(
                    chr_anchors['anchor']) & loops['a2'].isin(chr_anchors['anchor'])]
                if len(chr_contacts) > 0:
                    rows = np.vectorize(anchor_to_locus(chr_anchor_dict))(
                        chr_contacts['a1'].values)  # convert anchor names to row indices
                    cols = np.vectorize(anchor_to_locus(chr_anchor_dict))(
                        chr_contacts['a2'].values)  # convert anchor names to column indices
                    matrix = coo_matrix((chr_contacts['obs'], (rows, cols)),
                                        shape=(len(chr_anchors), len(chr_anchors))).A
                    matrix = matrix + matrix.T
                    # add col header
                    matrix = np.hstack(
                        (np.transpose([chr_anchors['header'].to_numpy()]), matrix))
                    # add row header
                    matrix = np.vstack(
                        (np.insert(chr_anchors['header'].to_numpy(), 0, ''), matrix))
                    np.savetxt(os.path.join(toki_dir, cell_name + '_' +
                               chr_name), matrix, fmt='%s', delimiter='\t')
                else:
                    #print('No reads found for', cell_name, chr_name)
                    matrix = np.zeros((len(chr_anchors), len(chr_anchors)))
                    # add col header
                    matrix = np.hstack(
                        (np.transpose([chr_anchors['header'].to_numpy()]), matrix))
                    # add row header
                    matrix = np.vstack(
                        (np.insert(chr_anchors['header'].to_numpy(), 0, ''), matrix))
                    np.savetxt(os.path.join(toki_dir, cell_name + '_' +
                               chr_name), matrix, fmt='%s', delimiter='\t')
        else:
            anchor_dict = anchor_list_to_dict(
                self.anchor_list['anchor'].values)
            if round_bins:
                self.anchor_list['header'] = self.anchor_list.apply(lambda r: '%s|%s|%s:%d-%d' % (r['anchor'], assembly, r['chr'],
                                                                                                  int(r['anchor'].split(
                                                                                                      '_')[-1]) * self.resolution,
                                                                                                  int(r['anchor'].split('_')[-1]) * self.resolution + self.resolution), axis=1)
            else:
                self.anchor_list['header'] = self.anchor_list.apply(
                    lambda r: '%s|%s|%s:%d-%d' % (r['anchor'], assembly, r['chr'], r['start'], r['end']), axis=1)
            if len(loops) > 0:
                rows = np.vectorize(anchor_to_locus(anchor_dict))(
                    loops['a1'].values)  # convert anchor names to row indices
                cols = np.vectorize(anchor_to_locus(anchor_dict))(
                    loops['a2'].values)  # convert anchor names to column indices
                matrix = coo_matrix((loops['obs'], (rows, cols)),
                                    shape=(len(self.anchor_list), len(self.anchor_list)))
                matrix = matrix + matrix.transpose()
                matrix = matrix.A
                matrix = np.hstack(
                    (np.transpose([self.anchor_list['header'].to_numpy()]), matrix))
                matrix = np.vstack(
                    (np.insert(self.anchor_list['header'].to_numpy(), 0, ''), matrix))
                np.savetxt(os.path.join(toki_dir, cell_name),
                           matrix, fmt='%s', delimiter='\t')
            else:
                matrix = np.zeros(
                    (len(self.anchor_list), len(self.anchor_list)))
                # add col header
                matrix = np.hstack(
                    (np.transpose([self.anchor_list['header'].to_numpy()]), matrix))
                # add row header
                matrix = np.vstack(
                    (np.insert(self.anchor_list['header'].to_numpy(), 0, ''), matrix))
                np.savetxt(os.path.join(toki_dir, cell_name),
                           matrix, fmt='%s', delimiter='\t')
        return chr_lengths

    def bin_pixels(self, loops, anchor_dict, anchor_chr_dict, chr_offsets, resolution, key='obs', use_chr_offset=True):
        loops['new_obs'] = 1
        loops['chr1'] = loops['a1'].map(anchor_chr_dict)
        loops['chr2'] = loops['a2'].map(anchor_chr_dict)
        loops = loops[loops['chr1'] == loops['chr2']]
        loops['a1_start'] = loops['a1'].map(anchor_dict)
        loops['a2_start'] = loops['a2'].map(anchor_dict)
        loops['bin1'] = loops['a1_start'] / resolution
        loops['bin1'] = loops['bin1'].astype(int)
        loops['bin2'] = loops['a2_start'] / resolution
        loops['bin2'] = loops['bin2'].astype(int)
        if use_chr_offset:
            loops['chr1_offset'] = loops['chr1'].map(chr_offsets)
            loops['chr2_offset'] = loops['chr2'].map(chr_offsets)
            loops['bin1'] = loops['bin1'] + loops['chr1_offset']
            loops['bin2'] = loops['bin2'] + loops['chr2_offset']
        loops['bin1'] = loops['bin1'].apply(lambda s: 'bin_' + str(s))
        loops['bin2'] = loops['bin2'].apply(lambda s: 'bin_' + str(s))
        loops = loops[['bin1', 'bin2', 'new_obs']]
        loops = loops.groupby(['bin1', 'bin2'])['new_obs'].sum().reset_index()
        loops.rename(columns={'new_obs': key}, inplace=True)
        return loops

    def bin_cell(self, cell_name, anchor_dict, anchor_chr_dict, chr_offsets, resolution):
        filename = 'frag_loop.' + cell_name.split('.')[0] + '.cis.filter'
        try:
            loops = self.get_cell_pixels(filename)
        except FileNotFoundError:
            loops = self.get_cell_pixels(filename.replace('.filter', ''))
        loops = self.bin_pixels(
            loops, anchor_dict, anchor_chr_dict, chr_offsets, resolution)
        return loops, cell_name

    def bin_cell_from_frags(self, cell_name, anchor_dict, anchor_chr_dict, chr_offsets, resolution):
        filename = 'frag_loop.' + cell_name.split('.')[0] + '.cis.filter'
        try:
            loops = self.get_cell_pixels(filename)
        except FileNotFoundError:
            loops = self.get_cell_pixels(filename.replace('.filter', ''))
        loops['chr1'] = loops['a1'].map(anchor_chr_dict)
        loops['chr2'] = loops['a2'].map(anchor_chr_dict)
        loops['a1_start'] = loops['a1'].map(anchor_dict)
        loops['a1_start'] = loops.apply(
            lambda row: row['a1_start'] + chr_offsets[row['chr1']], axis=1)
        loops['a2_start'] = loops['a2'].map(anchor_dict)
        loops['a2_start'] = loops.apply(
            lambda row: row['a2_start'] + chr_offsets[row['chr2']], axis=1)
        loops['bin1'] = loops['a1_start'] / resolution
        loops['bin1'] = loops['bin1'].astype(int)
        loops['bin1'] = loops['bin1'].apply(lambda s: 'bin_' + str(s))
        loops['bin2'] = loops['a2_start'] / resolution
        loops['bin2'] = loops['bin2'].astype(int)
        loops['bin2'] = loops['bin2'].apply(lambda s: 'bin_' + str(s))
        loops = loops[['bin1', 'bin2', 'obs']]
        loops = loops.groupby(['bin1', 'bin2'])['obs'].sum().reset_index()
        return loops, cell_name

    def get_uniform_bins(self, resolution, bin_offsets=True):
        chr_starts = {}
        chr_ends = {}
        chr_offsets = {}
        genome_len = 0
        for chr_name in sorted_nicely(self.anchor_list['chr'].unique()):
            chr_anchors = self.anchor_list[self.anchor_list['chr'] == chr_name]
            start = chr_anchors['start'].min()
            end = chr_anchors['end'].max()
            chr_starts[chr_name] = start
            chr_ends[chr_name] = end
            if not bin_offsets:  # offsets are in genomic coords
                chr_offsets[chr_name] = genome_len
            genome_len += end

        uniform_bins = pd.DataFrame()
        starts = np.arange(0, genome_len + resolution, resolution)
        ends = starts + resolution
        anchors = np.arange(0, len(starts))
        anchor = ['bin_' + str(v) for v in anchors]
        blank = 'none'
        chroms = [blank] * len(anchor)
        sorted_chrs = sorted_nicely(chr_starts.keys())
        prev_offset = 0
        for chr_name in sorted_chrs:
            start = int(chr_starts[chr_name] / resolution) + prev_offset
            end = int(chr_ends[chr_name] / resolution) + prev_offset
            idxs = np.arange(start, end + 1)
            starts[idxs] = np.arange(
                0, chr_ends[chr_name] - chr_starts[chr_name], resolution)
            ends = starts + resolution
            for i in idxs:
                chroms[i] = chr_name
            if bin_offsets:
                chr_offsets[chr_name] = prev_offset
            prev_offset += end - start

        uniform_bins['chr'] = pd.Series(chroms)
        uniform_bins['start'] = pd.Series(starts)
        uniform_bins['end'] = pd.Series(ends)
        uniform_bins['anchor'] = pd.Series(anchor)
        uniform_bins.dropna(inplace=True)
        uniform_bins = uniform_bins[uniform_bins['chr']
                                    != blank].reset_index(drop=True)
        uniform_bins['start'] = uniform_bins['start'].astype(int)
        uniform_bins['end'] = uniform_bins['end'].astype(int)
        return chr_offsets, uniform_bins

    def bin_cells_uniformly(self, out_dir, resolution=1000000, from_frags=True):
        os.makedirs(os.path.join(out_dir, self.res_name), exist_ok=True)
        anchor_ref = self.anchor_list.set_index('anchor')
        #anchor_ref['mid'] = (anchor_ref['start'] + anchor_ref['end']) / 2
        anchor_dict = anchor_ref['start'].to_dict()
        anchor_chr_dict = anchor_ref['chr'].to_dict()

        chr_offsets, uniform_bins = self.get_uniform_bins(
            resolution, bin_offsets=not from_frags)
        uniform_bins.to_csv('bins_%s.bed' % self.res_name,
                            sep='\t', header=False, index=False)

        results = []
        with Pool(16) as pool:
            current_cells = os.listdir(os.path.join(out_dir, self.res_name))
            for cell_i, cell_name in enumerate(sorted(self.cell_list)):
                if cell_name not in current_cells:
                    if from_frags:
                        results += [pool.apply_async(self.bin_cell_from_frags, args=(
                            cell_name, anchor_dict, anchor_chr_dict, chr_offsets, resolution))]
                    else:
                        results += [pool.apply_async(self.bin_cell, args=(
                            cell_name, anchor_dict, anchor_chr_dict, chr_offsets, resolution))]
            for res in tqdm(results):
                loops, cell_name = res.get(timeout=300)
                loops.to_csv(os.path.join(out_dir, self.res_name,
                             cell_name), sep='\t', header=False, index=False)

    def write_scHiCTools_matrices(self, out_dir, rewrite=True, resolution=1000000):
        print('Writing scHiCTools data (this only needs to be done once for each resolution or simulated replicate)...')
        network = []
        chr_lengths = {}

        chr_list = list(pd.unique(self.anchor_list['chr']))
        chr_anchor_dfs = {}
        chr_anchor_dicts = {}
        for cell_i, cell_name in tqdm(enumerate(sorted(self.cell_list)), total=len(self.cell_list)):
            anchor_to_anchor = self.get_cell_pixels(cell_name)
            if len(anchor_to_anchor) == 0:  # cell has no reads
                if self.verbose:
                    print('No reads found in', cell_name)
                continue
            os.makedirs(os.path.join(out_dir, cell_name), exist_ok=True)
            network.append(os.path.join(out_dir, cell_name))
            for chr_name in chr_list:
                if chr_name in chr_anchor_dicts.keys():  # reload chr data to save time
                    chr_anchor_dict = chr_anchor_dicts[chr_name]
                    chr_anchors = chr_anchor_dfs[chr_name]
                else:
                    chr_anchors = self.anchor_list.loc[self.anchor_list['chr'] == chr_name]
                    chr_anchors.reset_index(drop=True, inplace=True)
                    chr_anchor_dfs[chr_name] = chr_anchors
                    chr_length = len(chr_anchors)
                    chr_lengths[chr_name] = chr_length * resolution
                    chr_anchor_dict = anchor_list_to_dict(
                        chr_anchors['anchor'].values)
                    chr_anchor_dicts[chr_name] = chr_anchor_dict
                if rewrite:
                    chr_contacts = anchor_to_anchor[
                        anchor_to_anchor['a1'].isin(chr_anchors['anchor']) & anchor_to_anchor['a2'].isin(chr_anchors['anchor'])]
                    file_name = os.path.join(
                        out_dir, cell_name, '%s' % chr_name)
                    if len(chr_contacts) > 0:
                        rows = np.vectorize(anchor_to_locus(chr_anchor_dict))(
                            chr_contacts['a1'].values)  # convert anchor names to row indices
                        cols = np.vectorize(anchor_to_locus(chr_anchor_dict))(
                            chr_contacts['a2'].values)  # convert anchor names to column indices
                        matrix = coo_matrix((chr_contacts['obs'], (rows, cols)),
                                            shape=(len(chr_anchors), len(chr_anchors)))

                        save_npz(file_name, matrix)
                    else:
                        save_npz(file_name, coo_matrix(
                            np.zeros((len(chr_anchors), len(chr_anchors)))))
        return np.array(network), chr_lengths

    def get_sparse_matrix(self, cell_i, pad_depth=0):
        cell_name = self.cell_list[cell_i]
        loops = self.get_cell_pixels(cell_name)
        if len(loops) == 0:  # cell has no reads
            if self.verbose:
                print('No reads, skipping...')
            return None
        rows = np.vectorize(anchor_to_locus(self.anchor_dict))(
            loops['a1'].values)  # convert anchor names to row indices
        cols = np.vectorize(anchor_to_locus(self.anchor_dict))(
            loops['a2'].values)  # convert anchor names to column indices

        next = len(self.anchor_list) + (2 ** pad_depth -
                                        len(self.anchor_list) % (2 ** pad_depth))

        matrix = csr_matrix((loops['obs'], (rows, cols)),
                            shape=(next, next))
        return matrix

    def get_chr_sparse_matrix(self, cell_i, chr_name, pad_depth=0):
        cell_name = self.cell_list[cell_i]
        loops = self.get_cell_pixels(cell_name)
        if len(loops) == 0:  # cell has no reads
            if self.verbose:
                print('No reads, skipping...')
            return None
        chr_anchors = self.anchor_list[self.anchor_list['chr'] == chr_name]
        chr_contacts = loops[loops['a1'].isin(
            chr_anchors['anchor']) & loops['a2'].isin(chr_anchors['anchor'])]
        if len(chr_contacts) == 0:
            return None
        chr_anchor_dict = anchor_list_to_dict(chr_anchors['anchor'].values)
        rows = np.vectorize(anchor_to_locus(chr_anchor_dict))(
            chr_contacts['a1'].values)  # convert anchor names to row indices
        cols = np.vectorize(anchor_to_locus(chr_anchor_dict))(
            chr_contacts['a2'].values)  # convert anchor names to column indices

        #next = len(chr_anchors) + (2 ** pad_depth - len(chr_anchors) % (2 ** pad_depth))

        matrix = coo_matrix((chr_contacts['obs'], (rows, cols)),
                            shape=(len(chr_anchors), len(chr_anchors)))
        return matrix

    def preprocess_mat_by_chr(self, chr_name, loops, chr_anchor_dfs, chr_anchor_dicts, preprocessing, mask_visible=True):
        if chr_name in chr_anchor_dicts.keys():  # reload chr data to save time
            chr_anchor_dict = chr_anchor_dicts[chr_name]
            chr_anchors = chr_anchor_dfs[chr_name]
        else:
            chr_anchors = self.anchor_list.loc[self.anchor_list['chr'] == chr_name]
            chr_anchors.reset_index(drop=True, inplace=True)
            chr_anchor_dfs[chr_name] = chr_anchors
            chr_anchor_dict = anchor_list_to_dict(chr_anchors['anchor'].values)
            chr_anchor_dicts[chr_name] = chr_anchor_dict
        chr_contacts = loops.loc[loops['a1'].isin(
            chr_anchors['anchor']) & loops['a2'].isin(chr_anchors['anchor'])].copy()
        if len(chr_contacts) > 0:
            chr_contacts['chr1'] = chr_name
            chr_contacts['chr2'] = chr_name

            rows = np.vectorize(anchor_to_locus(chr_anchor_dict))(
                chr_contacts['a1'].values)  # convert anchor names to row indices
            cols = np.vectorize(anchor_to_locus(chr_anchor_dict))(
                chr_contacts['a2'].values)  # convert anchor names to column indices
            matrix = coo_matrix((chr_contacts['obs'], (rows, cols)),
                                shape=(len(chr_anchors), len(chr_anchors)))

            tmp_mat = matrix.A
            for op in preprocessing:
                if op.lower() == 'convolution':
                    tmp_mat = convolution(tmp_mat)
                elif op.lower() == 'random_walk':
                    tmp_mat = random_walk(tmp_mat)
                elif op.lower() == 'vc_sqrt_norm':
                    tmp_mat = VC_SQRT_norm(tmp_mat)
                elif op.lower() == 'network_enhance':
                    tmp_mat = network_enhance(tmp_mat)
                else:
                    print('Unrecognized preprocessing op', op)
            tmp_mat = coo_matrix((tmp_mat[rows, cols], (rows, cols)),
                                 shape=(len(chr_anchors), len(chr_anchors)))
        else:
            tmp_mat = coo_matrix((len(chr_anchors), len(chr_anchors)))

        return tmp_mat, chr_name

    def get_sparse_matrices(self, preprocessing=None):
        matrices = []
        chr_anchor_dfs = {}
        chr_anchor_dicts = {}
        for cell_name in tqdm(sorted(self.cell_list)):
            loops = self.get_cell_pixels(cell_name)
            if len(loops) == 0:  # cell has no reads
                print('No reads, skipping...')
                continue
            if preprocessing is not None:
                chr_mats = {}
                chr_list = list(pd.unique(self.anchor_list['chr']))
                results = []
                with Pool(8) as p:
                    for chr_name in chr_list:
                        results.append(p.apply_async(self.preprocess_mat_by_chr, args=(
                            chr_name, loops, chr_anchor_dfs, chr_anchor_dicts, preprocessing)))
                    tmp_mat = None
                    for res in results:
                        tmp_mat, chr_name = res.get(timeout=10)
                        chr_mats[chr_name] = tmp_mat
                block_mats = []
                for chr_name in chr_list:
                    if chr_name in chr_mats.keys():
                        block_mats.append(chr_mats[chr_name].A)
                    else:
                        block_mats.append(
                            np.zeros((len(chr_anchor_dfs[chr_name]), len(chr_anchor_dfs[chr_name]))))
                    matrix = csr_matrix(block_diag(*block_mats))
            else:
                rows = np.vectorize(anchor_to_locus(self.anchor_dict))(
                    loops['a1'].values)  # convert anchor names to row indices
                cols = np.vectorize(anchor_to_locus(self.anchor_dict))(
                    loops['a2'].values)  # convert anchor names to column indices
                matrix = csr_matrix((loops['obs'], (rows, cols)),
                                    shape=(len(self.anchor_list), len(self.anchor_list)))
            matrices.append(matrix)
        return matrices

    def get_cell_anchor_vectors(self, load=True):
        current_dir = os.listdir('.')
        if load and 'cell_anchor_vectors.npy' in current_dir and 'cell_labels.npy' in current_dir and 'cell_depths.npy' in current_dir:
            return np.load('cell_anchor_vectors.npy'), np.load('cell_labels.npy'), np.load('cell_depths.npy')
        else:
            self.cell_anchor_vectors = []
            depths = []
            batches = []
            labels = []
            for cell_i, cell_name in enumerate(sorted(self.cell_list)):
                cluster = str(self.reference.loc[cell_name, 'cluster'])
                batch = int(self.reference.loc[cell_name, 'batch'])
                depth = float(self.reference.loc[cell_name, 'depth'])
                label = np.argwhere(self.classes == cluster)[0]
                if load:
                    loops = self.get_cell_pixels(cell_name)
                    self.cell_anchor_vectors.append(loops['obs'].values)
                depths.append(depth)
                batches.append(batch)
                labels.append(label)
            return np.array(self.cell_anchor_vectors), np.array(labels), np.array(depths), np.array(batches)

    def filter_cells(self, min_depth=5000, saved_bad_cells='inadequate_cells.npy', ignoreXY=True, load=True, ignore_chr_filter=False):
        """scHiCluster recommends filtering cells with less than 5k contacts and cells where any chromosome
           does not have x reads for a chromosome length of x (Mb)"""
        if 'downsample' in self.dataset_name:
            remove_cells = []
            valid_cells = os.listdir(self.data_dir)
            for i, (cell_name, row) in enumerate(self.reference.iterrows()):
                if cell_name not in valid_cells:
                    remove_cells.append(cell_name)
            # drop all inadequate cells
            self.reference.drop(remove_cells, inplace=True, errors='ignore')
            if self.verbose:
                print('Cells after filtering: %d' % len(self.reference))
            return
        os.makedirs('data/inadequate_cells', exist_ok=True)
        if self.verbose:
            print('Cells before filtering: %d' % len(self.reference))
        saved_bad_cells = saved_bad_cells.replace(
            '.npy', '_%s.npy' % self.dataset_name)
        self.reference = self.reference[self.reference['depth'] >= min_depth].copy()
        self.full_reference.loc[self.full_reference['depth'] < min_depth, 'filtered_reason'] = 'reference depth < min_depth'
        if self.verbose:
            print('Cells before filtering by chr: %d' % len(self.reference))
        if saved_bad_cells in os.listdir('data/inadequate_cells') and not ignore_chr_filter:
            remove_cells = np.load(os.path.join(
                'data/inadequate_cells', saved_bad_cells))
            if remove_cells.size == 0:
                return
            suffix = remove_cells[0][remove_cells[0].rfind('.') + 1:]
            remove_cells = [s.replace(suffix, self.res_name)
                            for s in remove_cells]
            np.save(os.path.join('data/inadequate_cells',
                    saved_bad_cells), np.array(remove_cells))
        else:
            chr_list = list(pd.unique(self.anchor_list['chr']))
            chr_anchor_dict = {}
            chr_length_dict = {}
            remove_cells = []  # stores names (indices) of cells to be removed
            if self.data_dir is None:
                content_of_scool = cooler.fileops.list_coolers(self.scool_file)
            for i, (cell_name, row) in tqdm(enumerate(self.reference.iterrows()), total=len(self.reference), desc='Filtering cells'):
                # if using a cooler, check if cell is present
                if self.data_dir is None:
                    if '/cells/' + cell_name not in content_of_scool:
                        print('Could not find', cell_name)
                        remove_cells.append(cell_name)
                        continue
                try:
                    anchor_to_anchor = self.get_cell_pixels(cell_name)
                except FileNotFoundError as e:
                    print("Could not load", cell_name)
                    remove_cells.append(cell_name)
                    continue
                except UnboundLocalError as e:
                    print("Could not load", cell_name)
                    remove_cells.append(cell_name)
                    continue
                except KeyError as e:  # cannot find in .scool
                    print('Cannot find cell %s in .scool, filtering out...' %
                          cell_name)
                    remove_cells.append(cell_name)
                    continue
                if not ignore_chr_filter:
                    for chr_name in reversed(chr_list):  # reverse to maybe catch empty chroms early
                        if ignoreXY and ('chrX' in chr_name or 'chrY' in chr_name):
                            continue
                        if chr_name in chr_anchor_dict.keys() and chr_name in chr_length_dict.keys():  # reload chr data to save time
                            chr_anchors = chr_anchor_dict[chr_name]
                            chr_length = chr_length_dict[chr_name]
                        else:
                            chr_anchors = self.anchor_list[self.anchor_list['chr'] == chr_name]
                            chr_length = int(
                                (chr_anchors['end'].max() - chr_anchors['start'].min()) / 1e6)
                            chr_anchor_dict[chr_name] = chr_anchors
                            chr_length_dict[chr_name] = chr_length
                        a1_mask = anchor_to_anchor['a1'].isin(chr_anchors['anchor'])
                        a2_mask = anchor_to_anchor['a2'].isin(chr_anchors['anchor'])
                        chr_reads = int(anchor_to_anchor.loc[a1_mask & a2_mask, 'obs'].sum())
                        if chr_reads < chr_length:
                            if self.verbose:
                                print(len(remove_cells), 'Dropping', i,
                                      chr_name, chr_length, chr_reads)
                            remove_cells.append(cell_name)
                            break
            np.save(os.path.join('data/inadequate_cells',
                    saved_bad_cells), np.array(remove_cells))
        # drop all inadequate cells
        self.reference.drop(remove_cells, inplace=True, errors='ignore')
        self.full_reference.loc[self.full_reference['cell'].isin(remove_cells), 'filtered_reason'] = 'chr_reads < chr_length'
        self.full_reference.to_csv(f'data/{self.dataset_name}_filtered_ref', index=False, sep='\t')
        if self.verbose:
            filter_reasons = self.full_reference['filtered_reason'].unique()
            for reason in filter_reasons:
                print(reason, ':', self.full_reference['filtered_reason'].value_counts()[reason])
            print(f'Reference file with filtering criteria saved to data/{self.dataset_name}_filtered_ref')

    def downsample_mat(self, mat, p=0.5, minp=0.1, maxp=0.99):
        if random.random() >= p:
            # array to store new downsampled batch
            new_x = np.zeros(mat[..., 0].shape)
            # uniformly sample a downsampling percent
            downsample_percent = np.random.uniform(minp, maxp, size=1)
            for i, s in enumerate(mat[..., 0]):
                _, bins = self.downsample_strata(i, s, downsample_percent)
                new_x[i] = bins
            return new_x
        else:
            return mat[..., 0]

    def get_compressed_band_cell(self, cell, preprocessing=None, rw_max=100):
        if cell in self.sparse_matrices.keys():
            compressed_sparse_matrix = self.sparse_matrices[cell]
        else:
            anchor_to_anchor = self.get_cell_pixels(cell)
            if len(anchor_to_anchor) == 0:  # cell has no reads
                return None
            anchor_to_anchor = anchor_to_anchor[anchor_to_anchor['a1'].isin(
                self.anchor_list['anchor'])]
            anchor_to_anchor = anchor_to_anchor[anchor_to_anchor['a2'].isin(
                self.anchor_list['anchor'])]
            rows = np.vectorize(anchor_to_locus(self.anchor_dict))(
                anchor_to_anchor['a1'].values)  # convert anchor names to row indices
            cols = np.vectorize(anchor_to_locus(self.anchor_dict))(
                anchor_to_anchor['a2'].values)  # convert anchor names to column indices

            if self.use_raw_data:
                matrix = csr_matrix((anchor_to_anchor['obs'], (rows, cols)),
                                    shape=(len(self.anchor_list), len(self.anchor_list)))
            else:
                anchor_to_anchor['ratio'] = (
                    anchor_to_anchor['obs'] + 5) / (anchor_to_anchor['exp'] + 5)
                matrix = csr_matrix((anchor_to_anchor['ratio'], (rows, cols)),
                                    shape=(len(self.anchor_list), len(self.anchor_list)))
            if preprocessing is not None:
                tmp_mat = matrix.A
                for op in preprocessing:
                    if op.lower() == 'convolution':
                        tmp_mat = convolution(tmp_mat)
                    elif op.lower() == 'random_walk':
                        tmp_mat = random_walk(tmp_mat)
                        tmp_mat = np.int32(tmp_mat * rw_max)
                    elif op.lower() == 'vc_sqrt_norm':
                        tmp_mat = VC_SQRT_norm(tmp_mat)
                    elif op.lower() == 'google':
                        tmp_mat = graph_google(tmp_mat)
                        
                matrix = csr_matrix(tmp_mat)

            compressed_matrix = np.zeros(
                (self.limit2Mb, self.matrix_len + self.matrix_pad))
            for i in range(self.limit2Mb):
                diagonal = matrix.diagonal(k=i + self.rotated_offset)
                compressed_matrix[i, i:i + len(diagonal)] = diagonal
            if self.active_idxs is not None:
                compressed_sparse_matrix = csr_matrix(np.hstack(
                    [compressed_matrix[:, self.active_idxs], np.zeros((self.limit2Mb, self.matrix_pad))]))
                self.sparse_matrices[cell] = compressed_sparse_matrix
            else:
                compressed_sparse_matrix = csr_matrix(compressed_matrix)
                self.sparse_matrices[cell] = compressed_sparse_matrix
        return compressed_sparse_matrix

