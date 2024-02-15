import os
import re
import sys
import glob
import shutil
import json
import pickle
import argparse
import traceback
from matplotlib import colors
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

from tqdm import tqdm
from multiprocessing import Pool
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d

from score.sc_args import parse_args
from score.experiments.experiment import Experiment


def sorted_nicely(l):
    """
    Sorts an iterable object according to file system defaults
    Args:
        l (:obj:`iterable`) : iterable object containing items which can be interpreted as text
    Returns:
        `iterable` : sorted iterable
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)



class ISExperiment(Experiment):
    def __init__(self, name, x, y, depths, data_generator, tad_dir, **kwargs):
        super().__init__(name, x, y, depths, data_generator, **kwargs)
        self.tad_dir = tad_dir


    def get_embedding(self, iter_n=0):
        n_components = self.latent_dim
        z = []
        heatmap = []
        for cell_i, cell_name in tqdm(enumerate(sorted(self.data_generator.cell_list))):
            tads = np.loadtxt(os.path.join(self.tad_dir, cell_name))[:len(self.data_generator.anchor_list)]
            
            tads = np.nan_to_num(tads)
            if len(tads) == 0:
                tads = np.zeros(len(self.data_generator.anchor_list))
            if cell_name == 'CEMBA191126-8J-1-CEMBA191126-8J-2-A2_ad010.200kb':
                print(len(tads), len(z[-1]))
            z.append(tads)
            heatmap.append(gaussian_filter1d(tads, sigma=1))
        z = np.array(z)
        # lens = [len(x) for x in z]
        # print(lens)
        # print(np.unique(lens))
        # print(z)
        # heatmap = np.array(heatmap)
        # try:
        #     batches = self.data_generator.reference['cluster']
        #     lut = dict(zip(batches.unique(), list(colors.BASE_COLORS.keys()) + list(colors.TABLEAU_COLORS.keys())))
        #     row_colors = np.array(batches.map(lut))
        #     print(batches)
        #     print(row_colors)
        #     sns.clustermap(heatmap, metric='euclidean', cmap='vlag', standard_scale=0, row_colors=row_colors, method='ward', col_cluster=True)
        #     plt.savefig(os.path.join(self.out_dir, 'INS_clustermap.png'))
        #     plt.close()
        # except ValueError as e:
        #     print(e)
        # except TypeError as e:
        #     print(e)
        z_pca = PCA(n_components).fit_transform(z)
        return z_pca

def map_to_chr_scores(chr_scores, value_key):
    def f(row):
        #try:
        chr_scores[:min(row['binEnd'], len(chr_scores))] = row[value_key]
        # except IndexError:
        #     pass
        return chr_scores
    return f

def ins_score(cell_i, cell_name, train_generator, assembly, toki_dir, value_key, split_chr=True):
    os.makedirs(toki_dir, exist_ok=True)
    chr_lengths = train_generator.write_ins_matrices(cell_name, assembly, toki_dir)
    ins_scores = []
    if split_chr:
        for chr_name in sorted_nicely(train_generator.anchor_list['chr'].unique()):
            if chr_name == 'chrM':
                continue
            chr_ins_scores = np.zeros(chr_lengths[chr_name])
            in_file = '%s/%s_%s' % (toki_dir, cell_name, chr_name)
            for directory in sys.path:
                is_script = os.path.join(directory, 'score/methods/crane-nature-2015/scripts/matrix2insulation.pl')
                if os.path.isfile(is_script):
                    toki_cmd = "perl %s -b %d -ids %d -bmoe 3 -i %s" % (is_script, train_generator.resolution * 10,  train_generator.resolution * 4, in_file)
                    os.system(toki_cmd)
                    for file in glob.glob('./%s*' % cell_name.replace(train_generator.res_name, '')):
                        if file.endswith('.insulation'):
                            #try:
                            df = pd.read_csv(file, sep='\t')
                            df.fillna(0, inplace=True)
                            #chr_ins_scores = df.apply(map_to_chr_scores(chr_ins_scores, value_key), axis=1).mean()
                            chr_ins_scores = df[value_key].values[:len(chr_ins_scores)]
                            # for _, row in df.iterrows():
                            #     chr_ins_scores[row['binStart']: row['binEnd']] = row[value_key]
                            ins_scores += list(chr_ins_scores)
                            # except Exception as e:
                            #     print(e)
                            #     ins_scores += [0] * len(chr_lengths[chr_name])
                                
                            #     pass
                        os.remove(file)
                    break
    else:
        in_file = '%s/%s' % (toki_dir, cell_name)
        ins_scores = np.zeros(len(train_generator.anchor_list))
        for directory in sys.path:
            is_script = os.path.join(directory, 'score/methods/crane-nature-2015/scripts/matrix2insulation.pl')
            if os.path.isfile(is_script):
                toki_cmd = "perl %s -b %d -ids %d -bmoe 3 -i %s -im mean" % (is_script, train_generator.resolution * 10,  train_generator.resolution * 5, in_file)
                os.system(toki_cmd)
                for file in glob.glob('./%s*' % cell_name.replace(train_generator.res_name, '')):
                    if file.endswith('.insulation'):
                        try:
                            df = pd.read_csv(file, sep='\t')
                            df.fillna(0, inplace=True)
                            ins_scores = df.apply(map_to_chr_scores(ins_scores, value_key), axis=1).mean()
                        except Exception as e:
                            print(e)
                            pass
                    os.remove(file)
                break

    tads = np.array(ins_scores)
    return cell_i, cell_name, tads


def run_ins_score(tad_dir, toki_dir, train_generator, assembly, value_key='delta', multiproc=False, n_threads=8, split_chr=True):
    current_tads = os.listdir(tad_dir)
    results = []
    with Pool(n_threads) as pool:
        for cell_i, cell_name in enumerate(sorted(train_generator.cell_list)):
            if (cell_name) not in current_tads:
                results += [pool.apply_async(ins_score, args=(cell_i, cell_name, train_generator, assembly, os.path.join(toki_dir, str(cell_i)), value_key, split_chr))]
            #break
        for res in tqdm(results):
            cell_i, cell_name, tads = res.get(timeout=3600)
            np.savetxt(os.path.join(tad_dir, cell_name), tads)
            shutil.rmtree(os.path.join(toki_dir, str(cell_i)))  # delete dense matrices and intermediate chr TADs



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--assembly', default='mm10')
    parser.add_argument('--toki_dir', default='ins_data')
    parser.add_argument('--tad_count', action='store_true')
    x_train, y_train, depths, batches, train_generator, cm = parse_args(parser)

    args = parser.parse_args()

    assembly = args.assembly
    tad_count = args.tad_count
    toki_dir = os.path.join(args.toki_dir, train_generator.dataset_name, train_generator.res_name)
    if not tad_count:
        tad_dir = os.path.join(args.toki_dir, train_generator.dataset_name, train_generator.res_name + '_INS')
    else:
        tad_dir = os.path.join(args.toki_dir, train_generator.dataset_name, train_generator.res_name + '_INS_count')
    os.makedirs(toki_dir, exist_ok=True)
    os.makedirs(tad_dir, exist_ok=True)

    run_ins_score(tad_dir, toki_dir, train_generator, assembly)

    if tad_count:
        exp_name = "InsScore_Count"
    else:
        exp_name = "InsScore"
    experiment = ISExperiment(exp_name, x_train, y_train, depths, train_generator, tad_dir=tad_dir)
    experiment.run(load=False)
