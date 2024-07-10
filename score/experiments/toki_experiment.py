import os
import re
import shutil
import argparse
import numpy as np

from tqdm import tqdm
from multiprocessing import Pool
from sklearn.decomposition import PCA


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



class TokiExperiment(Experiment):
    def __init__(self, name, x, y, depths, data_generator, tad_dir, **kwargs):
        super().__init__(name, x, y, depths, data_generator, **kwargs)
        self.tad_dir = tad_dir


    def get_embedding(self, iter_n=0):
        n_components = self.latent_dim
        z = []
        for cell_i, cell_name in tqdm(enumerate(sorted(self.data_generator.cell_list))):
            tads = np.loadtxt(os.path.join(self.tad_dir, cell_name))
            z.append(tads)
        z = np.array(z)
        z_pca = PCA(n_components).fit_transform(z)
        return z_pca


def call_tads(cell_i, cell_name, train_generator, toki_dir, tad_count=True, core_n=12, delta_scale=1.0):
    from score.methods.TOKI.TOKI import run_detoki
    os.makedirs(toki_dir, exist_ok=True)
    chr_lengths = train_generator.write_toki_matrices(cell_name, toki_dir)
    chr_offset = 0
    tads = np.zeros(len(train_generator.anchor_list))
    for chr_name in sorted_nicely(train_generator.anchor_list['chr'].unique()):
        in_file = '%s/%s_%s' % (toki_dir, cell_name, chr_name)
        out_file = '%s_TAD' % in_file
        run_detoki(in_file, out_file, train_generator.resolution / 1000, size=[600, 4000], core=core_n, split=8000, delta_scale=delta_scale)
        try:
            with open(out_file, 'r') as f:
                if os.fstat(f.fileno()).st_size:  # ensure file is not empty
                    chr_tads = np.int32(np.ravel(np.loadtxt(out_file)))
                    if tad_count:
                        tads[chr_offset:chr_offset + chr_lengths[chr_name]] = train_generator.summarize_tads(cell_name, chr_name, chr_tads)
                    else:
                        tads[chr_tads + chr_offset] += 1
        except Exception as e:
            pass
        chr_offset += chr_lengths[chr_name]
        if delta_scale != 1.0:  # only run one chrom when testing
            break
    tads = np.array(tads)
    return cell_i, cell_name, tads


if __name__ == '__main__':
    from score.sc_args import parse_args
    parser = argparse.ArgumentParser()
    parser.add_argument('--toki_dir', default='toki_data')
    parser.add_argument('--tad_count', action='store_true')
    x_train, y_train, depths, batches, train_generator, cm = parse_args(parser)
    args = parser.parse_args()

    tad_count = args.tad_count
    toki_dir = os.path.join(args.toki_dir, train_generator.dataset_name, train_generator.res_name)
    if not tad_count:
        tad_dir = os.path.join(args.toki_dir, train_generator.dataset_name, train_generator.res_name + '_TAD')
    else:
        tad_dir = os.path.join(args.toki_dir, train_generator.dataset_name, train_generator.res_name + '_TAD_count')
    os.makedirs(toki_dir, exist_ok=True)
    os.makedirs(tad_dir, exist_ok=True)

    results = []
    with Pool(4) as pool:
        current_tads = os.listdir(tad_dir)
        for cell_i, cell_name in enumerate(sorted(train_generator.cell_list)):
            if (cell_name) not in current_tads:
                results += [pool.apply_async(call_tads, args=(cell_i, cell_name, train_generator, os.path.join(toki_dir, str(cell_i)), tad_count))]
        for res in tqdm(results):
            cell_i, cell_name, tads = res.get(timeout=1000)
            np.savetxt(os.path.join(tad_dir, cell_name), tads)
            shutil.rmtree(os.path.join(toki_dir, str(cell_i)))  # delete dense matrices and intermediate chr TADs

    exp_name = "deTOKI"
    experiment = TokiExperiment(exp_name, x_train, y_train, depths, train_generator, tad_dir=tad_dir)
    experiment.run(load=False)
