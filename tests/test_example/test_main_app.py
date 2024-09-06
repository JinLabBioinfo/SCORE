import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import sys
import pytest
from pathlib import Path
from unittest.mock import patch
from score.__main__ import app, version_callback


coarsened_scool_args = ['--scool', 'data/scools/oocyte_zygote_mm10_4M_coarsened.scool', '--resolution', '4M']
min_depth = '120000'
subsample_args = ['--subsample_n_cells', '65']


def test_embed():
    # tests the main app embedding logic on a preconfigured dataset
    test_args = ['app.py', 'embed', '--dataset_config', 'data/dataset_configs/oocyte_zygote_mm10.json', 
                 '--n_runs', "3", '--min_depth', min_depth]
    test_args += subsample_args
    os.chdir(Path(__file__).parent)
    with patch.object(sys, 'argv', test_args):
        app()
    assert True


def test_baseline_sweep():
    test_args = ['app.py', 'embed', '--dataset_config', 'data/dataset_configs/oocyte_zygote_mm10.json', 
                 '--baseline_sweep', '--no_viz', '--n_runs', '3', '--min_depth', min_depth]
    test_args += subsample_args
    os.chdir(Path(__file__).parent)
    with patch.object(sys, 'argv', test_args):
        app()
    assert True


def test_resolution_sweep():
    test_args = ['app.py', 'embed', '--dataset_config', 'data/dataset_configs/oocyte_zygote_mm10.json', 
                 '--resolution_sweep', '--no_viz', '--n_runs', '3', '--min_depth', min_depth]
    test_args += subsample_args
    os.chdir(Path(__file__).parent)
    with patch.object(sys, 'argv', test_args):
        app()
    assert True


def test_compare():
    test_args = ['app.py', 'compare', '--dataset_config', 'data/dataset_configs/oocyte_zygote_mm10.json', '--dset', 'oocyte_zygote_mm10']
    os.chdir(Path(__file__).parent)
    with patch.object(sys, 'argv', test_args):
        app()
    assert True


def test_schicluster():
    test_args = ['app.py', 'embed', '--dataset_config', 'data/dataset_configs/oocyte_zygote_mm10.json',
                 '--embedding_algs', 'scHiCluster', '--no_viz', '--eval_celltypes', ['ZygM', 'ZygP']]
    test_args += subsample_args
    os.chdir(Path(__file__).parent)
    with patch.object(sys, 'argv', test_args):
        app()
    assert True


def test_random_walk():
    test_args = ['app.py', 'embed', '--dataset_config', 'data/dataset_configs/oocyte_zygote_mm10.json',
                 '--embedding_algs', '2d_pca+vc_sqrt_norm,random_walk,quantile_0.8', '--no_viz',
                 '--random_walk_iter', '5', '--random_walk_ratio', '0.5', '--min_depth', min_depth]
    test_args += subsample_args
    os.chdir(Path(__file__).parent)
    with patch.object(sys, 'argv', test_args):
        app()
    assert True


def test_innerproduct():
    test_args = ['app.py', 'embed', '--dataset_config', 'data/dataset_configs/oocyte_zygote_mm10.json',
                 '--embedding_algs', 'InnerProduct', '--no_viz']
    os.chdir(Path(__file__).parent)
    with patch.object(sys, 'argv', test_args):
        app()
    assert True


def test_load():
    test_args = ['app.py', 'embed', '--dataset_config', 'data/dataset_configs/oocyte_zygote_mm10.json',
                 '--embedding_algs', 'InnerProduct', '--no_viz', '--load_results']
    os.chdir(Path(__file__).parent)
    with patch.object(sys, 'argv', test_args):
        app()
    assert True


def test_acroc():
    test_args = ['app.py', 'embed', '--dataset_config', 'data/dataset_configs/oocyte_zygote_mm10.json',
                 '--embedding_algs', 'InnerProduct', '--no_viz', '--continuous', '--min_depth', min_depth]
    test_args += subsample_args
    os.chdir(Path(__file__).parent)
    with patch.object(sys, 'argv', test_args):
        app()
    assert True


def test_innerproduct_viz():
    test_args = ['app.py', 'embed', '--dataset_config', 'data/dataset_configs/oocyte_zygote_mm10.json',
                 '--embedding_algs', 'InnerProduct', '--no_viz', '--viz_innerproduct', '--min_depth', min_depth]
    test_args += subsample_args
    os.chdir(Path(__file__).parent)
    with patch.object(sys, 'argv', test_args):
        app()
    assert True


def test_cistopic():
    test_args = ['app.py', 'embed', '--dataset_config', 'data/dataset_configs/oocyte_zygote_mm10.json',
                 '--embedding_algs', 'cisTopic', '--no_viz', '--min_depth', min_depth]
    test_args += subsample_args
    os.chdir(Path(__file__).parent)
    with patch.object(sys, 'argv', test_args):
        app()
    assert True


def test_pca():
    test_args = ['app.py', 'embed', '--dataset_config', 'data/dataset_configs/oocyte_zygote_mm10.json',
                 '--embedding_algs', '1d_pca', '--no_viz', '--min_depth', min_depth]
    test_args += subsample_args
    os.chdir(Path(__file__).parent)
    with patch.object(sys, 'argv', test_args):
        app()
    assert True


def test_lsi():
    test_args = ['app.py', 'embed', '--dataset_config', 'data/dataset_configs/oocyte_zygote_mm10.json',
                 '--embedding_algs', '1d_lsi', '--no_viz', '--min_depth', min_depth]
    test_args += subsample_args
    os.chdir(Path(__file__).parent)
    with patch.object(sys, 'argv', test_args):
        app()
    assert True


def test_insulation():
    test_args = ['app.py', 'embed', '--dataset_config', 'data/dataset_configs/oocyte_zygote_mm10.json',
                 '--embedding_algs', 'insulation', '--no_viz', '--min_depth', min_depth]
    test_args += subsample_args
    os.chdir(Path(__file__).parent)
    with patch.object(sys, 'argv', test_args):
        app()
    assert True


def test_detoki():
    test_args = ['app.py', 'embed', '--dataset_config', 'data/dataset_configs/oocyte_zygote_mm10.json',
                 '--embedding_algs', 'deTOKI', '--no_viz', '--toki_delta_scale', '0.05', '--min_depth', min_depth]
    # test deTOKI on lower resolution since it is faster
    test_args += coarsened_scool_args
    test_args += subsample_args
    os.chdir(Path(__file__).parent)
    with patch.object(sys, 'argv', test_args):
        app()
    assert True


def test_higashi():
    test_args = ['app.py', 'embed', '--dataset_config', 'data/dataset_configs/oocyte_zygote_mm10.json',
                 '--embedding_algs', 'higashi', '--no_viz', '--higashi_dryrun', '--min_depth', min_depth]
    test_args += subsample_args
    os.chdir(Path(__file__).parent)
    with patch.object(sys, 'argv', test_args):
        app()
    assert True


def test_higashi_rw():
    test_args = ['app.py', 'embed', '--dataset_config', 'data/dataset_configs/oocyte_zygote_mm10.json',
                 '--embedding_algs', 'higashi+vc_sqrt_norm,random_walk', '--no_viz', '--higashi_dryrun', '--min_depth', min_depth]
    test_args += subsample_args
    os.chdir(Path(__file__).parent)
    with patch.object(sys, 'argv', test_args):
        app()
    assert True


def test_fast_higashi():
    test_args = ['app.py', 'embed', '--dataset_config', 'data/dataset_configs/oocyte_zygote_mm10.json',
                 '--embedding_algs', 'fast_higashi', '--no_viz', '--higashi_dryrun', '--min_depth', min_depth]
    test_args += subsample_args
    os.chdir(Path(__file__).parent)
    with patch.object(sys, 'argv', test_args):
        app()
    assert True


def test_snapatac():
    test_args = ['app.py', 'embed', '--dataset_config', 'data/dataset_configs/oocyte_zygote_mm10.json',
                 '--embedding_algs', 'snapatac', '--no_viz', '--min_depth', min_depth]
    test_args += subsample_args
    test_args += coarsened_scool_args
    os.chdir(Path(__file__).parent)
    with patch.object(sys, 'argv', test_args):
        app()
    assert True


def test_3dvi():
    test_args = ['app.py', 'embed', '--dataset_config', 'data/dataset_configs/oocyte_zygote_mm10.json',
                 '--embedding_algs', '3DVI', '--no_viz', '--n_strata', '1', '--min_depth', min_depth]
    test_args += subsample_args
    os.chdir(Path(__file__).parent)
    with patch.object(sys, 'argv', test_args):
        app()
    assert True


def test_vade():
    test_args = ['app.py', 'embed', '--dataset_config', 'data/dataset_configs/oocyte_zygote_mm10.json',
                 '--embedding_algs', 'vade', '--no_viz', '--n_strata', '32', '--min_depth', min_depth]
    os.chdir(Path(__file__).parent)
    with patch.object(sys, 'argv', test_args):
        app()
    assert True



