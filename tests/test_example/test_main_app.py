import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch
from score.__main__ import app, version_callback


def test_embed():
    # tests the main app embedding logic on a preconfigured dataset
    test_args = ['app.py', 'embed', '--dataset_config', 'data/dataset_configs/oocyte_zygote_mm10.json', 
                 '--n_runs', "3"]
    os.chdir(Path(__file__).parent)
    with patch.object(sys, 'argv', test_args):
        app()
    assert True


def test_baseline_sweep():
    test_args = ['app.py', 'embed', '--dataset_config', 'data/dataset_configs/oocyte_zygote_mm10.json', 
                 '--baseline_sweep', '--no_viz', '--n_runs', '3']
    os.chdir(Path(__file__).parent)
    with patch.object(sys, 'argv', test_args):
        app()
    assert True


def test_resolution_sweep():
    test_args = ['app.py', 'embed', '--dataset_config', 'data/dataset_configs/oocyte_zygote_mm10.json', 
                 '--resolution_sweep', '--no_viz', '--n_runs', '3']
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


def test_higashi():
    test_args = ['app.py', 'embed', '--dataset_config', 'data/dataset_configs/oocyte_zygote_mm10.json',
                 '--embedding_algs', 'higashi', '--no_viz', '--higashi_dryrun']
    os.chdir(Path(__file__).parent)
    with patch.object(sys, 'argv', test_args):
        app()
    assert True


def test_3dvi():
    test_args = ['app.py', 'embed', '--dataset_config', 'data/dataset_configs/oocyte_zygote_mm10.json',
                 '--embedding_algs', '3DVI', '--no_viz', '--n_strata', '1']
    os.chdir(Path(__file__).parent)
    with patch.object(sys, 'argv', test_args):
        app()
    assert True


def test_vade():
    test_args = ['app.py', 'embed', '--dataset_config', 'data/dataset_configs/oocyte_zygote_mm10.json',
                 '--embedding_algs', 'vade', '--no_viz', '--n_strata', '32']
    os.chdir(Path(__file__).parent)
    with patch.object(sys, 'argv', test_args):
        app()
    assert True



