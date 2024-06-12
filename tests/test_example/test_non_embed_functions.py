import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch
from score.__main__ import app, version_callback


def test_version():
    version_callback(True)
    assert True


def test_help():
    test_args = ['app.py', '--help']
    os.chdir(Path(__file__).parent)
    with patch.object(sys, 'argv', test_args):
        app()
    assert True


def test_heatmaps():
    test_args = ['app.py', 'heatmaps', '--dataset_config', 'data/dataset_configs/oocyte_zygote_mm10.json']
    os.chdir(Path(__file__).parent)
    with patch.object(sys, 'argv', test_args):
        app()
    assert True


def test_agg():
    test_args = ['app.py', '1d_agg', '--dataset_config', 'data/dataset_configs/oocyte_zygote_mm10.json']
    os.chdir(Path(__file__).parent)
    with patch.object(sys, 'argv', test_args):
        app()
    assert True


def test_summary():
    test_args = ['app.py', 'summary', '--dataset_config', 'data/dataset_configs/oocyte_zygote_mm10.json']
    os.chdir(Path(__file__).parent)
    with patch.object(sys, 'argv', test_args):
        app()
    assert True


def test_bin():
    test_args = ['app.py', 'bin', '--dataset_config', 'data/dataset_configs/oocyte_zygote_mm10.json']
    os.chdir(Path(__file__).parent)
    with patch.object(sys, 'argv', test_args):
        app()
    assert True


def test_merge():
    test_args = ['app.py', 'merge', '--dataset_config', 'data/dataset_configs/oocyte_zygote_mm10.json']
    os.chdir(Path(__file__).parent)
    with patch.object(sys, 'argv', test_args):
        app()
    assert True