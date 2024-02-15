import os
import sys
import json
import pytest
import argparse
import numpy as np
from pathlib import Path
from scipy import sparse
from unittest.mock import patch

from score.__main__ import app
from score.sc_args import parse_args


def test_scool():
    # tests the main app dataset conversion logic on the provided example dataset
    # should output the .scool file used for the rest of the full integration testing
    # first remove any leftover files from previous runs
    clear_cmds = [f"rm -r data/example_datasets/oocyte_zygote_mm10",
                  f"rm -r results",
                  f"rm -r data/inadequate_cells",
                  f"rm -r schictools_data",
                  f"rm -r schic-topic-model",
                  f"rm -r threeDVI",
                  f"rm -r wandb"]
    # copy over the example data (at least what can be stored in the repo)
    cp_cmds = [f"cp -r ../../examples/data/dataset_colors data",
               f"cp -r ../../examples/data/dataset_configs data",
               f"cp -r ../../examples/data/higashi_data data",
               f"cp ../../examples/data/mm10.genome_split_1M data/mm10.genome_split_1M",
               f"cp ../../examples/data/oocyte_zygote_ref data/oocyte_zygote_ref"]
    unzip_cmd = f"unzip ../../examples/data/oocyte_zygote_mm10.zip -d data/example_datasets/oocyte_zygote_mm10"
    test_args = ["app.py", "cooler",
                 "--dset", "oocyte_zygote_mm10",
                 "--assembly", "mm10",
                 "--data_dir", "data/example_datasets/oocyte_zygote_mm10/1M",
                 "--anchor_file", "data/mm10.genome_split_1M",
                 "--reference", "data/oocyte_zygote_ref",
                 "--resolution", "1M",
                 "--ignore_filter", "--ignore_chr_filter"]
    os.chdir(Path(__file__).parent)
    os.makedirs("data/example_datasets/", exist_ok=True)
    os.makedirs("data/scools/", exist_ok=True)
    for clear_cmd in clear_cmds:
        try:
            os.system(clear_cmd)
        except Exception as e:
            print(e)
    os.system(unzip_cmd)
    for cp_cmd in cp_cmds:
        os.system(cp_cmd)
    with patch.object(sys, "argv", test_args):
        app()
    assert True


def test_basic_load():
    data_path = str(Path(__file__).parent) + "/data/dataset_configs"
    os.chdir(Path(__file__).parent)
    with open(os.path.join(data_path, "oocyte_zygote_mm10.json"), "r") as f:
        args = json.load(f)
    parser = argparse.ArgumentParser()
    parse_args(parser, extra_args=args)
    assert True


def triu_only(dataset):
    matrices = dataset.get_sparse_matrices()
    errors = []
    for m in matrices:
        v = sparse.tril(m, k=-1).sum()
        # ensure only upper triangle is stored and loaded
        errors.append(v != 0)
    return not any(errors)


def non_nans(dataset):
    matrices = dataset.get_sparse_matrices()
    errors = []
    for m in matrices:
        errors.append(np.any(np.isnan(m.data)))
    return not any(errors)


def cell_count(dataset):
    # 150 cells in the test dataset after filtering by 5k total reads and scHiCluster cis-read count filtering
    return dataset.n_cells, dataset.n_cells == 150


def test_data_properties():
    data_path = str(Path(__file__).parent) + "/data/dataset_configs"
    os.chdir(Path(__file__).parent)
    with open(os.path.join(data_path, "oocyte_zygote_mm10.json"), "r") as f:
        args = json.load(f)
    parser = argparse.ArgumentParser()
    args, x, y, depths, batches, dataset, _ = parse_args(
        parser, extra_args=args)

    errors = []
    if not triu_only(dataset):
        errors.append("Matrix loader should only generate triu values")
    if not non_nans(dataset):
        errors.append("Found NaN in at least one contact matrix")
    n_cells, passed = cell_count(dataset)
    if not passed:
        errors.append(
            f"Incorrect number of cells ({n_cells} =/= 150) after filtering")
    assert not errors, "errors occured:\n{}".format("\n".join(errors))


def no_filter_cell_count(dataset):
    # 169 cells in the test dataset
    return dataset.n_cells, dataset.n_cells == 169


def no_chr_filter_cell_count(dataset):
    # 152 cells in the test dataset if we only filter by 5k total reads
    return dataset.n_cells, dataset.n_cells == 152


def test_no_filter():
    errors = []
    data_path = str(Path(__file__).parent) + "/data/dataset_configs"
    os.chdir(Path(__file__).parent)

    with open(os.path.join(data_path, "oocyte_zygote_mm10.json"), "r") as f:
        args = json.load(f)
    args["ignore_chr_filter"] = True
    parser = argparse.ArgumentParser()
    args, x, y, depths, batches, dataset, _ = parse_args(
        parser, extra_args=args)
    n_cells, passed = no_chr_filter_cell_count(dataset)
    if not passed:
        errors.append(
            f"Incorrect number of cells ({n_cells} =/= 152) when ignoring cis-reads filter")

    with open(os.path.join(data_path, "oocyte_zygote_mm10.json"), "r") as f:
        args = json.load(f)
    args["ignore_filter"] = True
    args["ignore_chr_filter"] = True
    parser = argparse.ArgumentParser()
    args, x, y, depths, batches, dataset, _ = parse_args(
        parser, extra_args=args)
    n_cells, passed = no_filter_cell_count(dataset)
    if not passed:
        errors.append(
            f"Incorrect number of cells ({n_cells} =/= 169) when ignoring all filters")

    assert not errors, "errors occured:\n{}".format("\n".join(errors))
