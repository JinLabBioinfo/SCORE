import os
import sys
import json

def write_higashi_config(name, sizes, cytoband, out_file, base_config, max_dist, resolution, base_dir, n_epochs):
    with open(base_config, 'r') as f:
        d = json.load(f)

    d['config_name'] = name
    d['data_dir'] = base_dir + name + '/'
    d['temp_dir'] = base_dir + name + '/tmp/'
    os.makedirs(d['temp_dir'] + name + '/tmp/', exist_ok=True)
    d['genome_reference_path'] = sizes
    d['cytoband_path'] = cytoband
    d['maximum_distance'] = -1
    d['resolution'] = resolution
    d['resolution_cell'] = resolution
    d['resolution_fh'] = [resolution]
    d['minimum_distance'] = resolution
    d['embedding_epoch'] = n_epochs
    d['no_nbr_epoch'] = n_epochs
    d['with_nbr_epoch'] = n_epochs

    with open(out_file, "w") as f:
        json.dump(d, f)
