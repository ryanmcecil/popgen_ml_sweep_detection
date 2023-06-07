#!/bin/bash

python3 reproduce/sfs/compute_mean_std.py
python3 reproduce/sfs/slim_msms_sfs.py
python3 reproduce/sfs/thousand_genomes_sfs.py
