#!/bin/bash

# Bash code to reproduce results for reproducing imagene results with both msms and slim
python3 reproduce/imagene/sample_complexity.py
bash reproduce/imagene/msms/reproduce.sh
bash reproduce/imagene/slim/reproduce.sh
