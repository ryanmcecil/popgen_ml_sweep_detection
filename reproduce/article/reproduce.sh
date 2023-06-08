#!/bin/bash

# Bash file to reproduce all raw results produced for article: 
# On convolutional neural networks for selection inference:
# revealing the lurking role of preprocessing, and the surprising
# effectiveness of summary statistics

# Run run.py to reproduce results and plots from original analyses
python3 reproduce/article/code/run.py

# Run revision files to reproduce new results for article revisions
for f in reproduce/article/code/revisions/*.py; do python "$f"; done #csv results
for f in reproduce/article/code/revisions/plot/*.py; do python "$f"; done #matplotlib plots

# Extra code to check demographic models
bash reproduce/article/sfs/reproduce.sh