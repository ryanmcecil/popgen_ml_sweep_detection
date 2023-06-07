# Article Reproduction

This directory contains the code to generate the raw csv accuracy values and images used to make the following [preprint](https://www.biorxiv.org/content/10.1101/2023.02.26.530156v1).

The `code` folder contains all the code used to generate the results. It is split up into

- A list of `.py` files used in `run.py` to generate the main results for the article before the revision round.
- A folder titled `revisions` that contains the `.py` used to generate the new analyses to support our first revisions.
- A folder titled `sfs` that contains the `.py` files used to check that our simulated demographic models mimicked data from the thousand genomes project.

To run the code, call

```
bash reproduce.sh
```

which will create a new results folder in this directory.