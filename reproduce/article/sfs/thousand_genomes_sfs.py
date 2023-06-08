import os
import random

import numpy as np
from allel import read_vcf
from matplotlib import pyplot as plt

if __name__ == '__main__':
    pops = ['YRI_pops.txt', 'CEU_pops.txt']
    for pop in pops:
        file = os.path.join(os.getcwd(), 'reproduce/sfs', pop)
        with open(file) as pop_file:
            samples = pop_file.read().splitlines()
        samples = random.sample(samples, 64)
        print(samples)
        out = read_vcf(os.path.join(os.getcwd(), 'data/genetic/ALL.chr20.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf'),
                       fields=['calldata/GT'],
                       samples=samples)
        output = out['calldata/GT']
        var_counts = np.sum(output, axis=1)
        var_counts = var_counts.flatten()
        length = 128//2
        total_counts = np.bincount(var_counts, minlength=length)
        fig, ax = plt.subplots()

        x = np.arange(length)
        np.delete(x, 0)
        np.delete(total_counts, 0)
        np.delete(x, -1)
        np.delete(total_counts, -1)
        x = x / length
        y = total_counts / np.sum(total_counts)
        ax.plot(x, y, label='real genome')
        ax.autoscale(axis='x', tight=True)
        print('saving plot')
        plt.legend()
        ax.set_xlabel('derived allele frequency')
        ax.set_ylabel('site frequency')
