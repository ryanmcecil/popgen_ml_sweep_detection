from reproduce.sfs.compute_mean_std import *
from matplotlib import pyplot as plt
from allel import sfs_folded, read_vcf
import random


def plot_counts(counts, length, ax, label):
    x = np.arange(length)
    x = np.delete(x, 0)
    counts = np.delete(counts, 0)
    x = np.delete(x, -1)
    counts = np.delete(counts, -1)
    x = x / length
    y = counts / np.sum(counts)
    ax[0].plot(x, y, label=label)
    ax[0].set_xlabel('derived allele frequency')
    ax[0].set_ylabel('site frequency')

    ax[1].plot(x[0:14], y[0:14], label=label)
    ax[1].set_xlabel('derived allele frequency')
    ax[1].set_ylabel('site frequency')




if __name__ == '__main__':

    sims = [

             #    {'name': 'msms_neutral',
             #
             #     'settings': {
             #        'software': 'msms',
             #         'NREF': '10000',
             #         'N': 100,
             #         'DEMO': '-eN 0.0875 1 -eN 0.075 0.2 -eN 0 2',
             #         'LEN': '80000',
             #         'THETA': '48',
             #         'RHO': '32',
             #         'NCHROMS': '128',
             #         'SELPOS': '`bc <<< \'scale=2; 1/2\'`',
             #         'FREQ':'`bc <<< \'scale=6; 1/100\'`',
             #         'SELTIME': '`bc <<< \'scale=4; 600/40000\'`',
             #         'SELCOEFF': '0'},
             #
             #    'datatype': 'popgen_image'
             # },
             #
             #    {'name': 'slim_msms_neutral',
             #
             #     'settings': {'software': 'slim',
             #                     'template': 'msms_match.slim',
             #                     'N': 10,
             #                     'NINDIV': '64'
             #                     },
             #
             #     'datatype': 'popgen_image'
             #     },

                {'name': 'slim_neutral_pop1',

                 'settings': {'software': 'slim',
                             'template': 'schaffner_model_neutral.slim',
                             'N': 10000,
                             'NINDIV': '64'
                             },

                 'datatype': 'popgen_pop_image1'
                 },

                {'name': 'slim_neutral_pop2',

                 'settings': {'software': 'slim',
                              'template': 'schaffner_model_neutral.slim',
                              'N': 10000,
                              'NINDIV': '64'
                              },

                 'datatype': 'popgen_pop_image2'
                 },

                # {'name': 'slim_neutral_pop3',
                #
                #  'settings': {'software': 'slim',
                #               'template': 'schaffner_model_neutral.slim',
                #               'N': 10000,
                #               'NINDIV': '64'
                #               },
                #
                #  'datatype': 'popgen_pop_image3'
                #  },
            ]



    plots = {'sfs_folded': sfs_folded}

    N = 10000

    plt.clf()
    plot_name = os.path.join(os.getcwd(), 'reproduce/reproduce/results/sfs', f'neutral_sfs_folded_plot.png')
    fig, ax = plt.subplots(1,2)

    length = 130 // 2

    total_counts = {}
    # Plot sfs for each simulations
    for sim in sims:
        print(f'Computing plot for sim {sim["name"]}, for N={N}')
        simulator = retrieve_simulator(sim['settings']['software'])(sim['settings'])
        counts = np.zeros(length)
        for i in range(N):
            id = i + 1
            image = simulator.load_data(id_num=id, datatype=sim['datatype']).astype(np.int)
            greater_than_1 = image > 1
            image[greater_than_1] = 1
            idx = np.where(np.mean(image, axis=0) > 0.5)[0]
            image[:, idx] = np.abs(1 - image[:, idx])
            var_counts = np.sum(image, axis=0)
            counts += np.bincount(var_counts, minlength=length)
        total_counts[sim['name']] = counts

    # If we are plotting folded, plot thousand genomes for YRi and CEU populations
    pops = ['CEU_pops.txt', 'YRI_pops.txt']
    for pop in pops:
        print(f'Computing sfs for pop {pop}')
        file = os.path.join(os.getcwd(), 'data/pops', pop)
        with open(file) as pop_file:
            samples = pop_file.read().splitlines()
        with open(os.path.join(os.getcwd(), 'reproduce/sfs', '../../sfs/samples.txt')) as samples_file:
            samples_in_file = samples_file.read().split()
        samples = [sample for sample in samples if sample in samples_in_file]
        samples = random.sample(samples, 64)
        out = read_vcf(os.path.join(os.getcwd(), 'data/genetic/ALL.chr20.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf'),
                       fields=['calldata/GT'],
                       samples=samples)
        output = out['calldata/GT']
        output = np.reshape(output, (output.shape[0], output.shape[1] * output.shape[2]))

        # Throwing out columns that have something besides 1 or 0
        where_not_equal_to_zero = np.where(output > 1, 1, 0)
        where_not_equal_to_zero = np.sum(where_not_equal_to_zero, axis=1)
        locs = np.argwhere(where_not_equal_to_zero != 0)
        output = np.delete(output, locs, axis=0)

        idx = np.where(np.mean(output, axis=1) > 0.5)[0]
        output[idx, :] = np.abs(1 - output[idx, :])
        var_counts = np.sum(output, axis=1)
        var_counts = var_counts.flatten()
        counts = np.bincount(var_counts, minlength=length)
        total_counts[pop[:3]] = counts

    for name, count in total_counts.items():
        plot_counts(count, length, ax, name)
    ax[0].autoscale(axis='x', tight=True)
    ax[1].autoscale(axis='x', tight=True)
    print('saving plot')
    print(plot_name)
    plt.legend()
    plt.savefig(plot_name)
