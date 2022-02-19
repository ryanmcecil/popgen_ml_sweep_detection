from simulators.simulators import retrieve_simulator
from processors.popgen_processors import retrieve_processor


def simulate_and_convert(config):
    sim_settings = config['simulations']
    conversions_settings = config['conversions']
    for label, sim_config_list in sim_settings.items():
        for sim_config in sim_config_list:
            print(f'Simulating config: {sim_config}')
            simulator = retrieve_simulator(sim_config['software'])(sim_config, verbose_level=2,
                                                                   parallel=True,
                                                                   max_sub_processes=10)
            simulator.run_simulations()

            for processor_config in conversions_settings:
                processor = retrieve_processor(processor_config['conversion_type'])(config=processor_config,
                                                                                    simulator=simulator,
                                                                                    verbose_level=2,
                                                                                    parallel=True,
                                                                                    max_sub_processes=10)
                processor.run_conversions()





if __name__ == '__main__':
    sim_settings = {
        'neutral': [
            {'software': 'msms',
             'NREF': '10000',
             'N': 1000,
             'DEMO': '-eN 0.0875 1 -eN 0.075 0.2 -eN 0 2',
             'LEN': '80000',
             'THETA': '48',
             'RHO': '32',
             'NCHROMS': '128',
             'SELPOS': '`bc <<< \'scale=2; 1/2\'`',
             'FREQ':'`bc <<< \'scale=6; 1/100\'`',
             'SELTIME': '`bc <<< \'scale=4; 600/40000\'`',
             'SELCOEFF': '0',
             }
        ],
        'sweep': [
            {'software': 'msms',
             'N': 1000,
             'NREF': '10000',
             'DEMO': '-eN 0.0875 1 -eN 0.075 0.2 -eN 0 2',
             'LEN': '80000',
             'THETA': '48',
             'RHO': '32',
             'NCHROMS': '128',
             'SELPOS': '`bc <<< \'scale=2; 1/2\'`',
             'FREQ': '`bc <<< \'scale=6; 1/100\'`',
             'SELTIME': '`bc <<< \'scale=4; 600/40000\'`',
             'SELCOEFF': '0.01',
             }
        ]
    }

    conversion_settings = [{'conversion_type': 'imagene',
                           'sorting': 'Rows',
                           'min_minor_allele_freq': 0.1,
                           'resize_dimensions': 128
    }]

    settings = {
        'simulations': sim_settings,
        'conversions': conversion_settings
    }

    simulate_and_convert(settings)