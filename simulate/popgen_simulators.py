import os
from typing import Dict, List

import msprime
import numpy as np
import pyslim
import tskit
from allel import read_vcf

from simulate.popgen_simulator import PopGenSimulator


def retrieve_simulator(name: str):
    """Retrieves simulator class by name.

    Parameters
    ----------
    name: str
        Name of the simulator class.
    """
    if name == 'slim':
        return SLiM
    elif name == 'msms':
        return MSMS
    else:
        raise NotImplementedError


class SLiM(PopGenSimulator):

    def _retrieve_base_template_settings(self) -> Dict:
        """
        Returns
        -------
        Dict: Returns settings for slim template that do not change as dict.
        """
        if self.config['template'] == 'msms_match.slim':
            return {'NINDIV': self.config['NINDIV']}
        elif self.config['template'] == 'msms_match_selection.slim':
            return {'NINDIV': self.config['NINDIV'],
                    'SELCOEFF': str(float(self.config['SELCOEFF'])*2)}
        elif self.config['template'] == 'schaffner_model_sweep.slim':
            template = {'SELCOEFF': str(float(self.config['SELCOEFF'])*2),
                        'NINDIV': self.config['NINDIV'],
                        'SWEEPPOP': self.config['SWEEPPOP']}
            pops = [1, 2, 3]
            pops.remove(template['SWEEPPOP'])
            template['NONFOCALPOP1'] = pops[0]
            template['NONFOCALPOP2'] = pops[1]
            assert len(pops) == 2
            return template
        elif self.config['template'] == 'schaffner_model_neutral.slim':
            return {'NINDIV': self.config['NINDIV']}
        elif self.config['template'] == 'soft_sweep.slim':
            return {'NINDIV': self.config['NINDIV'], 'SELCOEFF': str(float(self.config['SELCOEFF'])*2), 'NMutatedIndivs': self.config['NMutatedIndivs']}
        else:
            raise NotImplementedError

    def _msms_match_recapitate(self,
                               rts,
                               alive_inds,
                               template_settings,
                               id_num,
                               id_out_file):
        keep_indivs = np.random.choice(alive_inds, int(
            template_settings['NINDIV']), replace=False)
        keep_nodes = []
        for i in keep_indivs:
            keep_nodes.extend(rts.individual(i).nodes)
        sts = rts.simplify(keep_nodes, keep_input_roots=True)

        # add neutral mutations
        ts = pyslim.SlimTreeSequence(msprime.sim_mutations(
            sts, rate=1.5e-8,
            model=msprime.SLiMMutationModel(type=0),
            keep=True, random_seed=self.seeds[id_num - 1]))

        # output to VCF
        indivlist = []
        for i in ts.individuals_alive_at(0):
            ind = ts.individual(i)
            if ts.node(ind.nodes[0]).is_sample():
                indivlist.append(i)
                assert ts.node(ind.nodes[1]).is_sample()
        with open(id_out_file + '.vcf', 'w') as vcffile:
            ts.write_vcf(vcffile, individuals=indivlist)

    def _schaffner_recapitate(self,
                              rts,
                              alive_inds,
                              template_settings,
                              id_num,
                              id_out_file):

        pop_indivs = [[i for i in alive_inds if rts.individual(i).metadata['subpopulation'] == k]
                      for k in [1, 2, 3]]

        # get vcf file for each population
        for k in [1, 2, 3]:
            keep_indivs = np.random.choice(
                pop_indivs[k - 1], int(template_settings['NINDIV']), replace=False)
            keep_nodes = []
            for i in keep_indivs:
                keep_nodes.extend(rts.individual(i).nodes)
            sts = rts.simplify(keep_nodes, keep_input_roots=True)
            # add neutral mutations
            ts = pyslim.SlimTreeSequence(msprime.sim_mutations(
                sts, rate=2.36e-8,
                model=msprime.SLiMMutationModel(type=0),
                keep=True, random_seed=self.seeds[id_num - 1]))
            # output to VCF
            indivlist = []
            for i in ts.individuals_alive_at(0):
                ind = ts.individual(i)
                if ts.node(ind.nodes[0]).is_sample():
                    indivlist.append(i)
                    assert ts.node(ind.nodes[1]).is_sample()
            with open(id_out_file + '_pop' + str(k) + '.vcf', 'w') as vcffile:
                ts.write_vcf(vcffile, individuals=indivlist)

    def _simulate(self, id_num: int) -> str:
        """Simulates the genetic data. Returns location of file.

        Parameters
        ----------
        id_num: (int) - ID of current simulate

        """
        # Get filenames
        slim_tmp_save_file = f'{self.tmp_simulation_folder}/{id_num:09d}.txt'
        input_file = f'{self.tmp_simulation_folder}/{id_num:09d}.trees'
        id_out_file = f'{self.tmp_simulation_folder}/{id_num:09d}'
        np.random.seed(self.seeds[id_num - 1])

        # Set seed and temporary save point
        template_settings = self.template_settings.copy()
        template_settings['SEED'] = str(self.seeds[id_num-1])
        template_settings['SLIM_TMP_SAVE'] = f'\"{slim_tmp_save_file}\"'

        # If schaffner sweep, output directly to data file, otherwise output tree
        if 'schaffner_model_sweep.slim' == self.config['template']:
            template_settings['OUTPUT'] = f'\"{id_out_file}\"'
        else:
            template_settings['OUTPUT'] = f'\"{input_file}\"'

        # Make slim template file and simulate.
        tmp_settings_file = self._make_template_settings_file(
            template_settings, id_num)
        os.system(
            f'{self.config["software"]} {tmp_settings_file} >/dev/null 2>&1')

        # If not schaffner sweep, then recapitate
        if not 'schaffner_model_sweep.slim' == self.config['template']:
            orig_ts = tskit.load(input_file)  # name of trees file from slim
            # recapitate
            rts = pyslim.recapitate(orig_ts, ancestral_Ne=10000, recombination_rate=1e-8,
                                    random_seed=self.seeds[id_num - 1])
            # simplify
            alive_inds = rts.individuals_alive_at(0)
            if 'msms_match' in self.config['template'] or 'soft_sweep' in self.config['template']:
                self._msms_match_recapitate(
                    rts, alive_inds, template_settings, id_num, id_out_file)
            elif 'schaffner' in self.config['template']:
                self._schaffner_recapitate(
                    rts, alive_inds, template_settings, id_num, id_out_file)

        # Delete extra files and return
        os.remove(tmp_settings_file)
        if os.path.isfile(input_file):
            os.remove(input_file)
        if os.path.isfile(slim_tmp_save_file):
            os.remove(slim_tmp_save_file)
        return id_out_file

    def _load_and_save_sim_data(self, id_out_file: str, id_num: int) -> [np.ndarray, np.ndarray]:
        """Loads the simulate data from given file and returns the image and positions as np array

        Parameters
        ----------
        id_out_file: (str) - Location of simulate file

        Returns
        -------
        [np.ndarray, np.ndarray]: Genetic image and the loci positions as np arrays
        """
        if 'msms_match' in self.config['template'] or 'soft_sweep' in self.config['template']:
            files = [(id_out_file + '.vcf', 'popgen_image', 'popgen_positions')]
        elif 'schaffner' in self.config['template']:
            files = [(id_out_file + '_pop' + str(k) + '.vcf',
                      f'popgen_pop_image{k}', f'popgen_pop_positions{k}') for k in [1, 2, 3]]
        else:
            raise NotImplementedError

        for file in files:
            out = read_vcf(file[0])
            # Get genetic data
            output = out['calldata/GT']
            data = np.zeros((output.shape[1]*output.shape[2], output.shape[0]))
            for i in range(2):
                for j in range(output.shape[1]):
                    data[i*output.shape[1] + j, :] = output[:, j, i]
            image = np.where(data == 0, 0, 1)
            positions = out['variants/POS']
            if id_num == 1:
                self.plot_example_image(image, name=file[1])
            self.save_data(image, id_num, file[1])
            self.save_data(positions, id_num, file[2])
            os.remove(file[0])


class MSMS(PopGenSimulator):

    def _retrieve_base_template_settings(self) -> Dict:
        """
        Returns
        -------
        Dict: Returns settings for msms template that do not change as dict.
        """
        return {'NREF': self.config['NREF'],
                'DEMO': self.config['DEMO'],
                'LEN': self.config['LEN'],
                'THETA': self.config['THETA'],
                'RHO': self.config['RHO'],
                'NCHROMS': self.config['NCHROMS'],
                'SELPOS': self.config['SELPOS'],
                'FREQ': self.config['FREQ'],
                'SELTIME': self.config['SELTIME'],
                'SELCOEFF': self.config['SELCOEFF']
                }

    def _simulate(self, id_num: int) -> str:
        """Simulates the genetic data. Returns location of file.

        Parameters
        ----------
        id_num: (int) - ID of current simulate

        """
        ts = self.template_settings.copy()
        ts['SEED'] = str(self.seeds[id_num-1])
        ts['OUTPUT'] = f'{self.tmp_simulation_folder}/{id_num:09d}.txt'
        tmp_bash_file = f'{self.tmp_simulation_folder}/{id_num:09d}.sh'
        with open(tmp_bash_file, "w") as bash_file:
            bash_file.write('#!/bin/bash\n')
            bash_file.write(f'java -jar $PWD/simulate/software/msms/lib/msms.jar -seed {ts["SEED"]} -N {ts["NREF"]} -ms {ts["NCHROMS"]} 1 '
                            f'-t {ts["THETA"]} -r {ts["RHO"]} {ts["LEN"]} -Sp {ts["SELPOS"]} -SI {ts["SELTIME"]} 1 {ts["FREQ"]} '
                            f'-SAA $(({int(float(float(ts["SELCOEFF"])*int(ts["NREF"])*4))})) -SAa $(({int(float(float(ts["SELCOEFF"])*int(ts["NREF"])*2))})) -Saa 0 '
                            f'-Smark {ts["DEMO"]} -thread 4 > {ts["OUTPUT"]}\n')
        os.system(f'bash {tmp_bash_file}')
        os.remove(tmp_bash_file)
        return f'{self.tmp_simulation_folder}/{id_num:09d}.txt'

    def _load_and_save_sim_data(self, file: str, id_num:  int):
        """Loads the simulate data from given file and returns the image and positions as np array

        Parameters
        ----------
        file: (str) - Location of simulate file
        """
        with open(file, 'r') as sim_file:
            lines = sim_file.readlines()
            segsites = int(lines[4].replace(
                'segsites: ', '').replace('\n', ''))
            image = np.zeros((int(self.config['NCHROMS']), segsites))
            positions = lines[5].replace(
                'positions: ', '').replace('\n', '').split()
            positions = np.asarray([float(position) for position in positions])
            for i, line in enumerate(lines[6:]):
                if line != '\n':
                    image[i, :] = [int(d) if d == '0' or d ==
                                   '1' else 1 for d in line.replace('\n', '')]
        os.remove(file)
        image = np.where(image == 0, 0, 1)
        if id_num == 1:
            self.plot_example_image(image, name='popgen_image')
        self.save_data(image, id_num, 'popgen_image')
        self.save_data(positions, id_num, 'popgen_positions')
        return image, positions


if __name__ == '__main__':
    """For Testing"""

    def simulate(settings: Dict[str, List[Dict[str, str]]],
                 parallel: bool = True,
                 max_sub_processes: int = 10):
        """Simulates the data specified by settings

        Parameters
        ----------
        settings: Dict[List[Dict]] - Dictionary containing the data to be simulated. The first level of the dictionary
            specifies the overall class labels of the data. The List of Dicts at the next level specifies the
            different simulations to run for each of the labels. Each Dict at the last level represents a different
            simulate setting. If data has already been simulated and saved, then this function will not waste time
            re-simulating the data.
        parallel: (bool) - If true, simulations are done in parallel
        max_sub_processes: (int) - Defines max number of simulations that can be run in parallel if parallel is true

        """
        print('Beginning simulations')
        for label, sim_config_list in settings.items():
            print(f'Simulating {label}s')
            for sim_config in sim_config_list:
                print(f'Simulating config: {sim_config}')
                simulator = retrieve_simulator(sim_config['software'])(sim_config, parallel=parallel,
                                                                       max_sub_processes=max_sub_processes)
                simulator.run_simulations()

    # settings = {
    #     'neutral': [
    #         {'software': 'msms',
    #          'NREF': '10000',
    #          'N': 100,
    #          'DEMO': '-eN 0.0875 1 -eN 0.075 0.2 -eN 0 2',
    #          'LEN': '80000',
    #          'THETA': '48',
    #          'RHO': '32',
    #          'NCHROMS': '128',
    #          'SELPOS': '`bc <<< \'scale=2; 1/2\'`',
    #          'FREQ':'`bc <<< \'scale=6; 1/100\'`',
    #          'SELTIME': '`bc <<< \'scale=4; 600/40000\'`',
    #          'SELCOEFF': '0',
    #          },

    #         {'software': 'slim',
    #          'template': 'msms_match.slim',
    #          'N': 10,
    #          'NINDIV': '64'
    #          }

    #     ],
    #     'sweep': [
    #         {'software': 'msms',
    #          'N': 100,
    #          'NREF': '10000',
    #          'DEMO': '-eN 0.0875 1 -eN 0.075 0.2 -eN 0 2',
    #          'LEN': '80000',
    #          'THETA': '48',
    #          'RHO': '32',
    #          'NCHROMS': '128',
    #          'SELPOS': '`bc <<< \'scale=2; 1/2\'`',
    #          'FREQ': '`bc <<< \'scale=6; 1/100\'`',
    #          'SELTIME': '`bc <<< \'scale=4; 600/40000\'`',
    #          'SELCOEFF': '0.01',
    #          },

    #         {'software': 'slim',
    #          'template': 'msms_match_selection.slim',
    #          'N': 10,
    #          'NINDIV': '64',
    #          'SELCOEFF': '0.01',
    #          }
    #     ]
    # }

    # configs = {
    #     'neutral': [
    #         {'software': 'msms',
    #          'NREF': '10000',
    #          'N': 50000,
    #          'DEMO': '-eN 0.0875 1 -eN 0.075 0.2 -eN 0 2',
    #          'LEN': '80000',
    #          'THETA': '48',
    #          'RHO': '32',
    #          'NCHROMS': '1000',
    #          'SELPOS': '`bc <<< \'scale=2; 1/2\'`',
    #          'FREQ': '`bc <<< \'scale=6; 1/100\'`',
    #          'SELTIME': '`bc <<< \'scale=4; 600/40000\'`',
    #          'SELCOEFF': '0',
    #          },
    #     ],
    #     'sweep': [
    #         {'software': 'msms',
    #          'N': 50000,
    #          'NREF': '10000',
    #          'DEMO': '-eN 0.0875 1 -eN 0.075 0.2 -eN 0 2',
    #          'LEN': '80000',
    #          'THETA': '48',
    #          'RHO': '32',
    #          'NCHROMS': '1000',
    #          'SELPOS': '`bc <<< \'scale=2; 1/2\'`',
    #          'FREQ': '`bc <<< \'scale=6; 1/100\'`',
    #          'SELTIME': '`bc <<< \'scale=4; 600/40000\'`',
    #          'SELCOEFF': '0.01',
    #          },
    #     ]
    # }

    # configs = [

    #     {'software': 'slim',
    #      'template': 'schaffner_model_sweep.slim',
    #      'N': 50000,
    #      'NINDIV': '64',
    #      'SELCOEFF': '0.005',
    #      'SWEEPPOP': 1,
    #      },

    #     {'software': 'slim',
    #      'template': 'schaffner_model_sweep.slim',
    #      'N': 50000,
    #      'NINDIV': '64',
    #      'SELCOEFF': '0.005',
    #      'SWEEPPOP': 2,
    #      },

    #     {'software': 'slim',
    #      'template': 'schaffner_model_sweep.slim',
    #      'N': 50000,
    #      'NINDIV': '64',
    #      'SELCOEFF': '0.005',
    #      'SWEEPPOP': 3,
    #      },

    # ]

    configs = [

        {'software': 'slim',
         'template': 'soft_sweep.slim',
         'N': 20000,
         'NINDIV': '64',
         'SELCOEFF': '0.01',
         'NMutatedIndivs': 600,
         },

    ]

    # for config in configs:
    #     simulator = SLiM(config, parallel=False)
    #     simulator.test_simulations()

    for config in configs:
        print(config)
        simulator = SLiM(config, parallel=True, max_sub_processes=5)
        simulator.run_simulations()
    # simulate(settings, parallel=True, max_sub_processes=4)
