"""
Functions define settings for models tested in all experiments
"""
from typing import Dict, List
from models.popgen_summary_statistics import all_image_and_position_statistics


# Dictates conversion/simulation settings for all ML models for all experiments; new results directory will be
# created for each
##########################################
def ml_sim_conversion_training_configs() -> Dict:
    """
    Returns
    -------
    Dict: Key represents name of simulation and conversion type while value is simulation/conversions/training
    settings.
    """
    return {
        #####################################
        # 'imagene_sim_row_sorted_s=0.01': {
        #     'simulations': {
        #         'neutral': [
        #             {'software': 'msms',
        #              'NREF': '10000',
        #              'N': 50000,
        #              'DEMO': '-eN 0.0875 1 -eN 0.075 0.2 -eN 0 2',
        #              'LEN': '80000',
        #              'THETA': '48',
        #              'RHO': '32',
        #              'NCHROMS': '128',
        #              'SELPOS': '`bc <<< \'scale=2; 1/2\'`',
        #              'FREQ': '`bc <<< \'scale=6; 1/100\'`',
        #              'SELTIME': '`bc <<< \'scale=4; 600/40000\'`',
        #              'SELCOEFF': '0',
        #              }
        #         ],
        #         'sweep': [
        #             {'software': 'msms',
        #              'N': 50000,
        #              'NREF': '10000',
        #              'DEMO': '-eN 0.0875 1 -eN 0.075 0.2 -eN 0 2',
        #              'LEN': '80000',
        #              'THETA': '48',
        #              'RHO': '32',
        #              'NCHROMS': '128',
        #              'SELPOS': '`bc <<< \'scale=2; 1/2\'`',
        #              'FREQ': '`bc <<< \'scale=6; 1/100\'`',
        #              'SELTIME': '`bc <<< \'scale=4; 600/40000\'`',
        #              'SELCOEFF': '0.01',
        #              }
        #         ]
        #     },
        #     'conversions': [{'conversion_type': 'imagene',
        #                      'sorting': 'Rows',
        #                      'min_minor_allele_freq': 0.01,
        #                      'resize_dimensions': 128
        #                      }],
        #     'training': {'epochs': 2,
        #                  'batch_size': 64,
        #                  'train_proportion': 0.8,
        #                  'validate_proportion': 0.1,
        #                  'test_proportion': 0.1,
        #                  'best_of': 10
        #                  }},
        ###############################################
        'imagene_sim_row_sorted_zero_padding_s=0.01': {
            'simulations': {
                'neutral': [
                    {'software': 'msms',
                     'NREF': '10000',
                     'N': 50000,
                     'DEMO': '-eN 0.0875 1 -eN 0.075 0.2 -eN 0 2',
                     'LEN': '80000',
                     'THETA': '48',
                     'RHO': '32',
                     'NCHROMS': '128',
                     'SELPOS': '`bc <<< \'scale=2; 1/2\'`',
                     'FREQ': '`bc <<< \'scale=6; 1/100\'`',
                     'SELTIME': '`bc <<< \'scale=4; 600/40000\'`',
                     'SELCOEFF': '0',
                     }
                ],
                'sweep': [
                    {'software': 'msms',
                     'N': 50000,
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
            },
            'conversions': [{'conversion_type': 'zero_padding_imagene',
                             'sorting': 'Rows',
                             'min_minor_allele_freq': 0.01,
                             'resize_dimensions': None,
                             }],
            'training': {'epochs': 2,
                         'batch_size': 64,
                         'train_proportion': 0.8,
                         'validate_proportion': 0.1,
                         'test_proportion': 0.1,
                         'best_of': 10
                         }},
        ####################################
        # 'imagene_sim_row_sorted_s=0.005': {
        #     'simulations': {
        #         'neutral': [
        #             {'software': 'msms',
        #              'NREF': '10000',
        #              'N': 50000,
        #              'DEMO': '-eN 0.0875 1 -eN 0.075 0.2 -eN 0 2',
        #              'LEN': '80000',
        #              'THETA': '48',
        #              'RHO': '32',
        #              'NCHROMS': '128',
        #              'SELPOS': '`bc <<< \'scale=2; 1/2\'`',
        #              'FREQ': '`bc <<< \'scale=6; 1/100\'`',
        #              'SELTIME': '`bc <<< \'scale=4; 600/40000\'`',
        #              'SELCOEFF': '0',
        #              }
        #         ],
        #         'sweep': [
        #             {'software': 'msms',
        #              'N': 50000,
        #              'NREF': '10000',
        #              'DEMO': '-eN 0.0875 1 -eN 0.075 0.2 -eN 0 2',
        #              'LEN': '80000',
        #              'THETA': '48',
        #              'RHO': '32',
        #              'NCHROMS': '128',
        #              'SELPOS': '`bc <<< \'scale=2; 1/2\'`',
        #              'FREQ': '`bc <<< \'scale=6; 1/100\'`',
        #              'SELTIME': '`bc <<< \'scale=4; 600/40000\'`',
        #              'SELCOEFF': '0.005',
        #              }
        #         ]
        #     },
        #     'conversions': [{'conversion_type': 'imagene',
        #                      'sorting': 'Rows',
        #                      'min_minor_allele_freq': 0.01,
        #                      'resize_dimensions': 128
        #                      }],
        #     'training': {'epochs': 2,
        #                  'batch_size': 64,
        #                  'train_proportion': 0.8,
        #                  'validate_proportion': 0.1,
        #                  'test_proportion': 0.1,
        #                  'best_of': 10
        #                  }},
        #################################################
        'imagene_sim_row_sorted_zero_padding_s=0.005': {
            'simulations': {
                'neutral': [
                    {'software': 'msms',
                     'NREF': '10000',
                     'N': 50000,
                     'DEMO': '-eN 0.0875 1 -eN 0.075 0.2 -eN 0 2',
                     'LEN': '80000',
                     'THETA': '48',
                     'RHO': '32',
                     'NCHROMS': '128',
                     'SELPOS': '`bc <<< \'scale=2; 1/2\'`',
                     'FREQ': '`bc <<< \'scale=6; 1/100\'`',
                     'SELTIME': '`bc <<< \'scale=4; 600/40000\'`',
                     'SELCOEFF': '0',
                     }
                ],
                'sweep': [
                    {'software': 'msms',
                     'N': 50000,
                     'NREF': '10000',
                     'DEMO': '-eN 0.0875 1 -eN 0.075 0.2 -eN 0 2',
                     'LEN': '80000',
                     'THETA': '48',
                     'RHO': '32',
                     'NCHROMS': '128',
                     'SELPOS': '`bc <<< \'scale=2; 1/2\'`',
                     'FREQ': '`bc <<< \'scale=6; 1/100\'`',
                     'SELTIME': '`bc <<< \'scale=4; 600/40000\'`',
                     'SELCOEFF': '0.005',
                     }
                ]
            },
            'conversions': [{'conversion_type': 'zero_padding_imagene',
                             'sorting': 'Rows',
                             'min_minor_allele_freq': 0.01,
                             'resize_dimensions': None,
                             }],
            'training': {'epochs': 2,
                         'batch_size': 64,
                         'train_proportion': 0.8,
                         'validate_proportion': 0.1,
                         'test_proportion': 0.1,
                         'best_of': 10
                         }},
        << << << < HEAD
        ###############################################
        # 'schaffner_sweep_pop1_row_sorted_s=0.01': {
        #     'simulations': {
        #         'neutral': [
        #             {'software': 'slim',
        #              'template': 'schaffner_model_neutral.slim',
        #              'N': 10000,
        #              'NINDIV': '64'
        #              }
        #         ],
        #         'sweep': [
        #             {'software': 'slim',
        #              'template': 'schaffner_model_sweep.slim',
        #              'N': 10000,
        #              'NINDIV': '64',
        #              'SELCOEFF': '0.01',
        #              'SWEEPPOP': 1,
        #              }
        #         ]
        #     },
        #     'conversions': [{'conversion_type': 'imagene',
        #                      'sorting': 'Rows',
        #                      'min_minor_allele_freq': 0.01,
        #                      'resize_dimensions': 128,
        #                      'pop': 1
        #                      }],
        #     'training': {'epochs': 2,
        #                  'batch_size': 64,
        #                  'train_proportion': 0.8,
        #                  'validate_proportion': 0.1,
        #                  'test_proportion': 0.1,
        #                  'best_of': 10
        #                  }},
        # ###############################################
        # 'schaffner_sweep_pop2_row_sorted_s=0.01': {
        #     'simulations': {
        #         'neutral': [
        #             {'software': 'slim',
        #              'template': 'schaffner_model_neutral.slim',
        #              'N': 10000,
        #              'NINDIV': '64'
        #              }
        #         ],
        #         'sweep': [
        #             {'software': 'slim',
        #              'template': 'schaffner_model_sweep.slim',
        #              'N': 10000,
        #              'NINDIV': '64',
        #              'SELCOEFF': '0.01',
        #              'SWEEPPOP': 2,
        #              }
        #         ]
        #     },
        #     'conversions': [{'conversion_type': 'imagene',
        #                      'sorting': 'Rows',
        #                      'min_minor_allele_freq': 0.01,
        #                      'resize_dimensions': 128,
        #                      'pop': 2
        #                      }],
        #     'training': {'epochs': 2,
        #                  'batch_size': 64,
        #                  'train_proportion': 0.8,
        #                  'validate_proportion': 0.1,
        #                  'test_proportion': 0.1,
        #                  'best_of': 10
        #                  }
        # },
        ###############################################
        # 'schaffner_sweep_pop3_row_sorted_s=0.01': {
        #     'simulations': {
        #         'neutral': [
        #             {'software': 'slim',
        #              'template': 'schaffner_model_neutral.slim',
        #              'N': 10000,
        #              'NINDIV': '64'
        #              }
        #         ],
        #         'sweep': [
        #             {'software': 'slim',
        #              'template': 'schaffner_model_sweep.slim',
        #              'N': 10000,
        #              'NINDIV': '64',
        #              'SELCOEFF': '0.01',
        #              'SWEEPPOP': 3,
        #              }
        #         ]
        #     },
        #     'conversions': [{'conversion_type': 'imagene',
        #                      'sorting': 'Rows',
        #                      'min_minor_allele_freq': 0.01,
        #                      'resize_dimensions': 128,
        #                      'pop': 3
        #                      }],
        #     'training': {'epochs': 2,
        #                  'batch_size': 64,
        #                  'train_proportion': 0.8,
        #                  'validate_proportion': 0.1,
        #                  'test_proportion': 0.1,
        #                  'best_of': 10
        #                  }
        # },
        == == == =
        ################################################
        'schaffner_sweep_pop1_row_sorted_s=0.01': {
            'simulations': {
                'neutral': [
                    {'software': 'slim',
                     'template': 'schaffner_model_neutral.slim',
                     'N': 10000,
                     'NINDIV': '64'
                     }
                ],
                'sweep': [
                    {'software': 'slim',
                     'template': 'schaffner_model_sweep.slim',
                     'N': 10000,
                     'NINDIV': '64',
                     'SELCOEFF': '0.01',
                     'SWEEPPOP': 1,
                     }
                ]
            },
            'conversions': [{'conversion_type': 'imagene',
                             'sorting': 'Rows',
                             'min_minor_allele_freq': 0.01,
                             'resize_dimensions': 128,
                             'pop': 1
                             }],
            'training': {'epochs': 2,
                         'batch_size': 64,
                         'train_proportion': 0.8,
                         'validate_proportion': 0.1,
                         'test_proportion': 0.1,
                         'best_of': 10
                         }},
        ################################################
        'schaffner_sweep_pop2_row_sorted_s=0.01': {
            'simulations': {
                'neutral': [
                    {'software': 'slim',
                     'template': 'schaffner_model_neutral.slim',
                     'N': 10000,
                     'NINDIV': '64'
                     }
                ],
                'sweep': [
                    {'software': 'slim',
                     'template': 'schaffner_model_sweep.slim',
                     'N': 10000,
                     'NINDIV': '64',
                     'SELCOEFF': '0.01',
                     'SWEEPPOP': 2,
                     }
                ]
            },
            'conversions': [{'conversion_type': 'imagene',
                             'sorting': 'Rows',
                             'min_minor_allele_freq': 0.01,
                             'resize_dimensions': 128,
                             'pop': 2
                             }],
            'training': {'epochs': 2,
                         'batch_size': 64,
                         'train_proportion': 0.8,
                         'validate_proportion': 0.1,
                         'test_proportion': 0.1,
                         'best_of': 10
                         }
        },
        ################################################
        'schaffner_sweep_pop3_row_sorted_s=0.01': {
            'simulations': {
                'neutral': [
                    {'software': 'slim',
                     'template': 'schaffner_model_neutral.slim',
                     'N': 10000,
                     'NINDIV': '64'
                     }
                ],
                'sweep': [
                    {'software': 'slim',
                     'template': 'schaffner_model_sweep.slim',
                     'N': 10000,
                     'NINDIV': '64',
                     'SELCOEFF': '0.01',
                     'SWEEPPOP': 3,
                     }
                ]
            },
            'conversions': [{'conversion_type': 'imagene',
                             'sorting': 'Rows',
                             'min_minor_allele_freq': 0.01,
                             'resize_dimensions': 128,
                             'pop': 3
                             }],
            'training': {'epochs': 2,
                         'batch_size': 64,
                         'train_proportion': 0.8,
                         'validate_proportion': 0.1,
                         'test_proportion': 0.1,
                         'best_of': 10
                         }
        },
        >>>>>> > 3ab9be5a0e89e17043c9d1df756f03b0d458ce83
        #################################################
        'schaffner_sweep_pop1_row_sorted_zero_padding_s=0.01': {
            'simulations': {
                'neutral': [
                    {'software': 'slim',
                     'template': 'schaffner_model_neutral.slim',
                     'N': 10000,
                     'NINDIV': '64'
                     }
                ],
                'sweep': [
                    {'software': 'slim',
                     'template': 'schaffner_model_sweep.slim',
                     'N': 10000,
                     'NINDIV': '64',
                     'SELCOEFF': '0.01',
                     'SWEEPPOP': 1,
                     }
                ]
            },
            'conversions': [{'conversion_type': 'zero_padding_imagene',
                             'sorting': 'Rows',
                             'pop': 1,
                             'min_minor_allele_freq': 0.01,
                             'resize_dimensions': None,
                             }],
            'training': {'epochs': 2,
                         'batch_size': 32,
                         'train_proportion': 0.8,
                         'validate_proportion': 0.1,
                         'test_proportion': 0.1,
                         'best_of': 10
                         }},
        #################################################
        'schaffner_sweep_pop2_row_sorted_zero_padding_s=0.01': {
            'simulations': {
                'neutral': [
                    {'software': 'slim',
                     'template': 'schaffner_model_neutral.slim',
                     'N': 10000,
                     'NINDIV': '64'
                     }
                ],
                'sweep': [
                    {'software': 'slim',
                     'template': 'schaffner_model_sweep.slim',
                     'N': 10000,
                     'NINDIV': '64',
                     'SELCOEFF': '0.01',
                     'SWEEPPOP': 2,
                     }
                ]
            },
            'conversions': [{'conversion_type': 'zero_padding_imagene',
                             'sorting': 'Rows',
                             'pop': 2,
                             'min_minor_allele_freq': 0.01,
                             'resize_dimensions': None,
                             }],
            'training': {'epochs': 2,
                         'batch_size': 32,
                         'train_proportion': 0.8,
                         'validate_proportion': 0.1,
                         'test_proportion': 0.1,
                         'best_of': 10
                         }},
        #################################################
        'schaffner_sweep_pop3_row_sorted_zero_padding_s=0.01': {
            'simulations': {
                'neutral': [
                    {'software': 'slim',
                     'template': 'schaffner_model_neutral.slim',
                     'N': 10000,
                     'NINDIV': '64'
                     }
                ],
                'sweep': [
                    {'software': 'slim',
                     'template': 'schaffner_model_sweep.slim',
                     'N': 10000,
                     'NINDIV': '64',
                     'SELCOEFF': '0.01',
                     'SWEEPPOP': 3,
                     }
                ]
            },
            'conversions': [{'conversion_type': 'zero_padding_imagene',
                             'sorting': 'Rows',
                             'pop': 3,
                             'min_minor_allele_freq': 0.01,
                             'resize_dimensions': None
                             }],
            'training': {'epochs': 2,
                         'batch_size': 32,
                         'train_proportion': 0.8,
                         'validate_proportion': 0.1,
                         'test_proportion': 0.1,
                         'best_of': 10
                         }},
        ################################################
        << << << < HEAD
        # 'schaffner_sweep_pop1_row_sorted_s=0.005': {
        #     'simulations': {
        #         'neutral': [
        #             {'software': 'slim',
        #              'template': 'schaffner_model_neutral.slim',
        #              'N': 10000,
        #              'NINDIV': '64'
        #              }
        #         ],
        #         'sweep': [
        #             {'software': 'slim',
        #              'template': 'schaffner_model_sweep.slim',
        #              'N': 10000,
        #              'NINDIV': '64',
        #              'SELCOEFF': '0.005',
        #              'SWEEPPOP': 1,
        #              }
        #         ]
        #     },
        #     'conversions': [{'conversion_type': 'imagene',
        #                      'sorting': 'Rows',
        #                      'min_minor_allele_freq': 0.01,
        #                      'resize_dimensions': 128,
        #                      'pop': 1
        #                      }],
        #     'training': {'epochs': 2,
        #                  'batch_size': 64,
        #                  'train_proportion': 0.8,
        #                  'validate_proportion': 0.1,
        #                  'test_proportion': 0.1,
        #                  'best_of': 10
        #                  }},
        # ################################################
        # 'schaffner_sweep_pop2_row_sorted_s=0.005': {
        #     'simulations': {
        #         'neutral': [
        #             {'software': 'slim',
        #              'template': 'schaffner_model_neutral.slim',
        #              'N': 10000,
        #              'NINDIV': '64'
        #              }
        #         ],
        #         'sweep': [
        #             {'software': 'slim',
        #              'template': 'schaffner_model_sweep.slim',
        #              'N': 10000,
        #              'NINDIV': '64',
        #              'SELCOEFF': '0.005',
        #              'SWEEPPOP': 2,
        #              }
        #         ]
        #     },
        #     'conversions': [{'conversion_type': 'imagene',
        #                      'sorting': 'Rows',
        #                      'min_minor_allele_freq': 0.01,
        #                      'resize_dimensions': 128,
        #                      'pop': 2
        #                      }],
        #     'training': {'epochs': 2,
        #                  'batch_size': 64,
        #                  'train_proportion': 0.8,
        #                  'validate_proportion': 0.1,
        #                  'test_proportion': 0.1,
        #                  'best_of': 10
        #                  }
        # },
        ################################################
        # 'schaffner_sweep_pop3_row_sorted_s=0.005': {
        #     'simulations': {
        #         'neutral': [
        #             {'software': 'slim',
        #              'template': 'schaffner_model_neutral.slim',
        #              'N': 10000,
        #              'NINDIV': '64'
        #              }
        #         ],
        #         'sweep': [
        #             {'software': 'slim',
        #              'template': 'schaffner_model_sweep.slim',
        #              'N': 10000,
        #              'NINDIV': '64',
        #              'SELCOEFF': '0.005',
        #              'SWEEPPOP': 3,
        #              }
        #         ]
        #     },
        #     'conversions': [{'conversion_type': 'imagene',
        #                      'sorting': 'Rows',
        #                      'min_minor_allele_freq': 0.01,
        #                      'resize_dimensions': 128,
        #                      'pop': 3
        #                      }],
        #     'training': {'epochs': 2,
        #                  'batch_size': 64,
        #                  'train_proportion': 0.8,
        #                  'validate_proportion': 0.1,
        #                  'test_proportion': 0.1,
        #                  'best_of': 10
        #                  }
        # },
        == == == =
        'schaffner_sweep_pop1_row_sorted_s=0.005': {
            'simulations': {
                'neutral': [
                    {'software': 'slim',
                     'template': 'schaffner_model_neutral.slim',
                     'N': 10000,
                     'NINDIV': '64'
                     }
                ],
                'sweep': [
                    {'software': 'slim',
                     'template': 'schaffner_model_sweep.slim',
                     'N': 10000,
                     'NINDIV': '64',
                     'SELCOEFF': '0.005',
                     'SWEEPPOP': 1,
                     }
                ]
            },
            'conversions': [{'conversion_type': 'imagene',
                             'sorting': 'Rows',
                             'min_minor_allele_freq': 0.01,
                             'resize_dimensions': 128,
                             'pop': 1
                             }],
            'training': {'epochs': 2,
                         'batch_size': 64,
                         'train_proportion': 0.8,
                         'validate_proportion': 0.1,
                         'test_proportion': 0.1,
                         'best_of': 10
                         }},
        #################################################
        'schaffner_sweep_pop2_row_sorted_s=0.005': {
            'simulations': {
                'neutral': [
                    {'software': 'slim',
                     'template': 'schaffner_model_neutral.slim',
                     'N': 10000,
                     'NINDIV': '64'
                     }
                ],
                'sweep': [
                    {'software': 'slim',
                     'template': 'schaffner_model_sweep.slim',
                     'N': 10000,
                     'NINDIV': '64',
                     'SELCOEFF': '0.005',
                     'SWEEPPOP': 2,
                     }
                ]
            },
            'conversions': [{'conversion_type': 'imagene',
                             'sorting': 'Rows',
                             'min_minor_allele_freq': 0.01,
                             'resize_dimensions': 128,
                             'pop': 2
                             }],
            'training': {'epochs': 2,
                         'batch_size': 64,
                         'train_proportion': 0.8,
                         'validate_proportion': 0.1,
                         'test_proportion': 0.1,
                         'best_of': 10
                         }
        },
        #################################################
        'schaffner_sweep_pop3_row_sorted_s=0.005': {
            'simulations': {
                'neutral': [
                    {'software': 'slim',
                     'template': 'schaffner_model_neutral.slim',
                     'N': 10000,
                     'NINDIV': '64'
                     }
                ],
                'sweep': [
                    {'software': 'slim',
                     'template': 'schaffner_model_sweep.slim',
                     'N': 10000,
                     'NINDIV': '64',
                     'SELCOEFF': '0.005',
                     'SWEEPPOP': 3,
                     }
                ]
            },
            'conversions': [{'conversion_type': 'imagene',
                             'sorting': 'Rows',
                             'min_minor_allele_freq': 0.01,
                             'resize_dimensions': 128,
                             'pop': 3
                             }],
            'training': {'epochs': 2,
                         'batch_size': 64,
                         'train_proportion': 0.8,
                         'validate_proportion': 0.1,
                         'test_proportion': 0.1,
                         'best_of': 10
                         }
        },
        >>>>>> > 3ab9be5a0e89e17043c9d1df756f03b0d458ce83
        #################################################
        'schaffner_sweep_pop1_row_sorted_zero_padding_s=0.005': {
            'simulations': {
                'neutral': [
                    {'software': 'slim',
                     'template': 'schaffner_model_neutral.slim',
                     'N': 10000,
                     'NINDIV': '64'
                     }
                ],
                'sweep': [
                    {'software': 'slim',
                     'template': 'schaffner_model_sweep.slim',
                     'N': 10000,
                     'NINDIV': '64',
                     'SELCOEFF': '0.005',
                     'SWEEPPOP': 1,
                     }
                ]
            },
            'conversions': [{'conversion_type': 'zero_padding_imagene',
                             'sorting': 'Rows',
                             'pop': 1,
                             'min_minor_allele_freq': 0.01,
                             'resize_dimensions': None
                             }],
            'training': {'epochs': 2,
                         'batch_size': 32,
                         'train_proportion': 0.8,
                         'validate_proportion': 0.1,
                         'test_proportion': 0.1,
                         'best_of': 10
                         }},
        #################################################
        'schaffner_sweep_pop2_row_sorted_zero_padding_s=0.005': {
            'simulations': {
                'neutral': [
                    {'software': 'slim',
                     'template': 'schaffner_model_neutral.slim',
                     'N': 10000,
                     'NINDIV': '64'
                     }
                ],
                'sweep': [
                    {'software': 'slim',
                     'template': 'schaffner_model_sweep.slim',
                     'N': 10000,
                     'NINDIV': '64',
                     'SELCOEFF': '0.005',
                     'SWEEPPOP': 2,
                     }
                ]
            },
            'conversions': [{'conversion_type': 'zero_padding_imagene',
                             'sorting': 'Rows',
                             'pop': 2,
                             'min_minor_allele_freq': 0.01,
                             'resize_dimensions': None
                             }],
            'training': {'epochs': 2,
                         'batch_size': 32,
                         'train_proportion': 0.8,
                         'validate_proportion': 0.1,
                         'test_proportion': 0.1,
                         'best_of': 10
                         }},
        #################################################
        'schaffner_sweep_pop3_row_sorted_zero_padding_s=0.005': {
            'simulations': {
                'neutral': [
                    {'software': 'slim',
                     'template': 'schaffner_model_neutral.slim',
                     'N': 10000,
                     'NINDIV': '64'
                     }
                ],
                'sweep': [
                    {'software': 'slim',
                     'template': 'schaffner_model_sweep.slim',
                     'N': 10000,
                     'NINDIV': '64',
                     'SELCOEFF': '0.005',
                     'SWEEPPOP': 3,
                     }
                ]
            },
            'conversions': [{'conversion_type': 'zero_padding_imagene',
                             'sorting': 'Rows',
                             'pop': 3,
                             'min_minor_allele_freq': 0.01,
                             'resize_dimensions': None
                             }],
            'training': {'epochs': 2,
                         'batch_size': 32,
                         'train_proportion': 0.8,
                         'validate_proportion': 0.1,
                         'test_proportion': 0.1,
                         'best_of': 10
                         }},
        << << << < HEAD
        # 'schaffner_sweep_pop1_row_sorted_s=0.01_N=100000': {
        #     'simulations': {
        #         'neutral': [
        #             {'software': 'slim',
        #              'template': 'schaffner_model_neutral.slim',
        #              'N': 50000,
        #              'NINDIV': '64'
        #              }
        #         ],
        #         'sweep': [
        #             {'software': 'slim',
        #              'template': 'schaffner_model_sweep.slim',
        #              'N': 50000,
        #              'NINDIV': '64',
        #              'SELCOEFF': '0.01',
        #              'SWEEPPOP': 1,
        #              }
        #         ]
        #     },
        #     'conversions': [{'conversion_type': 'imagene',
        #                      'sorting': 'Rows',
        #                      'min_minor_allele_freq': 0.01,
        #                      'resize_dimensions': 128,
        #                      'pop': 1
        #                      }],
        #     'training': {'epochs': 2,
        #                  'batch_size': 64,
        #                  'train_proportion': 0.8,
        #                  'validate_proportion': 0.1,
        #                  'test_proportion': 0.1,
        #                  'best_of': 10
        #                  }},
        # # ################################################
        # 'schaffner_sweep_pop2_row_sorted_s=0.01_N=100000': {
        #     'simulations': {
        #         'neutral': [
        #             {'software': 'slim',
        #              'template': 'schaffner_model_neutral.slim',
        #              'N': 50000,
        #              'NINDIV': '64'
        #              }
        #         ],
        #         'sweep': [
        #             {'software': 'slim',
        #              'template': 'schaffner_model_sweep.slim',
        #              'N': 50000,
        #              'NINDIV': '64',
        #              'SELCOEFF': '0.01',
        #              'SWEEPPOP': 2,
        #              }
        #         ]
        #     },
        #     'conversions': [{'conversion_type': 'imagene',
        #                      'sorting': 'Rows',
        #                      'min_minor_allele_freq': 0.01,
        #                      'resize_dimensions': 128,
        #                      'pop': 2
        #                      }],
        #     'training': {'epochs': 2,
        #                  'batch_size': 64,
        #                  'train_proportion': 0.8,
        #                  'validate_proportion': 0.1,
        #                  'test_proportion': 0.1,
        #                  'best_of': 10
        #                  }
        # },
        # ###############################################
        # 'schaffner_sweep_pop3_row_sorted_s=0.01_N=100000': {
        #     'simulations': {
        #         'neutral': [
        #             {'software': 'slim',
        #              'template': 'schaffner_model_neutral.slim',
        #              'N': 50000,
        #              'NINDIV': '64'
        #              }
        #         ],
        #         'sweep': [
        #             {'software': 'slim',
        #              'template': 'schaffner_model_sweep.slim',
        #              'N': 50000,
        #              'NINDIV': '64',
        #              'SELCOEFF': '0.01',
        #              'SWEEPPOP': 3,
        #              }
        #         ]
        #     },
        #     'conversions': [{'conversion_type': 'imagene',
        #                      'sorting': 'Rows',
        #                      'min_minor_allele_freq': 0.01,
        #                      'resize_dimensions': 128,
        #                      'pop': 3
        #                      }],
        #     'training': {'epochs': 2,
        #                  'batch_size': 64,
        #                  'train_proportion': 0.8,
        #                  'validate_proportion': 0.1,
        #                  'test_proportion': 0.1,
        #                  'best_of': 10
        #                  }
        # },
        == == == =
        'schaffner_sweep_pop1_row_sorted_s=0.01_N=100000': {
            'simulations': {
                'neutral': [
                    {'software': 'slim',
                     'template': 'schaffner_model_neutral.slim',
                     'N': 50000,
                     'NINDIV': '64'
                     }
                ],
                'sweep': [
                    {'software': 'slim',
                     'template': 'schaffner_model_sweep.slim',
                     'N': 50000,
                     'NINDIV': '64',
                     'SELCOEFF': '0.01',
                     'SWEEPPOP': 1,
                     }
                ]
            },
            'conversions': [{'conversion_type': 'imagene',
                             'sorting': 'Rows',
                             'min_minor_allele_freq': 0.01,
                             'resize_dimensions': 128,
                             'pop': 1
                             }],
            'training': {'epochs': 2,
                         'batch_size': 64,
                         'train_proportion': 0.8,
                         'validate_proportion': 0.1,
                         'test_proportion': 0.1,
                         'best_of': 10
                         }},
        # ################################################
        'schaffner_sweep_pop2_row_sorted_s=0.01_N=100000': {
            'simulations': {
                'neutral': [
                    {'software': 'slim',
                     'template': 'schaffner_model_neutral.slim',
                     'N': 50000,
                     'NINDIV': '64'
                     }
                ],
                'sweep': [
                    {'software': 'slim',
                     'template': 'schaffner_model_sweep.slim',
                     'N': 50000,
                     'NINDIV': '64',
                     'SELCOEFF': '0.01',
                     'SWEEPPOP': 2,
                     }
                ]
            },
            'conversions': [{'conversion_type': 'imagene',
                             'sorting': 'Rows',
                             'min_minor_allele_freq': 0.01,
                             'resize_dimensions': 128,
                             'pop': 2
                             }],
            'training': {'epochs': 2,
                         'batch_size': 64,
                         'train_proportion': 0.8,
                         'validate_proportion': 0.1,
                         'test_proportion': 0.1,
                         'best_of': 10
                         }
        },
        ###############################################
        'schaffner_sweep_pop3_row_sorted_s=0.01_N=100000': {
            'simulations': {
                'neutral': [
                    {'software': 'slim',
                     'template': 'schaffner_model_neutral.slim',
                     'N': 50000,
                     'NINDIV': '64'
                     }
                ],
                'sweep': [
                    {'software': 'slim',
                     'template': 'schaffner_model_sweep.slim',
                     'N': 50000,
                     'NINDIV': '64',
                     'SELCOEFF': '0.01',
                     'SWEEPPOP': 3,
                     }
                ]
            },
            'conversions': [{'conversion_type': 'imagene',
                             'sorting': 'Rows',
                             'min_minor_allele_freq': 0.01,
                             'resize_dimensions': 128,
                             'pop': 3
                             }],
            'training': {'epochs': 2,
                         'batch_size': 64,
                         'train_proportion': 0.8,
                         'validate_proportion': 0.1,
                         'test_proportion': 0.1,
                         'best_of': 10
                         }
        },
        >>>>>> > 3ab9be5a0e89e17043c9d1df756f03b0d458ce83
    }


# Defines ML architectures for Imagene architecture analysis
##############################################
def imagene_arch_analysis_configs(width: int) -> Dict:
    """Returns all configs tested in architecture analysis

    Parameters
    ----------
    width: (int) - Width of converted images

    Returns
    -------
    Dict: Key is name of architecture while value stores ML archiecture config

    """
    return {'Imagene':
            << << << < HEAD
            {
                'type': 'ml',
                'name': 'imagene',
                'image_height': 128,
                'image_width': width,
                'relu': True,
                'max_pooling': True,
                'convolution': True,
                'filters': 32,
                'depth': 3,
                'kernel_height': 3,
                'kernel_width': 3,
                'num_dense_layers': 1
            },

            '3 Conv Layers (32 3x3 kernels) + Max Pooling + Relu -> Dense (1 unit) + Sigmoid':
            == == == =
            {
                'type': 'ml',
                'name': 'imagene',
                'image_height': 128,
                'image_width': width,
                'relu': True,
                'max_pooling': True,
                'convolution': True,
                'filters': 32,
                'depth': 3,
                'kernel_height': 3,
                'kernel_width': 3,
                'num_dense_layers': 1
            },

            '3 Conv Layers (32 3x3 kernels) + Max Pooling + Relu -> Dense (1 unit) + Sigmoid':
            >>>>>> > 3ab9be5a0e89e17043c9d1df756f03b0d458ce83
            {
                'type': 'ml',
                'name': 'imagene',
                'image_height': 128,
                'image_width': width,
                'relu': True,
                'max_pooling': True,
                'convolution': True,
                'filters': 32,
                'depth': 3,
                'kernel_height': 3,
                'kernel_width': 3,
                'num_dense_layers': 0
            },

            << << << < HEAD
            '2 Conv Layers (32 3x3 kernels) + Max Pooling + Relu -> Dense (1 unit) + Sigmoid':
            == == == =
            '2 Conv Layers (32 3x3 kernels) + Max Pooling + Relu -> Dense (1 unit) + Sigmoid':
            >>>>>> > 3ab9be5a0e89e17043c9d1df756f03b0d458ce83
            {
                'type': 'ml',
                'name': 'imagene',
                'image_height': 128,
                'image_width': width,
                'relu': True,
                'max_pooling': True,
                'convolution': True,
                'filters': 32,
                'depth': 2,
                'kernel_height': 3,
                'kernel_width': 3,
                'num_dense_layers': 0
            },

            << << << < HEAD
            '1 Conv Layers (32 3x3 kernels) + Max Pooling + Relu -> Dense (1 unit) + Sigmoid':
            == == == =
            '1 Conv Layers (32 3x3 kernels) + Max Pooling + Relu -> Dense (1 unit) + Sigmoid':
            >>>>>> > 3ab9be5a0e89e17043c9d1df756f03b0d458ce83
            {
                'type': 'ml',
                'name': 'imagene',
                'image_height': 128,
                'image_width': width,
                'relu': True,
                'max_pooling': True,
                'convolution': True,
                'filters': 32,
                'depth': 1,
                'kernel_height': 3,
                'kernel_width': 3,
                'num_dense_layers': 0
            },

            << << << < HEAD
            '1 Conv Layers (16 3x3 kernels) + Max Pooling + Relu -> Dense (1 unit) + Sigmoid':
            == == == =
            '1 Conv Layers (16 3x3 kernels) + Max Pooling + Relu -> Dense (1 unit) + Sigmoid':
            >>>>>> > 3ab9be5a0e89e17043c9d1df756f03b0d458ce83
            {
                'type': 'ml',
                'name': 'imagene',
                'image_height': 128,
                'image_width': width,
                'relu': True,
                'max_pooling': True,
                'convolution': True,
                'filters': 16,
                'depth': 1,
                'kernel_height': 3,
                'kernel_width': 3,
                'num_dense_layers': 0
            },

            << << << < HEAD
            '1 Conv Layers (4 3x3 kernels) + Max Pooling + Relu -> Dense (1 unit) + Sigmoid':
            == == == =
            '1 Conv Layers (4 3x3 kernels) + Max Pooling + Relu -> Dense (1 unit) + Sigmoid':
            >>>>>> > 3ab9be5a0e89e17043c9d1df756f03b0d458ce83
            {
                'type': 'ml',
                'name': 'imagene',
                'image_height': 128,
                'image_width': width,
                'relu': True,
                'max_pooling': True,
                'convolution': True,
                'filters': 4,
                'depth': 1,
                'kernel_height': 3,
                'kernel_width': 3,
                'num_dense_layers': 0
            },

            << << << < HEAD
            '1 Conv Layers (4 3x1 kernels) + Max Pooling + Relu -> Dense (1 unit) + Sigmoid':
            {
                'type': 'ml',
                'name': 'imagene',
                'image_height': 128,
                'image_width': width,
                'relu': True,
                'max_pooling': True,
                'convolution': True,
                'filters': 4,
                'depth': 1,
                'kernel_height': 3,
                'kernel_width': 1,
                'num_dense_layers': 0
            },

            '1 Conv Layers (1 3x3 kernel) + Max Pooling + Relu -> Dense (1 unit) + Sigmoid':
            == == == =
            '1 Conv Layers (1 3x3 kernel) + Max Pooling + Relu -> Dense (1 unit) + Sigmoid':
            >>>>>> > 3ab9be5a0e89e17043c9d1df756f03b0d458ce83
            {
                'type': 'ml',
                'name': 'imagene',
                'image_height': 128,
                'image_width': width,
                'relu': True,
                'max_pooling': True,
                'convolution': True,
                'filters': 1,
                'depth': 1,
                'kernel_height': 3,
                'kernel_width': 3,
                'num_dense_layers': 0
            },

            << << << < HEAD
            '1 Conv Layers (1 2x2 kernel) + Max Pooling + Relu -> Dense (1 unit) + Sigmoid':
            == == == =
            '1 Conv Layers (1 2x2 kernel) + Max Pooling + Relu -> Dense (1 unit) + Sigmoid':
            >>>>>> > 3ab9be5a0e89e17043c9d1df756f03b0d458ce83
            {
                'type': 'ml',
                'name': 'imagene',
                'image_height': 128,
                'image_width': width,
                'relu': True,
                'max_pooling': True,
                'convolution': True,
                'filters': 1,
                'depth': 1,
                'kernel_height': 2,
                'kernel_width': 2,
                'num_dense_layers': 0
            },

            << << << < HEAD
            '1 Conv Layers (1 1x2 kernel) + Max Pooling + Relu -> Dense (1 unit) + Sigmoid':
            == == == =
            '1 Conv Layers (1 1x2 kernel) + Max Pooling + Relu -> Dense (1 unit) + Sigmoid':
            >>>>>> > 3ab9be5a0e89e17043c9d1df756f03b0d458ce83
            {
                'type': 'ml',
                'name': 'imagene',
                'image_height': 128,
                'image_width': width,
                'relu': True,
                'max_pooling': True,
                'convolution': True,
                'filters': 1,
                'depth': 1,
                'kernel_height': 1,
                'kernel_width': 2,
                'num_dense_layers': 0
            },

            << << << < HEAD
            '1 Conv Layers (1 2x1 kernel) + Max Pooling + Relu -> Dense (1 unit) + Sigmoid':
            == == == =
            '1 Conv Layers (1 2x1 kernel) + Max Pooling + Relu -> Dense (1 unit) + Sigmoid':
            >>>>>> > 3ab9be5a0e89e17043c9d1df756f03b0d458ce83
            {
                'type': 'ml',
                'name': 'imagene',
                'image_height': 128,
                'image_width': width,
                'relu': True,
                'max_pooling': True,
                'convolution': True,
                'filters': 1,
                'depth': 1,
                'kernel_height': 2,
                'kernel_width': 1,
                'num_dense_layers': 0
            },

            << << << < HEAD
            '1 Conv Layers (1 2x1 kernel) + Relu -> Dense (1 unit) + Sigmoid':
            == == == =
            '1 Conv Layers (1 2x1 kernel) + Relu -> Dense (1 unit) + Sigmoid':
            >>>>>> > 3ab9be5a0e89e17043c9d1df756f03b0d458ce83
            {
                'type': 'ml',
                'name': 'imagene',
                'image_height': 128,
                'image_width': width,
                'relu': True,
                'max_pooling': False,
                'convolution': True,
                'filters': 1,
                'depth': 1,
                'kernel_height': 2,
                'kernel_width': 1,
                'num_dense_layers': 0
            },

            << << << < HEAD
            '1 Conv Layers (1 2x1 kernel) + Max Pooling-> Dense (1 unit) + Sigmoid':
            == == == =
            '1 Conv Layers (1 2x1 kernel) + Max Pooling-> Dense (1 unit) + Sigmoid':
            >>>>>> > 3ab9be5a0e89e17043c9d1df756f03b0d458ce83
            {
                'type': 'ml',
                'name': 'imagene',
                'image_height': 128,
                'image_width': width,
                'relu': False,
                'max_pooling': True,
                'convolution': True,
                'filters': 1,
                'depth': 1,
                'kernel_height': 2,
                'kernel_width': 1,
                'num_dense_layers': 0
            },

            << << << < HEAD
            '1 Conv Layers (1 2x1 kernel) -> Dense (1 unit) + Sigmoid':
            == == == =
            '1 Conv Layers (1 2x1 kernel) -> Dense (1 unit) + Sigmoid':
            >>>>>> > 3ab9be5a0e89e17043c9d1df756f03b0d458ce83
            {
                'type': 'ml',
                'name': 'imagene',
                'image_height': 128,
                'image_width': width,
                'relu': False,
                'max_pooling': False,
                'convolution': True,
                'filters': 1,
                'depth': 1,
                'kernel_height': 2,
                'kernel_width': 1,
                'num_dense_layers': 0
            },

            << << << < HEAD
            'Dense (1 unit) + Sigmoid':
            == == == =
            'Dense (1 unit) + Sigmoid':
            >>>>>> > 3ab9be5a0e89e17043c9d1df756f03b0d458ce83
            {
                'type': 'ml',
                'name': 'imagene',
                'image_height': 128,
                'image_width': width,
                'relu': False,
                'max_pooling': False,
                'convolution': False,
                'filters': 1,
                'depth': 1,
                'kernel_height': 2,
                'kernel_width': 1,
                'num_dense_layers': 0
            }}


# Defines models and statistics to be compared and to test for sample complexity
##############################################
def ml_stat_comparison_model_configs(width: int) -> Dict:
    """Returns all ml model configs to be tested

    Parameters
    ----------
    width: (int) - Width of converted images

    Returns
    -------
    Dict: Key is name of architecture while value stores ML archiecture config

    """
    return {
        'Imagene': {
            << << << < HEAD
            'type': 'ml',
            'name': 'imagene',
            'image_height': 128,
            'image_width': width,
            'relu': True,
            'max_pooling': True,
            'convolution': True,
            'filters': 32,
            'depth': 3,
            'kernel_height': 3,
            'kernel_width': 3,
            'num_dense_layers': 1
        },

        'Mini-CNN': {
            == == == =
            'type': 'ml',
            'name': 'imagene',
            'image_height': 128,
            'image_width': width,
            'relu': True,
            'max_pooling': True,
            'convolution': True,
            'filters': 32,
            'depth': 3,
            'kernel_height': 3,
            'kernel_width': 3,
            'num_dense_layers': 1
        },

        'R-Imagene': {
            >>>>>> > 3ab9be5a0e89e17043c9d1df756f03b0d458ce83
            'type': 'ml',
            'name': 'imagene',
            'image_height': 128,
            'image_width': width,
            'relu': True,
            'max_pooling': False,
            'convolution': True,
            'filters': 1,
            'depth': 1,
            'kernel_height': 2,
            'kernel_width': 1,
            'num_dense_layers': 0
        },

        'DeepSet': {
            'type': 'ml',
            'name': 'deepset',
            'filters': 64,
            'image_height': 128,
            'image_width': width,
            'kernel_size': 5,
            'depth': 2,
            'num_dense_layers': 2,
            'num_units': 64
        }
    }


def stat_comparison_configs() -> Dict:
    """Returns all stat model configs to be tested

        Returns
        -------
        Dict: Key is name of statistic while value stores model config

        """
    return {
        "Garud's H": {'type': 'statistic',
                      'name': 'garud_h1'},
        "Tajima's D": {'type': 'statistic',
                       'name': 'tajima_d'},
        "IHS": {'type': 'statistic',
                'name': 'ihs_maxabs',
                'standardized': True},
        "NCol": {'type': 'statistic',
                 'name': 'n_columns'}
    }


def stat_comparison_training_settings() -> Dict:
    """
    Returns
    -------
    Dict: Settings for training statistic
    """
    return {
        'train_proportion': 0.8,
        'validate_proportion': 0.1,
        'test_proportion': 0.1}


def stat_comparison_conversion_config(stat_name: str, pop: int = None) -> List:
    """
    Returns
    -------
    List: Settings for converting data for statistic
    """
    if stat_name in all_image_and_position_statistics():
        if pop is None:
            return [{'conversion_type': 'raw_data',
                     'datatype': 'popgen_image'},
                    {'conversion_type': 'raw_data',
                     'datatype': 'popgen_positions'}]
        else:
            return [{'conversion_type': 'raw_data',
                     'datatype': 'popgen_image',
                     'pop': pop},
                    {'conversion_type': 'raw_data',
                     'datatype': 'popgen_positions',
                     'pop': pop}]
    else:
        if pop is None:
            return [{'conversion_type': 'raw_data',
                     'datatype': 'popgen_image'}]
        else:
            return [{'conversion_type': 'raw_data',
                     'datatype': 'popgen_image',
                     'pop': pop}]


# Defines models to visualize
##############################################
def ml_models_to_visualize(width: int) -> Dict:
    """Returns all ml model configs to be visualized

    Parameters
    ----------
    width: (int) - Width of converted images

    Returns
    -------
    Dict: Key is name of architecture while value stores ML archiecture config

    """
    return {'R-Imagene': {
        'type': 'ml',
        'name': 'imagene',
        'image_height': 128,
        'image_width': width,
        'relu': True,
        'max_pooling': False,
        'convolution': True,
        'filters': 1,
        'depth': 1,
        'kernel_height': 2,
        'kernel_width': 1,
        'num_dense_layers': 0
    },
    }
