import os
from multiprocessing import Process

from reproduce.article.code.arch_analysis import test_architectures
from reproduce.article.code.configs import ml_sim_conversion_training_configs
from reproduce.article.code.sample_complexity import test_sample_complexity
from reproduce.article.code.stat_comparison import stat_comparison
from reproduce.article.code.visualize import visualize_models
from reproduce.article.code.widths import retrieve_max_width_from_settings

generate_results = True


def experiments():
    """Returns list of experiments to run """
    return [test_architectures, stat_comparison, visualize_models, test_sample_complexity]


def run_process(fnc,
                settings,
                save_dir,
                generate_results):
    p = Process(target=fnc, args=(settings, save_dir, generate_results,))
    p.start()
    p.join()
    p.terminate()


if __name__ == '__main__':
    # Run experiments
    for name, settings in ml_sim_conversion_training_configs().items():

        # If zero padding, get width
        if settings['conversions'][0]['conversion_type'] == 'zero_padding_imagene':
            settings['conversions'][0]['resize_dimensions'] = retrieve_max_width_from_settings(
                settings)

        # Create results directory
        results_dir = os.path.join(os.getcwd(), 'reproduce/article/results')
        if not os.path.isdir(results_dir):
            os.mkdir(results_dir)
        save_dir = os.path.join(os.getcwd(), 'reproduce/article/results', name)
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        # Run experiments
        for exp in experiments():
            # Only test sample complexity for those with 100000 training samples
            if exp == test_sample_complexity:
                if settings['simulations']['neutral'][0]['N'] + settings['simulations']['sweep'][0]['N'] == 100000:
                    run_process(exp, settings, save_dir, generate_results)
            else:
                # Only run other functions for 20000 training samples, unless it is msms data
                if settings['simulations']['neutral'][0]['N'] + settings['simulations']['sweep'][0]['N'] == 20000 or \
                        'imagene' in name:
                    run_process(exp, settings, save_dir, generate_results)
