
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Creates bar plots to visualize model performances from S2 table
#############################################

save_folder = os.path.join(
    os.getcwd(), 'reproduce/article/results/revisions')


def get_interval(accuracy, simulation_type):
    if 'Three Pop' in simulation_type:
        return 1.96*np.sqrt(accuracy*(1-accuracy)/2000)
    else:
        return 1.96*np.sqrt(accuracy*(1-accuracy)/10000)


model_order = ['Imagene', 'Mini-CNN', 'DeepSet', 'Model D']

# Load the data
data = pd.read_csv(os.path.join(save_folder, 'model_performances.csv'))

model_mapping = {
    'Imagene': 'Imagene',
    'Mini-CNN': '\nMini-CNN',
    'DeepSet': 'DeepSet',
    "Garud's H": "\nGarud's H",
    "Tajima's D": "Tajima's D",
    "IHS": "\nIHS",
}


# Group the data by SELCOEFF
grouped_data_selcoeff = data.groupby('SELCOEFF')

for selcoeff, selcoeff_data in grouped_data_selcoeff:
    # Group the data by Simulation Type
    grouped_data_simtype = selcoeff_data.groupby('Simulation Type')

    # Create a new figure for the current SELCOEFF value
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    # Iterate over each group and create the subplots
    for i, ((simulation_type, group_data), ax) in enumerate(zip(grouped_data_simtype, axs.flatten())):
        # Group the data by Model
        grouped_data_model = group_data.groupby('Model')

        # Get the unique models and their corresponding accuracies
        models = grouped_data_model['Model'].unique()
        models = [model[0] for model in models]  # Remove the brackets

        accuracies = grouped_data_model['Accuracy'].mean()
        models, accuracies = zip(
            *sorted(zip(models, accuracies), key=lambda x: list(model_mapping.keys()).index(x[0])))

        # Map the model names to new names
        models = [model_mapping[model]
                  if model in model_mapping else model for model in models]

        # Get the confidence intervals for accuracies
        intervals = [get_interval(accuracy, simulation_type)
                     for accuracy in accuracies]

        # Set the x positions of the bars
        x = np.arange(len(models))

        # Plot the bar chart with error bars
        ax.bar(x, accuracies, align='center')
        ax.errorbar(x, accuracies, yerr=intervals,
                    fmt='none', capsize=5, color='black')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Model Type')
        ax.set_title(simulation_type)

    # Add more vertical space between subplots
    plt.subplots_adjust(hspace=0.4)

    # Add a title to the full figure with the value of the selection coefficient
    fig.suptitle(f'Performance Comparison\n\n s = {selcoeff}', fontsize=16)

    # Show the figures
    plt.savefig(os.path.join(
        save_folder, f'performance_comparisons_s={selcoeff}.png'))
