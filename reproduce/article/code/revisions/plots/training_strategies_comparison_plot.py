
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

save_folder = os.path.join(
    os.getcwd(), 'reproduce/article/results/revisions')

# Early stopping performance bar plot and values
############################################################################
data = pd.read_csv(os.path.join(
    save_folder,  'training_strategies_comparison.csv'))

# Group the data by SELCOEFF
grouped_data_selcoeff = data.groupby('SELCOEFF')

# Iterate over each SELCOEFF value
for selcoeff, selcoeff_data in grouped_data_selcoeff:
    # Group the data by Simulation Type
    grouped_data_simtype = selcoeff_data.groupby('Simulation Type')

    # Create a new figure for the current SELCOEFF value
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    # Set the width of each bar
    bar_width = 0.1

    # Set the spacing between each group of bars
    spacing = 0.05

    # Iterate over each group and create the grouped sub bar plots
    for i, ((simulation_type, group_data), ax) in enumerate(zip(grouped_data_simtype, axs.flatten())):
        # Calculate the x position of the bars for the current group
        x = np.arange(group_data['Training Strategy'].nunique())

        # Create an array of indices for each Model
        model_indices = np.arange(len(group_data['Model'].unique()))

        # Iterate over each group and create the grouped sub bar plots
        for i, (training_strategy, subgroup_data) in enumerate(group_data.groupby('Training Strategy')):

            # Calculate the x position of the bars for the current group
            x = model_indices + (bar_width + spacing) * i

            # Create the bars for the current group
            ax.bar(x, subgroup_data['Accuracy'],
                   width=bar_width, label=training_strategy)

        # Set the x-axis tick labels to the Model values
        ax.set_xticks(model_indices + (bar_width + spacing) *
                      (len(group_data.groupby('Training Strategy')) - 1) / 2)
        ax.set_xticklabels(group_data['Model'].unique())
        # Set the labels for the y-axis and x-axis of each subplot
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Model')

        # Set the title for the subplot
        ax.set_title(f'{simulation_type}')

    # Add more vertical space between subplots
    plt.subplots_adjust(hspace=0.4)

    # Add a title to the full figure with the value of the selection coefficient
    fig.suptitle(
        f'Training Strategy Comparison\n\n s = {selcoeff}', fontsize=16)

    # Create an empty subplot for the legend
    legend_ax = fig.add_subplot(111)
    legend_ax.axis('off')

    # Generate the legend for all subplots
    handles, labels = axs[0, 0].get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc='center')

    # Show the plot
    plt.savefig(os.path.join(
        save_folder, f'training_strategy_performance_comparison_s={selcoeff}.png'))

# Compute range of differences and save to csv file
grouped_data = data.groupby('Model')
ranges = pd.DataFrame(columns=[
    'Model', 'Range Early Stopping - Best of 10', 'Range Simulation on the Fly - Best of 10'])

# Iterate over each model and compute the ranges
for model, model_data in grouped_data:

    diff_early_stopping = np.asarray(model_data[model_data['Training Strategy'] == 'Early Stopping']['Accuracy']) - \
        np.asarray(
            model_data[model_data['Training Strategy'] == 'Best Of 10']['Accuracy'])
    diff_simulation = np.asarray(model_data[model_data['Training Strategy'] == 'Simulation on the Fly']['Accuracy']) - \
        np.asarray(
        model_data[model_data['Training Strategy'] == 'Best Of 10']['Accuracy'])
    ranges = ranges.append({
        'Model': model,
        'Range Early Stopping - Best of 10': (diff_early_stopping.min(), diff_early_stopping.max()),
        'Range Simulation on the Fly - Best of 10': (diff_simulation.min(), diff_simulation.max())
    }, ignore_index=True)

# Save the ranges
ranges.to_csv(os.path.join(
    save_folder, 'training_strategy_ranges.csv'), index=False)
