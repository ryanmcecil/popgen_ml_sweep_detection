#### MSMS

1. **imagene_results.py:** Reproduces the results of original Imagene model on test set from Imagene paper. 
Confusion matrices are used to visualize the results. Note that results may not be up to par in confusion matrices due
to the randomness of training and the size of the model.

2. **tiny_imagene_results.py:**  Reproduces the results of 'tiny' Imagene model on test set from Imagene paper. 
Confusion matrices are used to visualize the results.

3. **tiny_imagene_stat_comparison.py:**  Compares the 'tiny' Imagene model to summary statistics by creating
a correlation matrix depicting the correlations between the model's predictions and the computed statistics on the
msms test set. In addition, creates a csv table with the accuracies of the different methods.

4. **tiny_imagene_visualize.py:** Takes a neutral and sweep image from the test set, and then passes them through
the 'tiny' Imagene model. The images and the layer outputs are then visualized and saved.

6. **reproduce.sh**: Reproduces all of the results from the above py files.