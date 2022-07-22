from reproduce.article.visualization.code.viz_util import save_full_filename
from util.util import save_grey_image
import numpy as np
from imageio import imread
import os

if __name__ == '__main__':

    # Load binary image
    image = imread(os.path.join(os.getcwd(), 'reproduce/article/visualization/plots/example_neutral.png'))
    image = np.asarray(image / 255, dtype=np.int)

    # Take cross section
    x = image[40:60, 0:40]

    # Save binary image
    save_grey_image(x,
                    save_full_filename('general_image.png'),
                    colorbar=False,
                    xticks = [1,10,20,30],
                    yticks=[1,5,10,15])

    # Sort rows of image
    uniques, counts = np.unique(x, return_counts=True, axis=0)
    counter = 0
    for j in counts.argsort()[::-1]:
        for z in range(counts[j]):
            x[counter, :] = uniques[j, :]
            counter += 1

    # Save sorted image
    save_grey_image(x,
                    save_full_filename('general_image_sorted.png'),
                    colorbar=False,
                    xticks=[1, 10, 20, 30],
                    yticks=[1, 5, 10, 15])
