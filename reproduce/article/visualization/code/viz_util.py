import os


def save_full_filename(filename: str):
    """
    Parameters
    ----------
    filename - (str): Filname of plot

    Returns
    -------
    (str): Full path with filename to plots directory

    """
    return os.path.join(os.getcwd(), 'reproduce/article/visualization/plots', filename)
