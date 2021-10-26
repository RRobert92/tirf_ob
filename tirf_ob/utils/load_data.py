import numpy as np
from tifffile import tifffile


def load_image_to_numpy(dir: str):
    """
    Module to load 3D .tif file
    with the format [D x H x W]

    Args:
        dir:
    """

    try:
        image = tifffile.imread(dir)
    except RuntimeWarning:
        raise Warning("Directory for the file is not correct, file not found!")

    return image


def load_csv_to_numpy(dir: str):
    """
    Module to load csv files.
    Args:
         dir:
    """
    csv = np.genfromtxt(dir, delimiter=',')
    csv = csv[1:len(csv), 2:5].astype('uint16')

    return csv
