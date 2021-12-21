import os
from os.path import join
from typing import Optional

import click
import tifffile

from tirf_ob import version
from tirf_ob.utils.interpolate import interpolation_3D
from tirf_ob.utils.load_data import *


@click.command()
@click.option('-dir', '--dir',
              default=os.getcwd(),
              help='Directory to the image in *.tif format.',
              show_default=True)
@click.option('-o', '--output',
              default=join(os.getcwd(), 'output'),
              help='Output directory for saving processed data.',
              show_default=True)
@click.option('-m', '--mask_size',
              default=256,
              help='Size of the cropping mask',
              show_default=True)
@click.version_option(version=version)
def main(dir: str,
         mask_size: int,
         output: Optional[str] = None):
    """ Load data """
    img_list = os.listdir()
    if output is None:
        output = join(dir, 'output')

    for img_dir in img_list:

        if img_dir[-3:] == 'tif':
            csv_dir = img_dir[:-3] + 'csv'
            name = img_dir[:-4]
        elif img_dir[-4:] == 'tiff':
            csv_dir = img_dir[:-4] + 'csv'
            name = img_dir[:-5]
        else:
            csv_dir = None

        if csv_dir is not None:
            image = load_image_to_numpy(join(dir, img_dir))
            image_size = image.shape

            coord = load_csv_to_numpy(csv_dir)

            """ Interpolate coordinate"""
            coord = interpolation_3D(coord)
            idx = np.ravel_multi_index(coord.reshape(coord.shape[0], - 1).T,
                                       coord.max(0).ravel() + 1)
            coord = coord[np.sort(np.unique(idx, return_index=True)[1])]

            """ Main loop to corp image """
            corpped_image = np.zeros((image_size[0], mask_size, mask_size))
            for i in range(image_size[0]):
                idx = np.where(coord[:, 2] == (i + 1))[0]

                """ Find row with current Z position """
                if len(idx) == 1:
                    idx = idx
                elif len(idx) == 0:
                    assert i != 0 or i != (
                        image_size[0] - 1), 'For the interpolation first and the last point are needed!'
                    past = np.where(coord[:, 2] == i)[0]
                    feature = np.where(coord[:, 2] == (i + 2))[0]
                    idx = (coord[past, :] + coord[feature, :]) / 2
                else:
                    idx = np.array(int(round(idx.mean(), 0)))

                """ Define cropping area """
                if idx.size == 1:
                    crop_y_start, crop_y_stop = coord[idx, 0] - \
                        mask_size / 2, coord[idx, 0] + mask_size / 2
                    crop_x_start, crop_x_stop = coord[idx, 1] - \
                        mask_size / 2, coord[idx, 1] + mask_size / 2
                if idx.size == 3:
                    crop_y_start, crop_y_stop = idx[0, 0] - \
                        mask_size / 2, idx[0, 0] + mask_size / 2
                    crop_x_start, crop_x_stop = idx[0, 1] - \
                        mask_size / 2, idx[0, 1] + mask_size / 2

                """ Crop image and save """
                corpped_image[i, :, :] = image[i,
                                               int(crop_x_start):int(crop_x_stop),
                                               int(crop_y_start): int(crop_y_stop)]

                tifffile.imsave(join(output, name + '_trim.tif'),
                                corpped_image)


if __name__ == '__main__':
    main()
