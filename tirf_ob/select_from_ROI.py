from os import mkdir, listdir, getcwd
from os.path import isdir, join
from typing import Optional

import click
import tifffile

from tirf_ob.version import version
from tirf_ob.utils.interpolate import interpolation_3D
from tirf_ob.utils.load_data import *


@click.command()
@click.option('-dir', '--dir',
              default=getcwd(),
              help='Directory to the image in *.tif format.',
              show_default=True)
@click.option('-o', '--output',
              default=join(getcwd(), 'output'),
              help='Output directory for saving processed data.',
              show_default=True)
@click.option('-m', '--mask_size',
              default=256,
              help='Size of the cropping mask',
              show_default=True)
@click.option('-px', '--pixel_size',
              default=1.0,
              help='Size of the cropping mask',
              show_default=True)
@click.version_option(version=version)
def main(dir: str,
         mask_size: int,
         pixel_size: float,
         output: Optional[str] = None):
    """ Load data """
    img_list = listdir()
    if output is None:
        output = join(dir, 'output')

    if not isdir(output):
        mkdir(output)

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

            coord = load_csv_to_numpy(csv_dir)  # X Y Z
            coord[:, :2] = coord[:, :2] / pixel_size
            """ Interpolate coordinate"""
            coord = interpolation_3D(coord.astype(np.int64))
            coord = coord.astype(np.int64)
            idx = np.ravel_multi_index(coord.reshape(coord.shape[0], - 1).T,
                                       coord.max(0).ravel() + 1)
            coord = coord[np.sort(np.unique(idx, return_index=True)[1])]

            """ Main loop to corp image """
            cropped_image = np.zeros((image_size[0], mask_size, mask_size), dtype=image.dtype)
            for i in range(image_size[0]):
                idx = np.where(coord[:, 2] == (i + 1))[0]

                """ Find row with current Z position """
                if len(idx) == 1:
                    idx = idx
                elif len(idx) == 0:
                    assert i != 0 or i != (image_size[0] - 1), \
                        'For the interpolation first and the last point are needed!'
                    past = np.where(coord[:, 2] == i)[0]
                    feature = np.where(coord[:, 2] == (i + 2))[0]
                    idx = (coord[past, :] + coord[feature, :]) / 2
                else:
                    idx = np.array(int(round(idx.mean(), 0)))

                try:
                    idx = idx[0]
                except IndexError:
                    print(idx)
                    idx = int(idx)

                """ Define cropping area """
                p_coord = coord[idx, :]
                crop_x_start, crop_x_stop = p_coord[1] - (mask_size / 2),  \
                    p_coord[1] + (mask_size / 2)
                crop_y_start, crop_y_stop = p_coord[0] - (mask_size / 2), \
                    p_coord[0] + (mask_size / 2)

                """ Crop image and save """
                if crop_x_start < 0 and crop_y_start < 0:
                    padded_crop = image[i,
                                        0:int(crop_x_stop),
                                        0:int(crop_y_stop)]
                elif crop_x_start < 0 or crop_y_start < 0:
                    if crop_x_start < 0:
                        padded_crop = image[i,
                                            0:int(crop_x_stop),
                                            int(crop_y_start):int(crop_y_stop)]
                    else:
                        padded_crop = image[i,
                                            int(crop_x_start):int(crop_x_stop),
                                            0:int(crop_y_stop)]
                else:
                    padded_crop = image[i,
                                        int(crop_x_start):int(crop_x_stop),
                                        int(crop_y_start):int(crop_y_stop)]

                if padded_crop.shape == (mask_size, mask_size):
                    cropped_image[i, :] = padded_crop
                else:
                    pad = np.zeros((mask_size, mask_size), dtype=image.dtype)
                    pad_size = padded_crop.shape
                    shift = (mask_size / 2)
                    pad_x_start = int(shift - pad_size[0] / 2)
                    pad_x_stop = int(pad_x_start + pad_size[0])
                    pad_y_start = int(shift - pad_size[1] / 2)
                    pad_y_stop = int(pad_y_start + pad_size[1])

                    pad[pad_x_start:pad_x_stop, pad_y_start:pad_y_stop] = padded_crop
                    cropped_image[i, :] = pad

            tifffile.imwrite(join(output, name + '_trim.tif'),
                             cropped_image)


if __name__ == '__main__':
    main()
