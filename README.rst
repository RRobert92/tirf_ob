================================
TIRF Object Cropping [tirf_ob]
================================

.. image:: https://img.shields.io/github/v/release/RRobert92/tirf_ob
        :target: https://img.shields.io/github/v/release/SMLC-NYSBC/Semantic_Label_Creator

.. image:: https://github.com/RRobert92/tirf_ob/actions/workflows/python-publish_PyPi.yml/badge.svg
        :target: https://github.com/SMLC-NYSBC/Semantic_Label_Creator/actions/workflows/python-publish_PyPi.yml

.. image:: https://readthedocs.org/projects/tirf_ob/badge/?version=latest
        :target: https://semantic-label-creator.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

Python package for cropping images based on input coordinates.

* Documentation: https://semantic-label-creator.readthedocs.io/en/latest/

============
Installation
============


Stable release
--------------

To install Semantic_Label_Creator, run this command in your terminal:

.. code-block:: console

    $ pip install tirf_ob

This is the preferred method to install tirf_ob , as it will always install the most recent stable release.

From sources
------------

The sources for tirf_ob can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/RRobert92/tirf_ob
    $ python setup.py install

or install is with pip:

.. code-block:: console

    $ pip install tirf_ob


.. _Github repo: https://github.com/RRobert92/tirf_ob
.. _tarball: https://github.com/RRobert92/tirf_ob/tarball/master

=====
Usage
=====

To use tirf_ob with terminal::

    crop_tirf -dir_img C:/... -dir_csv C:/... -o C:/.../output -m 256

 string [-dir_img] Directory of the image that should be cropped.
    [-default] None
 string [-dir_csv] Directory of the .csv file with coordinates.
    [-default] None
 string [-o]   Output directory to the folder where all of cropped image is stored.
    [-default] None
 int    [-m]  Size of corpping area in pixels.
    [-default] 256