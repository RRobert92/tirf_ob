from setuptools import setup, find_packages
from tirf_ob.version import version

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['click==8.0.1',
                'numpy==1.21.0',
                'tifffile==2021.7.2'
                ]

setup(
    author="Robert Kiewisz",
    author_email='robert.kiewisz@gmail.com',
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8',
    ],
    description="Croping tool for TIRF images",
    long_description=readme,
    entry_points={
        'console_scripts': [
            'crop_trif=tirf_ob.select_from_ROI:main',
        ],
    },
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description_content_type='text/x-rst',
    include_package_data=True,
    keywords='tirf_ob',
    name='tirf_ob',
    packages=find_packages(include=['tirf_ob',
                                    'tirf_ob.*',
                                    'tirf_ob.utils.*']),
    version=version,
    zip_safe=False,
)
