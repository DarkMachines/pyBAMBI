#!/usr/bin/env python3
from setuptools import setup, find_packages


def readme():
    with open('README.rst') as f:
        return f.read()


def get_version(short=False):
    with open('README.rst') as f:
        for line in f:
            if ':Version:' in line:
                ver = line.split(':')[2].strip()
                if short:
                    subver = ver.split('.')
                    return '%s.%s' % tuple(subver[:2])
                else:
                    return ver


setup(name='pyBAMBI',
      version=get_version(),
      description='pyBAMBI: Resurrecting BAMBI for the pythonic deep learning era ',
      long_description=readme(),
      author='Dark Machines collaboration',
      author_email='wh260@cam.ac.uk',
      url='https://github.com/williamjameshandley/pyBAMBI',
      packages=find_packages(),
      install_requires=['pymultinest','pypolychord'],
      setup_requires=[],
      extras_require={
          'docs': ['sphinx', 'sphinx_rtd_theme', 'numpydoc'],
          },
      tests_require=['pytest'],
      include_package_data=True,
      classifiers=[
                   'Development Status :: 1 - Planning',
                   'Intended Audience :: Developers',
                   'Intended Audience :: Science/Research',
                   'Natural Language :: English',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3.4',
                   'Programming Language :: Python :: 3.5',
                   'Programming Language :: Python :: 3.6',
                   'Programming Language :: Python :: 3.7',
                   'Topic :: Scientific/Engineering',
                   'Topic :: Scientific/Engineering :: Physics',
                   'Topic :: Scientific/Engineering :: Astronomy',
                   'Topic :: Scientific/Engineering :: Mathematics',
      ],
      )
