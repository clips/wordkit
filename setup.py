# -*- coding: utf-8 -*-
"""Setup file."""

from setuptools import setup
from setuptools import find_packages


setup(name='wordkit',
      version='1.0.0',
      description='Word featurization',
      author='Stéphan Tulkens',
      author_email='stephan.tulkens@uantwerpen.be',
      url='https://github.com/clips/wordkit',
      license='GPLv3',
      packages=find_packages(exclude=['examples']),
      install_requires=['numpy>=1.11.0',
                        'ipapy',
                        'pandas',
                        'scikit_learn'],
      classifiers=[
          'Intended Audience :: Developers',
          'Programming Language :: Python :: 3',],
      keywords='machine learning',
      zip_safe=True)
