# -*- coding: utf-8 -*-
"""Setup file."""

from setuptools import setup
from setuptools import find_packages


setup(name='wordkit',
      version='0.1.0',
      description='Word featurization',
      author='Stéphan Tulkens',
      author_email='stephan.tulkens@uantwerpen.be',
      url='https://github.com/stephantul/wordkit',
      license='MIT',
      packages=find_packages(exclude=['examples']),
      install_requires=['numpy>=1.11.0'],
      classifiers=[
          'Intended Audience :: Developers',
          'Programming Language :: Python :: 3',],
      keywords='machine learning',
      zip_safe=True)
