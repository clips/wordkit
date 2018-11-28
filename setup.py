# -*- coding: utf-8 -*-
"""Setup file."""
from setuptools import setup
from setuptools import find_packages
import re

VERSIONFILE = "wordkit/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
mo = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", verstrline, re.M)

if mo:
    version_string = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

setup(name='wordkit',
      version=version_string,
      description='Word featurization',
      author='Stéphan Tulkens',
      author_email='stephan.tulkens@uantwerpen.be',
      url='https://github.com/clips/wordkit',
      license='GPLv3',
      packages=find_packages(exclude=['examples', 'images']),
      install_requires=['numpy',
                        'ipapy',
                        'pandas',
                        'scikit_learn'],
      classifiers=[
          'Intended Audience :: Developers',
          'Programming Language :: Python :: 3'],
      keywords='machine learning',
      zip_safe=True,
      python_requires='>=3')
