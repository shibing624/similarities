# -*- coding: utf-8 -*-
import sys

from setuptools import setup, find_packages

# Avoids IDE errors, but actual version is read from version.py
__version__ = None
exec(open('similarities/version.py').read())

if sys.version_info < (3,):
    sys.exit('Sorry, Python3 is required.')

with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()

setup(
    name='similarities',
    version=__version__,
    description='Similarities is a toolkit for compute similarity scores between two sets of strings.',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='XuMing',
    author_email='xuming624@qq.com',
    url='https://github.com/shibing624/similarities',
    license="Apache License 2.0",
    zip_safe=False,
    python_requires=">=3.6.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords='similarities,Chinese Text Similarity Calculation Tool,similarity,word2vec',
    install_requires=[
        "text2vec>=1.1.5",
        "transformers>=4.6.0",
        "jieba>=0.39",
        "loguru",
        "Pillow",
        'pytrec_eval',
        "hnswlib",
        # "opencv-python",
        # "annoy",
    ],
    packages=find_packages(),
)
