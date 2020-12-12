# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 21:52:31 2020

@author: Okuda
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tfaug",
    version="0.0.2",
    author="piyop",
    author_email="t.okuda@keio.com",
    description="tensorflow easy image augmantation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/piyop/tfaug",
    packages=setuptools.find_packages(),
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "tensorflow >= 2.0",
        "tensorflow-addons >= 0.7.1",   
        ],
    python_requires='>=3.5',
)