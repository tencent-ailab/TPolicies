from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import setup


setup(
    name='TPolicies',
    version='1.0',
    description='Tencent Policies Library for Various IL & RL Applications',
    keywords='Policies',
    packages=[
      'tpolicies',
    ],
    install_requires=[
      # 'tensorflow',
      'numpy',
      'joblib',
    ]
)
