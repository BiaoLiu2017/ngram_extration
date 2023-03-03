# -*- coding:utf-8 -*-
"""
-------------------------------------------------
   File Name：     setup
   Author :        Biao Liu
   Create Data：    2023/3/3 11:05
   Description :
-------------------------------------------------
"""

import setuptools

# 读取项目的readme介绍
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ngram_extration",# 项目名称，保证它的唯一性，不要跟已存在的包名冲突即可
    version="0.0.4",
    author="Biao Liu", # 项目作者
    author_email="liubiao2017@gmail.com",
    description="用于提取语料库的最大词频情况下的最长字符串，即用于挖掘语料库中的高频长字符串", # 项目的一句话描述
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BiaoLiu2017/ngram_extration",# 项目地址
    packages=setuptools.find_packages(),
    install_requires=[
        'scikit-learn',
        'numpy',
        'tqdm'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)